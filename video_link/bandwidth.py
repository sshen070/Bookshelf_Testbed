"""Weighted bandwidth allocation across camera feeds.

Pure functions -- no MQTT, no GStreamer, no I/O. That is deliberate: the policy
is the part worth reasoning about and testing, and it should be possible to do
that without standing up a broker and three Jetsons.

    python3 bandwidth.py        # runs the scenarios at the bottom

THE POLICY
----------
Each camera declares a weight (how much its feed matters right now) and a range
[min_kbps, max_kbps]. The allocator divides a total budget in proportion to
weight, subject to those bounds.

Proportional-by-weight alone is wrong in two ways, and both are handled here:

  * A camera capped at max_kbps gains nothing from extra bitrate -- pushing
    4 Mbps at a static wall is waste. Its surplus is redistributed to cameras
    that can still use it.
  * A camera below min_kbps is not "degraded", it is USELESS -- H.264 at
    200 kbps cannot resolve a hairline crack, so the bits spent on it buy
    nothing at all. Better to run fewer cameras properly than all of them
    uselessly, which is what shedding does when the budget cannot cover floors.

The redistribution is iterative (progressive filling / water-filling): clamp,
freeze the clamped, re-divide the remainder among the rest, repeat. It
terminates because each pass freezes at least one camera or none, and if none
freeze the allocation is already feasible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# The link is never filled to 100%. The encoder's bitrate setting is a target
# its output varies around, not a hard ceiling, and RTP/UDP/IP headers add
# roughly 5-8% on top of the payload. Allocating the whole pipe guarantees
# overshoot, and overshoot on a constrained link means queuing and loss.
DEFAULT_HEADROOM = 0.90


@dataclass
class Camera:
    """One feed's claim on the budget.

    `weight` is relative, not absolute -- only the ratios between cameras
    matter, so a caller can use 1/2/3 or 10/20/30 interchangeably.
    """

    camera_id: str
    weight: float = 1.0
    min_kbps: int = 300
    max_kbps: int = 8000
    active: bool = True

    def __post_init__(self):
        if self.min_kbps > self.max_kbps:
            raise ValueError(f"{self.camera_id}: min_kbps {self.min_kbps} > max_kbps {self.max_kbps}")
        if self.weight < 0:
            raise ValueError(f"{self.camera_id}: negative weight {self.weight}")


@dataclass
class Allocation:
    """The result of one allocation pass."""

    kbps: dict[str, int] = field(default_factory=dict)
    shed: list[str] = field(default_factory=list)
    total_allocated: int = 0
    budget: int = 0
    note: str = ""


def allocate(
    total_kbps: int,
    cameras: list[Camera],
    headroom: float = DEFAULT_HEADROOM,
) -> Allocation:
    """Divide `total_kbps` among `cameras` by weight, respecting their bounds."""
    budget = int(total_kbps * headroom)
    active = [c for c in cameras if c.active and c.weight > 0]
    if not active:
        return Allocation(budget=budget, note="no active cameras")

    # Oversubscribed: the floors alone exceed what we have. Shedding the least
    # important feeds entirely beats starving every feed below usefulness --
    # see the module docstring.
    shed: list[str] = []
    keep = sorted(active, key=lambda c: c.weight, reverse=True)
    while keep and sum(c.min_kbps for c in keep) > budget:
        shed.append(keep.pop().camera_id)

    if not keep:
        return Allocation(
            shed=shed,
            budget=budget,
            note=f"budget {budget}kbps below the cheapest camera's floor",
        )

    # Progressive filling. `frozen` holds cameras pinned at a bound; the rest
    # share whatever budget is left, in proportion to weight.
    frozen: dict[str, int] = {}
    while True:
        free = [c for c in keep if c.camera_id not in frozen]
        if not free:
            break
        remaining = budget - sum(frozen.values())
        weight_sum = sum(c.weight for c in free)
        if weight_sum <= 0:
            break

        newly_frozen = False
        for camera in free:
            share = remaining * camera.weight / weight_sum
            if share < camera.min_kbps:
                frozen[camera.camera_id] = camera.min_kbps
                newly_frozen = True
            elif share > camera.max_kbps:
                frozen[camera.camera_id] = camera.max_kbps
                newly_frozen = True
        if not newly_frozen:
            # Nothing hit a bound, so the proportional split is feasible as-is.
            for camera in free:
                frozen[camera.camera_id] = int(remaining * camera.weight / weight_sum)
            break

    allocation = {c.camera_id: int(frozen.get(c.camera_id, 0)) for c in keep}

    # Integer truncation above leaves a few kbps unspent. Hand the remainder to
    # the highest-weight camera that can still absorb it, so the budget is not
    # quietly under-used.
    leftover = budget - sum(allocation.values())
    if leftover > 0:
        for camera in sorted(keep, key=lambda c: c.weight, reverse=True):
            room = camera.max_kbps - allocation[camera.camera_id]
            if room > 0:
                take = min(room, leftover)
                allocation[camera.camera_id] += take
                leftover -= take
                if leftover <= 0:
                    break

    return Allocation(
        kbps=allocation,
        shed=shed,
        total_allocated=sum(allocation.values()),
        budget=budget,
        note=f"{len(allocation)} active, {len(shed)} shed" if shed else f"{len(allocation)} active",
    )


def event_weight(base_weight: float, seconds_since_event: float | None, boost: float = 4.0,
                 decay_seconds: float = 30.0) -> float:
    """Raise a camera's weight after WAFT reports motion, then decay it back.

    A camera that just saw something is the one worth spending bits on. The
    decay matters as much as the boost: without it, the first camera to ever
    trigger would hold priority forever, and the allocation would reflect
    history rather than what is happening now.
    """
    if seconds_since_event is None or seconds_since_event >= decay_seconds:
        return base_weight
    fraction_elapsed = max(0.0, seconds_since_event) / decay_seconds
    return base_weight * (1.0 + (boost - 1.0) * (1.0 - fraction_elapsed))


# ---- self-test ----------------------------------------------------------


def _show(title: str, result: Allocation) -> None:
    print(f"\n{title}")
    print(f"  budget {result.budget} kbps  ->  allocated {result.total_allocated} kbps  ({result.note})")
    for camera_id, kbps in sorted(result.kbps.items(), key=lambda kv: -kv[1]):
        print(f"    {camera_id:12s} {kbps:6d} kbps")
    for camera_id in result.shed:
        print(f"    {camera_id:12s}    SHED")


def _selftest() -> int:
    failures = 0

    def check(label, condition):
        nonlocal failures
        if not condition:
            failures += 1
            print(f"    FAIL: {label}")
        else:
            print(f"    ok: {label}")

    # 1. Equal weights split evenly.
    cams = [Camera(f"cam{i}", weight=1.0) for i in range(1, 4)]
    r = allocate(10_000, cams)
    _show("equal weights, 10 Mbps", r)
    check("all three within 1 kbps of each other", max(r.kbps.values()) - min(r.kbps.values()) <= 1)
    check("stays inside headroom", r.total_allocated <= 9000)

    # 2. Importance skews the split.
    cams = [Camera("spillway", weight=5.0), Camera("crest", weight=2.0), Camera("access", weight=1.0)]
    r = allocate(10_000, cams)
    _show("weighted 5:2:1, 10 Mbps", r)
    check("spillway gets the most", r.kbps["spillway"] == max(r.kbps.values()))
    check("ratio roughly 5:1", 4.0 < r.kbps["spillway"] / r.kbps["access"] < 6.0)

    # 3. A capped camera hands its surplus to the others.
    cams = [Camera("static", weight=8.0, max_kbps=800), Camera("busy", weight=1.0, max_kbps=8000)]
    r = allocate(10_000, cams)
    _show("high weight but low cap -- surplus must be redistributed", r)
    check("capped camera is at its cap", r.kbps["static"] == 800)
    check("surplus went to the other camera", r.kbps["busy"] > 5000)

    # 4. Oversubscribed: shed the least important rather than starve all.
    cams = [Camera(f"cam{i}", weight=float(6 - i), min_kbps=1000) for i in range(1, 6)]
    r = allocate(3_000, cams)
    _show("5 cameras, 3 Mbps, 1 Mbps floors -- must shed", r)
    check("some cameras were shed", len(r.shed) > 0)
    check("every surviving camera is at or above its floor", all(v >= 1000 for v in r.kbps.values()))
    check("the highest-weight camera survived", "cam1" in r.kbps)
    check("the lowest-weight camera was shed", "cam5" in r.shed)

    # 5. Budget below even one floor.
    r = allocate(100, [Camera("only", min_kbps=1000)])
    _show("budget below the cheapest floor", r)
    check("nothing allocated", r.total_allocated == 0)
    check("the camera is shed, not given an unusable trickle", r.shed == ["only"])

    # 6. Event boost, and its decay.
    print("\nevent weight boost (base 1.0, boost 4x over 30s)")
    for age in (0, 10, 20, 30, None):
        w = event_weight(1.0, age)
        print(f"    {str(age)+'s ago' if age is not None else 'no event':>12}  weight {w:.2f}")
    check("fresh event is boosted 4x", abs(event_weight(1.0, 0) - 4.0) < 0.01)
    check("decays back to base by 30s", abs(event_weight(1.0, 30) - 1.0) < 0.01)
    check("no event means base weight", event_weight(1.0, None) == 1.0)

    # 7. Inactive cameras get nothing.
    cams = [Camera("on", weight=1.0), Camera("off", weight=99.0, active=False)]
    r = allocate(5_000, cams)
    _show("an inactive camera must not hold bandwidth", r)
    check("inactive camera absent from the allocation", "off" not in r.kbps)
    check("active camera got the whole budget", r.kbps["on"] == r.budget)

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{failures} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())