"""Latency and energy benchmark, against the pipeline this would replace.

Single-shot latency understates the difference between a CPU-bound and a
GPU-bound always-on loop: the two draw from different rails and throttle
differently. So workloads run as SUSTAINED loops while tegrastats samples
VDD_IN, and the headline number is mJ per frame -- the metric that decides
whether a remote node on a power budget can afford to run this at all.

tegrastats is Jetson-only. Without it the power columns read as nan and the
latency numbers remain valid.
"""
from __future__ import annotations

import re
import statistics
import subprocess
import threading
import time

VDD_IN = re.compile(r"VDD_IN (\d+)mW")


class PowerSampler(threading.Thread):
    """Sample whole-module draw in the background while a workload runs."""

    def __init__(self, interval_ms: int = 200):
        super().__init__(daemon=True)
        self.interval_ms = interval_ms
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._proc = None

    def run(self) -> None:
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except (FileNotFoundError, OSError):
            return  # not a Jetson; latency still works
        try:
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                m = VDD_IN.search(line)
                if m:
                    self.samples.append(int(m.group(1)))
        finally:
            if self._proc:
                self._proc.terminate()

    def stop(self) -> None:
        self._stop.set()


def run_workload(name: str, step, duration: float = 20.0, warmup: int = 3) -> dict:
    """Run `step` in a tight loop for `duration` seconds, sampling power."""
    for _ in range(warmup):
        step()

    sampler = PowerSampler()
    sampler.start()
    time.sleep(1.0)

    frames, latencies = 0, []
    end = time.perf_counter() + duration
    while time.perf_counter() < end:
        t0 = time.perf_counter()
        step()
        latencies.append(time.perf_counter() - t0)
        frames += 1

    sampler.stop()
    time.sleep(0.3)

    # Drop the leading samples: they cover the pre-loop window, not the load.
    steady = sampler.samples[max(1, len(sampler.samples) // 5):]
    power_mw = statistics.mean(steady) if steady else float("nan")
    fps = frames / duration
    return {
        "name": name,
        "fps": fps,
        "ms_median": statistics.median(latencies) * 1e3,
        "mw": power_mw,
        "mj_per_frame": power_mw / fps if fps else float("nan"),
        "frames": frames,
    }


def format_table(rows: list[dict], idle_mw: float | None = None) -> str:
    lines = [
        f"{'workload':<34}{'fps':>8}{'ms/frame':>11}{'mW':>10}{'mJ/frame':>12}",
        "-" * 75,
    ]
    for r in rows:
        lines.append(
            f"{r['name']:<34}{r['fps']:>8.2f}{r['ms_median']:>11.1f}"
            f"{r['mw']:>10.0f}{r['mj_per_frame']:>12.0f}"
        )
    if idle_mw is not None:
        lines.append("\nmarginal cost over idle (the number that scales with duty cycle):")
        for r in rows:
            if r["name"].startswith("idle"):
                continue
            marginal = r["mw"] - idle_mw
            lines.append(
                f"  {r['name']:<34}{marginal:>8.0f} mW above idle, "
                f"{marginal / r['fps']:>7.0f} mJ/frame"
            )
    return "\n".join(lines)
