"""Bandwidth allocator -- runs ON THE SERVER, divides the link among cameras.

    python3 allocator.py --broker localhost --total 10000 --http-port 8090

Cameras announce themselves; the allocator divides the budget by weight and
publishes a grant to each. Nothing is configured here in advance -- a camera
exists because it announced, so adding a Jetson needs no change on this side.

    video/<id>/announce   (camera, retained)  bounds and weight
    video/<id>/stats      (camera)            what it is actually sending
    video/<id>/bitrate    (allocator)         its grant, in kbps

The allocation policy itself lives in bandwidth.py, which has no I/O and can be
reasoned about on its own -- run `python3 bandwidth.py` to see it work.

Grants are republished whenever the answer CHANGES, plus a slow heartbeat.
Republishing an unchanged grant every cycle would make every camera's encoder
churn for nothing; never republishing would leave a camera that missed a message
stuck at a stale rate forever.

HTTP control (with --http-port):
    GET  /                      dashboard
    GET  /api/state             allocations and camera state as JSON
    POST /api/total   {"kbps": 5000}                 change the link budget
    POST /api/weight  {"camera_id": "x", "weight": 5} change one camera's priority
    POST /api/event   {"camera_id": "x"}             mark motion; boosts, then decays
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bandwidth import Camera, allocate, event_weight

# A camera that has not been heard from in this long is presumed gone and stops
# holding bandwidth. Without it, one crashed Jetson permanently reserves its
# share and every surviving camera runs degraded.
STALE_AFTER_S = 30.0
HEARTBEAT_S = 30.0


def verify_event(event: dict, camera: dict, now: float, policy: dict) -> tuple[bool, str]:
    """Decide whether an event earns a stream. Pure -- no I/O, no state mutation.

    This is the whole point of the handshake: a camera that streams on its own
    detection floods the link with false positives, and on a constrained link
    the false positives crowd out the real events. Four independent reasons to
    refuse, checked cheapest-first:

      confidence  the detector itself is unsure
      cooldown    this camera just streamed; a swaying branch is not N events
      concurrency more cameras want the link than it can carry at once
      bandwidth   what is left cannot even meet this camera's floor

    Returns (approved, reason) -- the reason is reported to the camera and shown
    on the dashboard, because "denied" with no cause is impossible to debug.
    """
    confidence = event.get("confidence")
    minimum = policy.get("min_confidence", 0.0)
    if minimum > 0 and confidence is not None and confidence < minimum:
        return False, f"confidence {confidence:.2f} below {minimum:.2f}"

    last = camera.get("last_stream_end", 0.0)
    cooldown = policy.get("cooldown_seconds", 0.0)
    if last and now - last < cooldown:
        return False, f"cooldown ({cooldown - (now - last):.0f}s left)"

    if policy.get("active_count", 0) >= policy.get("max_concurrent", 99):
        return False, f"max concurrent streams ({policy['max_concurrent']}) reached"

    if policy.get("available_kbps", 0) < camera.get("min_kbps", 300):
        return False, (f"only {policy.get('available_kbps', 0)}kbps free, "
                       f"needs {camera.get('min_kbps', 300)}")

    return True, "verified"


class Allocator:
    def __init__(self, total_kbps: int, boost: float, decay: float,
                 min_confidence: float = 0.0, cooldown: float = 10.0,
                 max_concurrent: int = 99, stream_seconds: float = 30.0,
                 max_stream_seconds: float = 120.0, trigger_mode: bool = False):
        self.total_kbps = total_kbps
        self.boost = boost
        self.decay = decay
        self.min_confidence = min_confidence
        self.cooldown = cooldown
        self.max_concurrent = max_concurrent
        self.stream_seconds = stream_seconds
        self.max_stream_seconds = max_stream_seconds
        # In trigger mode only cameras with a live streaming window compete for
        # bandwidth. Idle cameras holding a share would defeat the point: the
        # budget exists to be spent on whatever is actually happening.
        self.trigger_mode = trigger_mode
        self.lock = threading.Lock()
        self.cameras: dict[str, dict] = {}
        self.last_grants: dict[str, int] = {}
        self.last_result = None

    # ---- camera bookkeeping ---------------------------------------------

    def announce(self, camera_id: str, payload: dict) -> None:
        with self.lock:
            entry = self.cameras.setdefault(camera_id, {"base_weight": 1.0, "last_event": None})
            entry.update({
                "camera_id": camera_id,
                "base_weight": float(payload.get("weight", entry.get("base_weight", 1.0))),
                "min_kbps": int(payload.get("min_kbps", 300)),
                "max_kbps": int(payload.get("max_kbps", 8000)),
                "resolution": f"{payload.get('width','?')}x{payload.get('height','?')}",
                "fps": payload.get("fps"),
                "last_seen": time.time(),
            })

    def record_stats(self, camera_id: str, payload: dict) -> None:
        with self.lock:
            entry = self.cameras.get(camera_id)
            if entry is None:
                return
            entry["last_seen"] = time.time()
            entry["measured_kbps"] = payload.get("measured_kbps")
            entry["granted_kbps"] = payload.get("granted_kbps")

    def set_weight(self, camera_id: str, weight: float) -> bool:
        with self.lock:
            if camera_id not in self.cameras:
                return False
            self.cameras[camera_id]["base_weight"] = float(weight)
            return True

    def mark_event(self, camera_id: str) -> bool:
        """Flag motion on a camera, raising its weight until the boost decays."""
        with self.lock:
            if camera_id not in self.cameras:
                return False
            self.cameras[camera_id]["last_event"] = time.time()
            return True

    # ---- event verification ---------------------------------------------

    def judge_event(self, camera_id: str, event: dict) -> dict:
        """Verify one event and, if approved, open a streaming window."""
        now = time.time()
        with self.lock:
            camera = self.cameras.get(camera_id)
            if camera is None:
                # Never announced, so its bounds are unknown and it cannot be
                # allocated for. Refusing is safer than inventing defaults.
                return {"approved": False, "reason": "unknown camera; no announce seen"}

            active = [c for c in self.cameras.values()
                      if c.get("streaming_until", 0) > now and c["camera_id"] != camera_id]
            committed = sum(c.get("min_kbps", 300) for c in active)
            policy = {
                "min_confidence": self.min_confidence,
                "cooldown_seconds": self.cooldown,
                "max_concurrent": self.max_concurrent,
                "active_count": len(active),
                "available_kbps": max(0, int(self.total_kbps * 0.9) - committed),
            }
            approved, reason = verify_event(event, camera, now, policy)

            camera["last_decision"] = {"approved": approved, "reason": reason, "at": now}
            camera["events_seen"] = camera.get("events_seen", 0) + 1
            if approved:
                requested = float(event.get("requested_seconds") or self.stream_seconds)
                duration = min(requested, self.max_stream_seconds)
                camera["streaming_until"] = now + duration
                # An approved event is by definition the interesting one, so it
                # also boosts the weight -- the camera that just saw something
                # should out-bid the ones that did not.
                camera["last_event"] = now
                camera["events_approved"] = camera.get("events_approved", 0) + 1
                return {"approved": True, "reason": reason, "duration_seconds": duration}
            return {"approved": False, "reason": reason}

    def end_stream(self, camera_id: str) -> None:
        with self.lock:
            camera = self.cameras.get(camera_id)
            if camera is not None:
                camera["streaming_until"] = 0.0
                camera["last_stream_end"] = time.time()

    # ---- the allocation itself ------------------------------------------

    def compute(self):
        now = time.time()
        with self.lock:
            cams = []
            for entry in self.cameras.values():
                age = now - entry.get("last_seen", 0)
                since_event = (now - entry["last_event"]) if entry.get("last_event") else None
                effective = event_weight(entry["base_weight"], since_event, self.boost, self.decay)
                entry["effective_weight"] = round(effective, 2)
                entry["stale"] = age > STALE_AFTER_S
                entry["streaming"] = entry.get("streaming_until", 0) > now
                # In trigger mode a camera earns bandwidth by streaming, not by
                # existing. Outside it, every live camera competes as before.
                competing = (entry["streaming"] if self.trigger_mode else True)
                cams.append(Camera(
                    camera_id=entry["camera_id"],
                    weight=effective,
                    min_kbps=entry.get("min_kbps", 300),
                    max_kbps=entry.get("max_kbps", 8000),
                    active=not entry["stale"] and competing,
                ))
            total = self.total_kbps
        return allocate(total, cams)

    def state(self) -> dict:
        with self.lock:
            cameras = [dict(v) for v in self.cameras.values()]
            result = self.last_result
        return {
            "total_kbps": self.total_kbps,
            "budget_kbps": result.budget if result else 0,
            "allocated_kbps": result.total_allocated if result else 0,
            "grants": dict(self.last_grants),
            "shed": list(result.shed) if result else [],
            "note": result.note if result else "",
            "cameras": sorted(cameras, key=lambda c: -c.get("effective_weight", 0)),
        }


def run_control_plane(allocator: Allocator, broker: str, port: int, interval: float):
    import paho.mqtt.client as mqtt

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="bandwidth-allocator")

    def on_connect(c, userdata, flags, reason_code, properties=None):
        if reason_code.is_failure:
            print(f"[mqtt] refused: {reason_code}", file=sys.stderr)
            return
        c.subscribe("video/+/announce", qos=1)
        c.subscribe("video/+/stats", qos=0)
        c.subscribe("video/+/event", qos=1)
        c.subscribe("video/+/stream_ended", qos=1)
        print(f"[mqtt] connected to {broker}:{port}, watching video/+/announce and video/+/event",
              flush=True)

    def on_message(c, userdata, msg):
        parts = msg.topic.split("/")
        if len(parts) != 3:
            return
        camera_id, kind = parts[1], parts[2]
        try:
            payload = json.loads(msg.payload)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        if kind == "announce":
            first = camera_id not in allocator.cameras
            allocator.announce(camera_id, payload)
            if first:
                print(f"[mqtt] camera joined: {camera_id} "
                      f"(weight {payload.get('weight')}, "
                      f"{payload.get('min_kbps')}-{payload.get('max_kbps')}kbps)", flush=True)
        elif kind == "stats":
            allocator.record_stats(camera_id, payload)
        elif kind == "event":
            decision = allocator.judge_event(camera_id, payload)
            # Reply on the camera's own topic, echoing the event_id so a camera
            # can tell which of its events this answers.
            c.publish(
                f"video/{camera_id}/stream",
                json.dumps({"event_id": payload.get("event_id"), **decision}),
                qos=1,
            )
            snapshot = payload.get("snapshot_bytes")
            verdict = "APPROVED" if decision["approved"] else "denied"
            print(f"[event] {camera_id} {payload.get('trigger','?')} "
                  f"conf={payload.get('confidence')} "
                  f"snap={snapshot or 0}B -> {verdict}: {decision['reason']}", flush=True)
        elif kind == "stream_ended":
            allocator.end_stream(camera_id)
            print(f"[event] {camera_id} stream ended", flush=True)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port, keepalive=60)
    client.loop_start()

    last_heartbeat = 0.0
    while True:
        result = allocator.compute()
        with allocator.lock:
            allocator.last_result = result

        now = time.time()
        heartbeat_due = now - last_heartbeat > HEARTBEAT_S
        changed = result.kbps != allocator.last_grants

        if changed or heartbeat_due:
            for camera_id, kbps in result.kbps.items():
                client.publish(
                    f"video/{camera_id}/bitrate",
                    json.dumps({"kbps": kbps}),
                    qos=1, retain=True,
                )
            # A shed camera must be told, or it keeps sending at its old rate
            # and the budget the allocator "freed" is still being consumed.
            for camera_id in result.shed:
                client.publish(f"video/{camera_id}/bitrate",
                               json.dumps({"kbps": 0, "shed": True}), qos=1, retain=True)
            if changed:
                summary = "  ".join(f"{k}={v}" for k, v in sorted(result.kbps.items()))
                shed = f"  shed={','.join(result.shed)}" if result.shed else ""
                print(f"[alloc] budget={result.budget}  {summary}{shed}", flush=True)
            allocator.last_grants = dict(result.kbps)
            last_heartbeat = now

        time.sleep(interval)


# ---- http ---------------------------------------------------------------

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Bandwidth allocation</title>
<style>
 :root { color-scheme: dark; }
 body { margin:0; padding:28px 22px; background:#0d0d0d; color:#c3c2b7;
        font-family: system-ui,-apple-system,"Segoe UI",sans-serif; font-size:14px; }
 .wrap { max-width:1000px; margin:0 auto; }
 h1 { font-size:19px; margin:0 0 3px; color:#fff; font-weight:600; }
 .sub { color:#898781; font-size:13px; margin:0 0 22px; }
 .tiles { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:22px; }
 .tile { background:#1a1a19; border:1px solid rgba(255,255,255,.1); border-radius:8px;
         padding:13px 17px; min-width:120px; }
 .tile .l { color:#898781; font-size:12px; }
 .tile .v { font-size:26px; font-weight:600; color:#fff; }
 .card { background:#1a1a19; border:1px solid rgba(255,255,255,.1);
         border-radius:8px; overflow-x:auto; }
 table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
 th,td { text-align:left; padding:9px 13px; white-space:nowrap; }
 th { font-size:11px; text-transform:uppercase; letter-spacing:.04em;
      color:#898781; border-bottom:1px solid #2c2c2a; }
 td { border-bottom:1px solid #2c2c2a; }
 tr:last-child td { border-bottom:none; }
 .id { font-weight:600; color:#fff; }
 .bar { height:6px; background:#2c2c2a; border-radius:3px; overflow:hidden; min-width:110px; }
 .bar span { display:block; height:100%; background:#2a78d6; }
 .shed { color:#d03b3b; font-weight:600; }
 .boost { color:#fab219; }
 .muted { color:#898781; }
 button { background:#1a1a19; color:#c3c2b7; border:1px solid rgba(255,255,255,.18);
          border-radius:5px; padding:4px 9px; cursor:pointer; font-size:12px; }
 button:hover { border-color:#2a78d6; color:#fff; }
 input { background:#0d0d0d; color:#fff; border:1px solid rgba(255,255,255,.18);
         border-radius:5px; padding:5px 8px; width:96px; font-size:13px; }
</style></head><body><div class="wrap">
<h1>Bandwidth allocation</h1>
<p class="sub">Grants recomputed continuously &middot; published over MQTT</p>
<div class="tiles">
  <div class="tile"><div class="l">Link total</div><div class="v"><span id="total">-</span></div></div>
  <div class="tile"><div class="l">Allocated</div><div class="v" id="alloc">-</div></div>
  <div class="tile"><div class="l">Cameras</div><div class="v" id="count">-</div></div>
</div>
<p>Link budget (kbps): <input id="tot" type="number" step="500">
   <button onclick="setTotal()">Apply</button>
   <span class="muted" id="msg"></span></p>
<div class="card"><table>
<thead><tr><th>Camera</th><th>Grant</th><th>Share</th><th>Measured</th>
<th>Weight</th><th>Range</th><th>Priority</th></tr></thead>
<tbody id="rows"></tbody></table></div>
</div>
<script>
async function post(path, body) {
  await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'},
                     body: JSON.stringify(body)});
  refresh();
}
function setTotal() { post('/api/total', {kbps: parseInt(document.getElementById('tot').value)}); }
function bump(id, w) { post('/api/weight', {camera_id:id, weight:w}); }
function event(id) { post('/api/event', {camera_id:id}); }
async function refresh() {
  const s = await (await fetch('/api/state', {cache:'no-store'})).json();
  document.getElementById('total').textContent = s.total_kbps;
  document.getElementById('alloc').textContent = s.allocated_kbps;
  document.getElementById('count').textContent = s.cameras.length;
  const t = document.getElementById('tot');
  if (document.activeElement !== t) t.value = s.total_kbps;
  const max = Math.max(1, ...Object.values(s.grants));
  document.getElementById('rows').innerHTML = s.cameras.map(c => {
    const g = s.grants[c.camera_id];
    const isShed = s.shed.includes(c.camera_id);
    const boosted = c.effective_weight > c.base_weight + 0.01;
    return `<tr>
      <td><span class="id">${c.camera_id}</span>
          ${c.stale ? '<span class="muted"> (stale)</span>' : ''}</td>
      <td>${isShed ? '<span class="shed">SHED</span>' : g + ' kbps'}</td>
      <td><div class="bar"><span style="width:${isShed?0:(100*g/max)}%"></span></div></td>
      <td class="muted">${c.measured_kbps != null ? c.measured_kbps + ' kbps' : '—'}</td>
      <td>${c.effective_weight}${boosted ? ' <span class="boost">&#9650;</span>' : ''}
          <span class="muted">(base ${c.base_weight})</span></td>
      <td class="muted">${c.min_kbps}–${c.max_kbps}</td>
      <td><button onclick="bump('${c.camera_id}',${(c.base_weight*2).toFixed(1)})">2&times;</button>
          <button onclick="bump('${c.camera_id}',1)">reset</button>
          <button onclick="event('${c.camera_id}')">event</button></td>
    </tr>`;
  }).join('');
}
refresh(); setInterval(refresh, 1000);
</script></body></html>"""


def make_handler(allocator: Allocator):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                self._json(200, allocator.state())
            else:
                self.send_error(404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 10000:
                self._json(400, {"error": "bad body"})
                return
            try:
                payload = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as e:
                self._json(400, {"error": str(e)})
                return

            if self.path == "/api/total":
                try:
                    allocator.total_kbps = max(0, int(payload["kbps"]))
                except (KeyError, TypeError, ValueError):
                    self._json(400, {"error": "expected {\"kbps\": int}"})
                    return
                self._json(200, {"ok": True, "total_kbps": allocator.total_kbps})
            elif self.path == "/api/weight":
                ok = allocator.set_weight(payload.get("camera_id", ""), payload.get("weight", 1.0))
                self._json(200 if ok else 404, {"ok": ok})
            elif self.path == "/api/event":
                ok = allocator.mark_event(payload.get("camera_id", ""))
                self._json(200 if ok else 404, {"ok": ok})
            else:
                self.send_error(404)

        def log_message(self, *args):
            pass

    return Handler


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Divide a bandwidth budget across camera feeds by importance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--broker", default="localhost", help="MQTT broker")
    p.add_argument("--broker-port", type=int, default=1883)
    p.add_argument("--total", type=int, default=10000, help="total link budget, kbps")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between allocation passes")
    p.add_argument("--boost", type=float, default=4.0, help="weight multiplier for a camera with a fresh event")
    p.add_argument("--decay", type=float, default=30.0, help="seconds for an event boost to decay away")
    p.add_argument("--http-port", type=int, help="serve the dashboard and control API on this port")

    gate = p.add_argument_group("event verification (for --trigger-mode cameras)")
    gate.add_argument("--trigger-mode", action="store_true",
                      help="only cameras with an approved streaming window get bandwidth")
    gate.add_argument("--min-confidence", type=float, default=0.0,
                      help="reject events the detector itself is less sure of than this")
    gate.add_argument("--cooldown", type=float, default=10.0,
                      help="seconds after a stream ends before the same camera may stream again")
    gate.add_argument("--max-concurrent", type=int, default=99,
                      help="how many cameras may stream at once")
    gate.add_argument("--stream-seconds", type=float, default=30.0,
                      help="default granted window when a camera does not ask for one")
    gate.add_argument("--max-stream-seconds", type=float, default=120.0,
                      help="ceiling on a granted window, whatever the camera requests")
    args = p.parse_args(argv)

    allocator = Allocator(
        args.total, args.boost, args.decay,
        min_confidence=args.min_confidence, cooldown=args.cooldown,
        max_concurrent=args.max_concurrent, stream_seconds=args.stream_seconds,
        max_stream_seconds=args.max_stream_seconds, trigger_mode=args.trigger_mode,
    )

    if args.http_port:
        httpd = ThreadingHTTPServer(("0.0.0.0", args.http_port), make_handler(allocator))
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"[http] dashboard on :{args.http_port}")
        print(f"  over SSH: ssh -L {args.http_port}:localhost:{args.http_port} <user>@<this-host>", flush=True)

    print(f"[alloc] total budget {args.total} kbps", flush=True)
    try:
        run_control_plane(allocator, args.broker, args.broker_port, args.interval)
    except KeyboardInterrupt:
        print("\nstopping")
    except OSError as e:
        print(f"Could not reach broker {args.broker}:{args.broker_port}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())