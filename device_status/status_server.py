"""Status collector -- runs ON THE SERVER, receives heartbeats from devices.

The receiving half of status_client.py. Accepts the same payload over either
transport the client can send it on, and both can run at once:

    python status_server.py --http-port 8000            # accept POSTs
    python status_server.py --mqtt-host localhost       # subscribe to a broker
    python status_server.py --mqtt-host localhost --http-port 8000   # both

Standalone: imports nothing from the rest of this repo. paho-mqtt is needed
only for --mqtt-host; the HTTP path and the dashboard run on the stdlib alone.

With --http-port it also serves:
    GET /            a dashboard of every device it has heard from
    GET /api/devices the same thing as JSON
    GET /api/history?device=<id>&limit=N   recent heartbeats, if --db is on

Devices are never configured here. A device exists because it sent something,
which means a new Jetson needs no server-side change to show up.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# Health thresholds. Deliberately generous -- this is a bookshelf testbed, not a
# datacenter, and a heartbeat that cries wolf gets ignored.
STALE_AFTER_S = 180.0      # no heartbeat for this long -> presumed offline
TEMP_WARN_C = 80.0         # Orin throttles around 85-90C
TEMP_CRIT_C = 90.0
DISK_WARN_PCT = 90.0
MEM_WARN_PCT = 90.0


# ---- registry -----------------------------------------------------------


class Registry:
    """Latest status per device, plus optional history on disk.

    In-memory is the source of truth for the dashboard; SQLite is append-only
    history, opened with check_same_thread=False because MQTT callbacks and HTTP
    handlers write from different threads. One lock covers both -- writes are a
    few per minute, so there is nothing to gain from finer granularity.
    """

    def __init__(self, db_path: str | None = None, stale_after: float = STALE_AFTER_S):
        self.stale_after = stale_after
        self._lock = threading.Lock()
        self._devices: dict[str, dict] = {}
        self._db: sqlite3.Connection | None = None
        if db_path:
            self._db = sqlite3.connect(db_path, check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS heartbeats ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " device_id TEXT NOT NULL,"
                " received_at REAL NOT NULL,"
                " payload TEXT NOT NULL)"
            )
            self._db.execute("CREATE INDEX IF NOT EXISTS idx_device_time ON heartbeats(device_id, received_at)")
            self._db.commit()

    def record(self, payload: dict, source: str, retained: bool = False) -> str | None:
        """Store one heartbeat. Returns the device id, or None if unusable.

        `retained` marks a message the broker replayed out of its own storage
        when we subscribed, rather than one a device just sent. The distinction
        decides how old the reading is: stamping a replay with `now` would make
        a device that died last week read "last seen 0s ago", which is the one
        thing the dashboard exists to notice. Its own `timestamp` field is the
        real observation time, so use that and let staleness apply normally.
        """
        device_id = payload.get("device_id")
        if not isinstance(device_id, str) or not device_id:
            return None
        now = time.time()
        received_at = now
        if retained:
            stamped = _parse_timestamp(payload.get("timestamp"))
            # Capped at now: a device with a fast clock must not appear to have
            # reported from the future, which would make it permanently fresh.
            received_at = min(stamped, now) if stamped is not None else now
        with self._lock:
            self._devices[device_id] = {
                "device_id": device_id,
                "received_at": received_at,
                "source": source,
                "online": True,
                "status": payload,
            }
            # Replays are re-reads of something already seen, not new
            # observations -- recording them would duplicate a row into history
            # on every server restart.
            if self._db is not None and not retained:
                self._db.execute(
                    "INSERT INTO heartbeats (device_id, received_at, payload) VALUES (?, ?, ?)",
                    (device_id, received_at, json.dumps(payload)),
                )
                self._db.commit()
        return device_id

    def mark_offline(self, device_id: str, reason: str) -> None:
        """Record a Last Will. Keeps the last known status; only liveness changes.

        Deliberately does not create an entry for a device never seen -- a will
        from a device that has never reported has no status to show, and
        inventing a row for it would put a permanent ghost on the dashboard.
        """
        with self._lock:
            entry = self._devices.get(device_id)
            if entry is not None:
                entry["online"] = False
                entry["offline_reason"] = reason

    def snapshot(self) -> list[dict]:
        """Every device, newest heartbeat first, with liveness re-evaluated.

        Staleness is computed on read rather than by a sweeper thread: the
        answer is a subtraction, and a timer that has to fire to make the
        dashboard correct is a second thing that can fail.
        """
        now = time.time()
        with self._lock:
            entries = [dict(e) for e in self._devices.values()]
        for e in entries:
            age = now - e["received_at"]
            e["age_seconds"] = round(age, 1)
            if e["online"] and age > self.stale_after:
                e["online"] = False
                e["offline_reason"] = "stale"
            e["health"] = _health(e)
        entries.sort(key=lambda e: e["received_at"], reverse=True)
        return entries

    def history(self, device_id: str, limit: int = 100) -> list[dict]:
        if self._db is None:
            return []
        with self._lock:
            rows = self._db.execute(
                "SELECT received_at, payload FROM heartbeats WHERE device_id = ?"
                " ORDER BY received_at DESC LIMIT ?",
                (device_id, limit),
            ).fetchall()
        return [{"received_at": r[0], "status": json.loads(r[1])} for r in rows]


def _parse_timestamp(value) -> float | None:
    """ISO-8601 UTC (what the client stamps) -> epoch seconds."""
    if not isinstance(value, str):
        return None
    try:
        # fromisoformat gained 'Z' support in 3.11; this keeps 3.10 working.
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _health(entry: dict) -> dict:
    """Reduce a heartbeat to one status word plus the reasons behind it.

    Returns every reason, not just the worst, so the dashboard can say
    "offline; disk 94%" rather than making someone open the JSON to find the
    second problem.
    """
    if not entry.get("online"):
        return {"level": "critical", "reasons": [entry.get("offline_reason", "offline")]}

    status = entry.get("status") or {}
    reasons: list[str] = []
    level = "good"

    temps = status.get("temperatures_c") or {}
    hottest = max(temps.values(), default=None)
    if hottest is not None:
        if hottest >= TEMP_CRIT_C:
            level = "serious"
            reasons.append(f"{hottest:.0f}°C")
        elif hottest >= TEMP_WARN_C:
            level = "warning"
            reasons.append(f"{hottest:.0f}°C")

    disk = (status.get("disk") or {}).get("used_pct")
    if disk is not None and disk >= DISK_WARN_PCT:
        level = "serious" if level == "serious" else "warning"
        reasons.append(f"disk {disk:.0f}%")

    mem = (status.get("memory") or {}).get("used_pct")
    if mem is not None and mem >= MEM_WARN_PCT:
        level = "serious" if level == "serious" else "warning"
        reasons.append(f"mem {mem:.0f}%")

    battery = status.get("battery") or {}
    pct = battery.get("percent")
    if pct is not None and pct <= 20 and battery.get("status") != "Charging":
        level = "serious"
        reasons.append(f"battery {pct}%")

    return {"level": level, "reasons": reasons}


# ---- formatting ---------------------------------------------------------


def summarize(payload: dict) -> str:
    """A one-line console receipt of a heartbeat.

    Built from whatever is actually present, so a payload from a non-Jetson (no
    GPU, no rails) prints a shorter line rather than a row of "None". The full
    payload is always stored regardless -- see --verbose to print it too.
    """
    bits = []
    uptime = payload.get("uptime_seconds")
    if uptime is not None:
        bits.append(f"up {human_duration(uptime)}")
    cpu_pct = (payload.get("cpu") or {}).get("used_pct")
    if cpu_pct is not None:
        bits.append(f"cpu {cpu_pct:.0f}%")
    gpu_pct = (payload.get("gpu") or {}).get("load_pct")
    if gpu_pct is not None:
        bits.append(f"gpu {gpu_pct:.0f}%")
    hottest = max((payload.get("temperatures_c") or {}).values(), default=None)
    if hottest is not None:
        bits.append(f"{hottest:.0f}°C")
    watts = (payload.get("power") or {}).get("total_watts")
    if watts is not None:
        bits.append(f"{watts:.1f}W")
    mem_pct = (payload.get("memory") or {}).get("used_pct")
    if mem_pct is not None:
        bits.append(f"mem {mem_pct:.0f}%")
    disk_pct = (payload.get("disk") or {}).get("used_pct")
    if disk_pct is not None:
        bits.append(f"disk {disk_pct:.0f}%")
    battery_pct = (payload.get("battery") or {}).get("percent")
    if battery_pct is not None:
        bits.append(f"bat {battery_pct}%")
    return "  ".join(bits) if bits else "(no readings)"


def human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m"
    return f"{seconds}s"


# Status colors are the reserved status palette, and each ships with an icon and
# a word -- never color alone, since two of these sit below 3:1 on the light
# surface and a red/green pair is exactly what a colorblind reader cannot split.
STATUS_STYLE = {
    "good": ("●", "Healthy", "var(--status-good)"),
    "warning": ("▲", "Warning", "var(--status-warning)"),
    "serious": ("▲", "Serious", "var(--status-serious)"),
    "critical": ("■", "Offline", "var(--status-critical)"),
}

PAGE_CSS = """
:root {
  color-scheme: light;
  --surface-1: #fcfcfb;
  --page: #f9f9f7;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --hairline: rgba(11,11,11,0.10);
  --gridline: #e1e0d9;
  --status-good: #0ca30c;
  --status-warning: #fab219;
  --status-serious: #ec835a;
  --status-critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --hairline: rgba(255,255,255,0.10);
    --gridline: #2c2c2a;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1: #1a1a19;
  --page: #0d0d0d;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted: #898781;
  --hairline: rgba(255,255,255,0.10);
  --gridline: #2c2c2a;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 24px;
  background: var(--page); color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 14px; line-height: 1.5;
}
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
.sub { color: var(--text-secondary); margin: 0 0 24px; font-size: 13px; }
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }
.tile {
  background: var(--surface-1); border: 1px solid var(--hairline);
  border-radius: 8px; padding: 14px 18px; min-width: 128px;
}
.tile .label { color: var(--text-secondary); font-size: 12px; }
.tile .value { font-size: 30px; font-weight: 600; letter-spacing: -0.01em; }
.card {
  background: var(--surface-1); border: 1px solid var(--hairline);
  border-radius: 8px; overflow-x: auto;
}
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 10px 14px; white-space: nowrap; }
th {
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.04em; color: var(--text-muted);
  border-bottom: 1px solid var(--gridline);
}
td { border-bottom: 1px solid var(--gridline); }
tr:last-child td { border-bottom: none; }
.device { font-weight: 600; }
/* Truncated because the Orin's model string is 65 characters and would
   otherwise widen the first column enough to push Disk off the right edge. */
.model {
  color: var(--text-muted); font-size: 12px; font-variant-numeric: normal;
  max-width: 260px; overflow: hidden; text-overflow: ellipsis;
}
/* The status hue rides the icon, not the word: two of the four status steps
   sit below 3:1 on the light surface (warning is 1.79), which is fine for a
   glyph read as a shape but not for text that has to be read. The word stays
   in primary ink, so the pairing survives greyscale, low vision, and CVD. */
.badge { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; }
.badge .dot { font-size: 11px; line-height: 1; }
.badge .word { color: var(--text-primary); }
.reasons { color: var(--text-secondary); font-size: 12px; }
.muted { color: var(--text-muted); }
.empty { padding: 40px 20px; text-align: center; color: var(--text-secondary); }
footer { margin-top: 20px; color: var(--text-muted); font-size: 12px; }
"""


def _esc(value) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_dashboard(entries: list[dict], refresh: int) -> str:
    counts = {"good": 0, "warning": 0, "serious": 0, "critical": 0}
    for e in entries:
        counts[e["health"]["level"]] += 1
    online = sum(1 for e in entries if e["online"])

    rows = []
    for e in entries:
        s = e.get("status") or {}
        icon, word, color = STATUS_STYLE[e["health"]["level"]]
        reasons = ", ".join(e["health"]["reasons"])
        temps = s.get("temperatures_c") or {}
        hottest = max(temps.values(), default=None)
        power = (s.get("power") or {}).get("total_watts")
        mem = (s.get("memory") or {}).get("used_pct")
        dsk = (s.get("disk") or {}).get("used_pct")
        battery = s.get("battery") or {}
        pct = battery.get("percent")

        rows.append(
            f"""<tr>
  <td><div class="device">{_esc(e['device_id'])}</div>
      <div class="model" title="{_esc(s.get('model') or '')}">{_esc(s.get('model') or '')}</div></td>
  <td><span class="badge"><span class="dot" style="color:{color}">{icon}</span><span class="word">{word}</span></span>
      {f'<div class="reasons">{_esc(reasons)}</div>' if reasons else ''}</td>
  <td>{_esc(human_duration(s.get('uptime_seconds')))}</td>
  <td>{_esc(human_duration(e['age_seconds']))} ago</td>
  <td>{f'{hottest:.1f}&deg;C' if hottest is not None else '<span class="muted">&mdash;</span>'}</td>
  <td>{f'{power:.2f} W' if power is not None else '<span class="muted">&mdash;</span>'}</td>
  <td>{f'{pct}%' if pct is not None else '<span class="muted">no battery</span>'}</td>
  <td>{f'{mem:.0f}%' if mem is not None else '<span class="muted">&mdash;</span>'}</td>
  <td>{f'{dsk:.0f}%' if dsk is not None else '<span class="muted">&mdash;</span>'}</td>
</tr>"""
        )

    table = (
        """<table><thead><tr>
  <th>Device</th><th>Status</th><th>Uptime</th><th>Last seen</th>
  <th>Peak temp</th><th>Power</th><th>Battery</th><th>Memory</th><th>Disk</th>
</tr></thead><tbody>"""
        + "".join(rows)
        + "</tbody></table>"
        if rows
        else '<div class="empty">No devices have reported yet.</div>'
    )

    attention = counts["warning"] + counts["serious"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{refresh}">
<title>Device Status</title>
<style>{PAGE_CSS}</style>
</head><body><div class="wrap">
<h1>Device status</h1>
<p class="sub">{len(entries)} device(s) reporting &middot; refreshed {now}</p>
<div class="tiles">
  <div class="tile"><div class="label">Online</div><div class="value">{online}</div></div>
  <div class="tile"><div class="label">Needs attention</div><div class="value">{attention}</div></div>
  <div class="tile"><div class="label">Offline</div><div class="value">{counts['critical']}</div></div>
</div>
<div class="card">{table}</div>
<footer>JSON at <code>/api/devices</code> &middot; auto-refreshes every {refresh}s</footer>
</div></body></html>"""


# ---- http ---------------------------------------------------------------


def make_http_handler(registry: Registry, refresh: int, verbose: bool = False):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DeviceStatus/1.0"

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, indent=2).encode(), "application/json")

        def do_POST(self):
            path = urlparse(self.path).path
            if path not in ("/status", "/api/status", "/"):
                self._send_json(404, {"error": "not found"})
                return

            length = int(self.headers.get("Content-Length") or 0)
            # Cap the read: without it, a bad client claiming a huge
            # Content-Length would have the server allocate it.
            if length <= 0 or length > 1_000_000:
                self._send_json(400, {"error": "missing or oversized body"})
                return
            try:
                payload = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self._send_json(400, {"error": f"invalid JSON: {e}"})
                return
            if not isinstance(payload, dict):
                self._send_json(400, {"error": "expected a JSON object"})
                return

            device_id = registry.record(payload, source="http")
            if device_id is None:
                self._send_json(400, {"error": "payload has no device_id"})
                return
            print(f"[http] {device_id}: {summarize(payload)}")
            if verbose:
                print(json.dumps(payload, indent=2))
            self._send_json(202, {"ok": True, "device_id": device_id})

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send(200, render_dashboard(registry.snapshot(), refresh).encode(), "text/html; charset=utf-8")
            elif parsed.path == "/api/devices":
                self._send_json(200, {"devices": registry.snapshot()})
            elif parsed.path == "/api/history":
                params = parse_qs(parsed.query)
                device = (params.get("device") or [None])[0]
                if not device:
                    self._send_json(400, {"error": "?device=<id> required"})
                    return
                try:
                    limit = min(int((params.get("limit") or ["100"])[0]), 1000)
                except ValueError:
                    limit = 100
                self._send_json(200, {"device_id": device, "history": registry.history(device, limit)})
            elif parsed.path == "/healthz":
                self._send_json(200, {"ok": True})
            else:
                self._send_json(404, {"error": "not found"})

        def log_message(self, *args):
            pass  # the handlers above print what matters; this is per-request noise

    return Handler


# ---- mqtt ---------------------------------------------------------------


def start_mqtt(registry: Registry, host: str, port: int, base_topic: str, username=None, password=None, verbose=False):
    """Subscribe to every device's status topic. Returns the connected client."""
    import paho.mqtt.client as mqtt

    # '#' matches the parent level too, so this one filter covers both
    # devices/<id>/status and the devices/<id>/status/lwt wills.
    topic_filter = f"{base_topic}/+/status/#"

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code.is_failure:
            print(f"[mqtt] connection refused: {reason_code}", file=sys.stderr)
            return
        # Subscribe from on_connect, not after connect(), so an automatic
        # reconnect re-subscribes -- a resumed session that never re-subscribed
        # would sit there silently receiving nothing.
        client.subscribe(topic_filter, qos=1)
        print(f"[mqtt] connected to {host}:{port}, subscribed to {topic_filter}")

    def on_message(client, userdata, msg):
        parts = msg.topic.split("/")
        # devices/<id>/status[/lwt]
        device_id = parts[1] if len(parts) > 2 else None
        try:
            payload = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[mqtt] {msg.topic}: bad JSON ({e})", file=sys.stderr)
            return
        if not isinstance(payload, dict):
            print(f"[mqtt] {msg.topic}: expected a JSON object", file=sys.stderr)
            return

        if msg.topic.endswith("/lwt"):
            target = payload.get("device_id") or device_id
            if not target:
                return
            # A will the broker replays from storage (retain flag set) says only
            # "this device died at some point in the past" -- it carries no date,
            # and the device may well be running again now. Honouring it would
            # show a live device as offline after every server restart, until
            # its next heartbeat. Live wills arrive with the flag clear (the
            # broker sets RETAIN=0 for an established subscription), and those
            # are the ones that mean something right now; for the rest, the
            # staleness timer reaches the same verdict from evidence.
            if msg.retain:
                print(f"[mqtt] {target}: ignoring retained will (staleness decides)")
                return
            registry.mark_offline(target, payload.get("status", "offline"))
            print(f"[mqtt] {target}: last will -> offline")
            return

        recorded = registry.record(payload, source="mqtt", retained=bool(msg.retain))
        if recorded is None:
            print(f"[mqtt] {msg.topic}: payload has no device_id", file=sys.stderr)
        else:
            print(f"[mqtt] {recorded}: {summarize(payload)}")
            if verbose:
                print(json.dumps(payload, indent=2))

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="status-server")
    if username:
        client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host, port, keepalive=60)
    client.loop_start()
    return client


# ---- cli ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect device status heartbeats over MQTT and/or HTTP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--http-port", type=int, help="serve the dashboard and accept POSTs on this port")
    p.add_argument("--bind", default="0.0.0.0", help="address the HTTP server listens on")
    p.add_argument("--mqtt-host", help="broker to subscribe to")
    p.add_argument("--mqtt-port", type=int, default=1883, help="broker port")
    p.add_argument("--base-topic", default="devices", help="topic prefix; subscribes to <base>/+/status/#")
    p.add_argument("--username", help="broker username")
    p.add_argument("--password", help="broker password")
    p.add_argument("--db", help="SQLite file for heartbeat history (omit for memory only)")
    p.add_argument("--stale-after", type=float, default=STALE_AFTER_S, help="seconds without a heartbeat before a device reads offline")
    p.add_argument("--refresh", type=int, default=15, help="dashboard auto-refresh interval in seconds")
    p.add_argument("-v", "--verbose", action="store_true", help="print each full payload, not just a summary line")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.http_port and not args.mqtt_host:
        print("Need --http-port, --mqtt-host, or both.", file=sys.stderr)
        return 2

    registry = Registry(db_path=args.db, stale_after=args.stale_after)
    if args.db:
        print(f"History -> {args.db}")

    mqtt_client = None
    if args.mqtt_host:
        try:
            mqtt_client = start_mqtt(
                registry, args.mqtt_host, args.mqtt_port, args.base_topic,
                args.username, args.password, args.verbose,
            )
        except ImportError:
            print("paho-mqtt is not installed. pip install -r requirements.txt", file=sys.stderr)
            return 1
        except OSError as e:
            print(f"Could not connect to {args.mqtt_host}:{args.mqtt_port}: {e}", file=sys.stderr)
            if isinstance(e, ConnectionRefusedError):
                print(
                    "Nothing is listening on that port. This server SUBSCRIBES to an MQTT\n"
                    "broker (mosquitto) -- it is not itself a broker, so --mqtt-host needs\n"
                    "one running. Install one (see mosquitto.conf.example), or drop MQTT\n"
                    "and run --http-port 8000 to accept POSTs directly.",
                    file=sys.stderr,
                )
            return 1

    httpd = None
    if args.http_port:
        httpd = ThreadingHTTPServer(
            (args.bind, args.http_port), make_http_handler(registry, args.refresh, args.verbose)
        )
        print(f"[http] listening on {args.bind}:{args.http_port} -- dashboard at http://localhost:{args.http_port}/")

    try:
        if httpd is not None:
            httpd.serve_forever()
        else:
            # MQTT-only: paho's loop owns a background thread, so park here.
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        if httpd is not None:
            httpd.shutdown()
        if mqtt_client is not None:
            mqtt_client.disconnect()
            mqtt_client.loop_stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
