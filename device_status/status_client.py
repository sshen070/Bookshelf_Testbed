"""Device status heartbeat -- runs ON THE JETSON, reports how it's doing.

Standalone: imports nothing from the rest of this repo, and nothing here is
imported by it. Only paho-mqtt is a third-party dependency, and only when
actually publishing over MQTT (--http and --dry-run run on the stdlib alone).

Reports identity (model, L4T, kernel), liveness (uptime, boot time), load
(CPU per core, GPU, memory, swap, disk), environment (temperatures, power
rails, throttle counters, fan) and connectivity (per-interface link, address,
throughput, wifi signal).

Everything is read straight from /proc and /sys, so there is no dependency on
tegrastats, jetson-stats, or root. Every reading is optional by construction:
a sensor that isn't present on the board comes back as null rather than
raising, which is what lets the same script run on a Pi or a laptop.

CPU percentages and network rates are averages over the gap since the previous
heartbeat, so at --interval 60 they describe the whole minute rather than an
instant. Only the very first heartbeat, which has no baseline to difference
against, samples the CPU over a 150ms window instead.

    python status_client.py --dry-run                     # print, send nothing
    python status_client.py --host 192.168.1.50           # one heartbeat, MQTT
    python status_client.py --host 192.168.1.50 --interval 60   # forever
    python status_client.py --http http://server:8000/status    # POST instead
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HWMON = Path("/sys/class/hwmon")
THERMAL = Path("/sys/class/thermal")
POWER_SUPPLY = Path("/sys/class/power_supply")
NET_CLASS = Path("/sys/class/net")
NVPMODEL_STATUS = Path("/var/lib/nvpmodel/status")

# Interfaces present on every Jetson that never carry traffic worth reporting:
# loopback, Docker's bridge, and the L4T USB-gadget plumbing.
BORING_INTERFACES = ("lo", "docker0", "l4tbr0")

# Rate and utilisation figures need two samples. Rather than sleeping for a
# window on every heartbeat, the previous reading is kept here and the delta
# taken across the whole gap between heartbeats -- cheaper, and a far more
# representative average than a 100ms glimpse. Only the first call, which has
# nothing to diff against, samples inline.
_prev_cpu: dict[str, tuple[int, int]] | None = None
_prev_net: tuple[float, dict[str, tuple[int, int]]] | None = None


# ---- sysfs helpers ------------------------------------------------------
# Reads return None instead of raising. Sysfs is full of files that exist but
# error on read (a disabled sensor gives EINVAL/ENODATA), so a missing value is
# the normal case to handle, not an exceptional one.


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (OSError, ValueError, TypeError):
        # TypeError is not defensive padding: the Orin's cv0/cv1/cv2-thermal
        # zones exist but fail their read, and the failure surfaces from inside
        # the codecs layer as "can't concat NoneType to bytes" rather than as
        # the OSError the syscall actually returned.
        return None


def _read_int(path: Path) -> int | None:
    raw = _read(path)
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


# ---- collectors ---------------------------------------------------------


def uptime_seconds() -> float | None:
    raw = _read(Path("/proc/uptime"))
    return round(float(raw.split()[0]), 1) if raw else None


def boot_time() -> str | None:
    """When the host booted, as an absolute timestamp.

    Uptime alone can't tell a consumer whether two heartbeats came from the same
    boot; a changed boot_time is an unambiguous "this device rebooted".
    """
    raw = _read(Path("/proc/stat"))
    if not raw:
        return None
    for line in raw.splitlines():
        if line.startswith("btime "):
            try:
                return datetime.fromtimestamp(int(line.split()[1]), timezone.utc).isoformat(timespec="seconds")
            except (IndexError, ValueError):
                return None
    return None


def load_average() -> list[float] | None:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return None


def _read_cpu_times() -> dict[str, tuple[int, int]]:
    """{'cpu': (idle, total), 'cpu0': (idle, total), ...} from /proc/stat."""
    times: dict[str, tuple[int, int]] = {}
    raw = _read(Path("/proc/stat"))
    if not raw:
        return times
    for line in raw.splitlines():
        if not line.startswith("cpu"):
            break  # the cpu lines are first; everything after is other counters
        parts = line.split()
        try:
            values = [int(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(values) < 5:
            continue
        # idle + iowait: a core blocked on I/O is not doing work, and counting
        # it as busy makes a disk-bound box look CPU-bound.
        times[parts[0]] = (values[3] + values[4], sum(values))
    return times


def cpu() -> dict | None:
    """Utilisation, core count and clocks.

    Percentages are the average since the previous heartbeat, so at a 60s
    interval this is a 60s mean rather than an instantaneous spike.
    """
    global _prev_cpu
    current = _read_cpu_times()
    if not current:
        return None
    previous = _prev_cpu
    if previous is None:
        # First call only: no baseline exists, so take one over a short window.
        time.sleep(0.15)
        previous, current = current, _read_cpu_times()
        if not current:
            return None
    _prev_cpu = current

    def used_pct(key: str) -> float | None:
        if key not in previous or key not in current:
            return None
        idle_delta = current[key][0] - previous[key][0]
        total_delta = current[key][1] - previous[key][1]
        if total_delta <= 0:
            return None
        return round(100 * (1 - idle_delta / total_delta), 1)

    cores = sorted(k for k in current if k != "cpu")
    freqs = []
    for core in cores:
        khz = _read_int(Path(f"/sys/devices/system/cpu/{core}/cpufreq/scaling_cur_freq"))
        if khz is not None:
            freqs.append(khz / 1000)
    max_khz = _read_int(Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"))

    return {
        "used_pct": used_pct("cpu"),
        "per_core_pct": [used_pct(c) for c in cores],
        "cores": len(cores),
        "freq_mhz": round(sum(freqs) / len(freqs)) if freqs else None,
        "max_freq_mhz": round(max_khz / 1000) if max_khz else None,
    }


def _gpu_dir() -> Path | None:
    """Locate the Tegra GPU node, whose address differs per SoC.

    Globbed rather than hardcoded: it is 17000000.gpu on Orin but a different
    address on Xavier and older boards, and the older layout used /sys/devices/gpu.0.
    """
    for pattern in ("/sys/devices/platform/bus@0/*.gpu", "/sys/devices/platform/*.gpu", "/sys/devices/gpu.0"):
        matches = sorted(Path("/").glob(pattern.lstrip("/")))
        if matches:
            return matches[0]
    return None


def gpu() -> dict | None:
    """Tegra GPU utilisation and clock.

    The `load` node is per-mille, NOT percent -- it reads 999 under a full CUDA
    burn and 0 at idle (measured on this Orin). Reporting it raw would show a
    busy GPU as "999%", so it is divided by 10 here.
    """
    gpu_dir = _gpu_dir()
    if gpu_dir is None:
        return None
    load = _read_int(gpu_dir / "load")
    info: dict = {"load_pct": round(load / 10, 1) if load is not None else None}

    # devfreq reports in Hz; the subdirectory is named after the node address.
    for devfreq in sorted((gpu_dir / "devfreq").glob("*")) if (gpu_dir / "devfreq").is_dir() else []:
        current_hz = _read_int(devfreq / "cur_freq")
        max_hz = _read_int(devfreq / "max_freq")
        if current_hz:
            info["freq_mhz"] = round(current_hz / 1_000_000)
        if max_hz:
            info["max_freq_mhz"] = round(max_hz / 1_000_000)
        break
    return info


def memory() -> dict[str, int | float] | None:
    raw = _read(Path("/proc/meminfo"))
    if not raw:
        return None
    fields = {}
    for line in raw.splitlines():
        key, _, rest = line.partition(":")
        try:
            fields[key] = int(rest.split()[0])  # kB
        except (IndexError, ValueError):
            continue
    total, available = fields.get("MemTotal"), fields.get("MemAvailable")
    if not total:
        return None
    used = total - available if available is not None else None
    info: dict[str, int | float | None] = {
        "total_mb": round(total / 1024),
        "available_mb": round(available / 1024) if available is not None else None,
        # Percent of RAM genuinely unavailable to a new process. Deliberately
        # from MemAvailable, not MemFree -- MemFree counts reclaimable page
        # cache as used and reads ~90% on a healthy box that has been up a while.
        "used_pct": round(100 * used / total, 1) if used is not None else None,
    }
    # Swap matters more than usual here: the Orin shares 8GB between CPU and
    # GPU, and swap climbing is the first sign a model no longer fits.
    swap_total, swap_free = fields.get("SwapTotal"), fields.get("SwapFree")
    if swap_total:
        info["swap_total_mb"] = round(swap_total / 1024)
        if swap_free is not None:
            info["swap_used_pct"] = round(100 * (swap_total - swap_free) / swap_total, 1)
    return info


def disk(path: str = "/") -> dict[str, int | float] | None:
    try:
        st = os.statvfs(path)
    except OSError:
        return None
    total = st.f_blocks * st.f_frsize
    free = st.f_bavail * st.f_frsize
    if not total:
        return None
    return {
        "total_gb": round(total / 1024**3, 1),
        "free_gb": round(free / 1024**3, 1),
        "used_pct": round(100 * (total - free) / total, 1),
    }


def _iface_ipv4(name: str) -> str | None:
    """The IPv4 address bound to one interface, via SIOCGIFADDR.

    An ioctl rather than parsing `ip addr`, so this stays dependency-free and
    doesn't fork a process per interface per heartbeat.
    """
    try:
        import fcntl
        import struct
    except ImportError:
        return None  # not Linux; every other field still works
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = struct.pack("256s", name[:15].encode())
        return socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x8915, packed)[20:24])
    except OSError:
        return None  # interface is up but has no v4 address
    finally:
        sock.close()


def _wifi_signal(name: str) -> dict | None:
    """Link quality and RSSI for a wireless interface, from /proc/net/wireless."""
    raw = _read(Path("/proc/net/wireless"))
    if not raw:
        return None
    for line in raw.splitlines():
        if not line.strip().startswith(f"{name}:"):
            continue
        parts = line.split()
        try:
            # Values carry a trailing '.' meaning "updated"; strip before parsing.
            return {
                "link_quality": float(parts[2].rstrip(".")),
                "signal_dbm": float(parts[3].rstrip(".")),
            }
        except (IndexError, ValueError):
            return None
    return None


def network() -> dict | None:
    """Per-interface link state and throughput, plus the primary address.

    Throughput is a rate across the heartbeat gap, not the raw kernel counter --
    a counter that only ever goes up tells you nothing without differencing it,
    and doing that server-side would break the moment a heartbeat is missed.
    """
    global _prev_net
    now = time.time()
    counters: dict[str, tuple[int, int]] = {}
    interfaces = []

    for iface in sorted(NET_CLASS.glob("*")) if NET_CLASS.is_dir() else []:
        name = iface.name
        if name in BORING_INTERFACES:
            continue
        state = _read(iface / "operstate")
        if state != "up":
            continue  # down interfaces are noise on a board with 6 of them
        rx = _read_int(iface / "statistics/rx_bytes")
        tx = _read_int(iface / "statistics/tx_bytes")
        if rx is not None and tx is not None:
            counters[name] = (rx, tx)
        speed = _read_int(iface / "speed")
        entry = {
            "name": name,
            "state": state,
            "ipv4": _iface_ipv4(name),
            "mac": _read(iface / "address"),
            # Wireless interfaces report speed as -1 or error; omit rather than lie.
            "link_speed_mbps": speed if speed and speed > 0 else None,
            "rx_bytes": rx,
            "tx_bytes": tx,
        }
        wifi = _wifi_signal(name)
        if wifi:
            entry["wifi"] = wifi
        interfaces.append(entry)

    if _prev_net is not None:
        prev_time, prev_counters = _prev_net
        elapsed = now - prev_time
        if elapsed > 0:
            for entry in interfaces:
                previous = prev_counters.get(entry["name"])
                if not previous or entry["rx_bytes"] is None:
                    continue
                # Counters wrap or reset when an interface bounces; a negative
                # delta means the baseline is meaningless, so skip this round.
                rx_delta = entry["rx_bytes"] - previous[0]
                tx_delta = entry["tx_bytes"] - previous[1]
                if rx_delta >= 0 and tx_delta >= 0:
                    entry["rx_kbps"] = round(rx_delta * 8 / elapsed / 1000, 1)
                    entry["tx_kbps"] = round(tx_delta * 8 / elapsed / 1000, 1)
    _prev_net = (now, counters)

    if not interfaces:
        return None
    return {"primary_ipv4": _primary_ipv4(), "interfaces": interfaces}


def _primary_ipv4() -> str | None:
    """The address the host would actually send from.

    A UDP socket is 'connected' to pick the route; no packet is sent, and the
    address need not be reachable. This answers "which NIC is really in use"
    on a box with both ethernet and wifi up, which no per-interface read can.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 9))  # TEST-NET-1: reserved, never routed
        return sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()


def os_info() -> dict:
    """Software identity -- what is actually running on the box."""
    distro = None
    raw = _read(Path("/etc/os-release"))
    if raw:
        for line in raw.splitlines():
            if line.startswith("PRETTY_NAME="):
                distro = line.partition("=")[2].strip().strip('"')
                break

    # L4T is the JetPack BSP version, and it is what actually explains a
    # CUDA/driver mismatch -- far more useful on a Jetson than the distro alone.
    l4t = None
    release = _read(Path("/etc/nv_tegra_release"))
    if release:
        head = release.splitlines()[0]
        try:
            major = head.split("# R")[1].split(" ")[0]
            revision = head.split("REVISION:")[1].split(",")[0].strip()
            l4t = f"R{major}.{revision}"
        except IndexError:
            l4t = head.strip("# ").strip() or None

    return {
        "hostname": socket.gethostname(),
        "kernel": platform.release(),
        "distro": distro,
        "l4t": l4t,
        "python": platform.python_version(),
        "arch": platform.machine(),
    }


def cameras() -> list[str]:
    """Video capture nodes present -- an empty list means the camera vanished."""
    return sorted(p.name for p in Path("/dev").glob("video*"))


def temperatures() -> dict[str, float]:
    """Every thermal zone the kernel exposes, in °C, keyed by zone type.

    Zones that exist but have no reading (the cv*-thermal ones on an idle Orin)
    are skipped rather than reported as 0 -- a fake 0°C looks like a healthy
    sensor to an alerting rule, which is worse than an absent key.
    """
    temps = {}
    for zone in sorted(THERMAL.glob("thermal_zone*")):
        name = _read(zone / "type")
        milli = _read_int(zone / "temp")
        if name and milli is not None:
            temps[name] = round(milli / 1000, 1)
    return temps


def _find_hwmon(name: str) -> Path | None:
    for h in sorted(HWMON.glob("hwmon*")):
        if _read(h / "name") == name:
            return h
    return None


def power() -> dict | None:
    """Live power draw from the INA3221 rail monitor (Jetson dev kits).

    This is the closest thing this board has to a battery reading: the kit is
    mains-powered, so what matters is how much it is pulling, not how much is
    left. VDD_IN is total board draw; the other rails break it down.
    """
    hwmon = _find_hwmon("ina3221")
    if hwmon is None:
        return None
    rails = {}
    total_mw = 0
    for i in (1, 2, 3):
        label = _read(hwmon / f"in{i}_label")
        mv = _read_int(hwmon / f"in{i}_input")
        ma = _read_int(hwmon / f"curr{i}_input")
        if not label or mv is None or ma is None:
            continue
        mw = round(mv * ma / 1000)
        rails[label] = {"volts": round(mv / 1000, 3), "amps": round(ma / 1000, 3), "watts": round(mw / 1000, 2)}
        if label == "VDD_IN":
            total_mw = mw
    if not rails:
        return None
    info: dict = {"total_watts": round(total_mw / 1000, 2) if total_mw else None, "rails": rails}

    # Which nvpmodel profile is active. Read from the status file rather than
    # shelling out to `nvpmodel -q`, which would fork a process per heartbeat.
    # A board silently booted into a low-power mode is a classic cause of
    # "inference got slower and nothing changed".
    pmode = _read(NVPMODEL_STATUS)
    if pmode and "pmode:" in pmode:
        try:
            info["nvpmodel"] = int(pmode.split("pmode:")[1].strip()[:4])
        except ValueError:
            pass

    # Overcurrent throttle counters. These are cumulative since boot, so what
    # matters is that they climb between heartbeats -- a rising count means the
    # supply cannot hold up under load, which shows up as mysterious slowdowns
    # long before anything actually fails.
    soctherm = _find_hwmon("soctherm_oc")
    if soctherm is not None:
        events = {}
        for counter in sorted(soctherm.glob("oc*_event_cnt")):
            value = _read_int(counter)
            if value is not None:
                events[counter.name.replace("_event_cnt", "")] = value
        if events:
            info["throttle_events"] = events
    return info


def battery() -> dict | None:
    """Battery state, for hosts that have one.

    Returns None on this Jetson -- /sys/class/power_supply is empty on a
    mains-powered dev kit. Kept because the same heartbeat is the thing you
    want on a battery-backed node, and reporting null is honest where
    reporting 100% would be a lie that hides a dead UPS.
    """
    if not POWER_SUPPLY.is_dir():
        return None
    for supply in sorted(POWER_SUPPLY.iterdir()):
        if _read(supply / "type") != "Battery":
            continue
        capacity = _read_int(supply / "capacity")
        info = {
            "name": supply.name,
            "percent": capacity,
            "status": _read(supply / "status"),
        }
        # Vendors expose one or the other, in µAh or µWh; report whichever exists.
        for key, src in (("energy_full_wh", "energy_full"), ("charge_full_ah", "charge_full")):
            val = _read_int(supply / src)
            if val is not None:
                info[key] = round(val / 1_000_000, 2)
        return info
    return None


def fan_rpm() -> int | None:
    hwmon = _find_hwmon("pwm_tach")
    return _read_int(hwmon / "rpm") if hwmon else None


def board_model() -> str | None:
    # device-tree strings are NUL-terminated; strip it or it ends up in JSON.
    raw = _read(Path("/proc/device-tree/model"))
    return raw.replace("\x00", "").strip() if raw else None


def collect_status(device_id: str) -> dict:
    """One heartbeat: everything this host can say about itself right now.

    Key names are append-only. A collector added here must not rename or drop an
    existing field -- the server keys its health rules off `memory.used_pct`,
    `disk.used_pct`, `temperatures_c` and `battery.percent`, and a consumer
    written against an older payload has to keep working after an upgrade.
    """
    return {
        "device_id": device_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": board_model(),
        "os": os_info(),
        "uptime_seconds": uptime_seconds(),
        "boot_time": boot_time(),
        "load_average": load_average(),
        "cpu": cpu(),
        "gpu": gpu(),
        "memory": memory(),
        "disk": disk(),
        "temperatures_c": temperatures(),
        "power": power(),
        "battery": battery(),
        "fan_rpm": fan_rpm(),
        "network": network(),
        "cameras": cameras(),
    }


# ---- transports ---------------------------------------------------------


def send_http(url: str, payload: dict, timeout: float) -> bool:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            print(f"POST {url} -> {resp.status}")
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError) as e:
        print(f"POST {url} failed: {e}", file=sys.stderr)
        return False


class MqttSender:
    """Publishes status to a broker, connecting once and reusing the session.

    The publish happens only after the broker's CONNACK has arrived, not merely
    after connect() returns -- connect() completes the TCP handshake, and
    publishing on the next line races the MQTT one.
    """

    def __init__(self, host: str, port: int, topic: str, device_id: str, username=None, password=None, timeout=10.0):
        import paho.mqtt.client as mqtt  # imported here so --http/--dry-run need no dependency

        self.topic = topic
        self.timeout = timeout
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"status-{device_id}")
        if username:
            self.client.username_pw_set(username, password)

        # Last Will: the broker publishes this if we vanish without a clean
        # disconnect. A device that loses power cannot announce its own death,
        # and without a will "offline" is indistinguishable from "idle".
        self.client.will_set(
            f"{topic}/lwt",
            json.dumps({"device_id": device_id, "status": "offline"}),
            qos=1,
            retain=True,
        )
        self.client.connect(host, port, keepalive=60)
        self.client.loop_start()
        print(f"Connected to {host}:{port}, publishing to {topic}")

    def send(self, payload: dict) -> bool:
        # QoS 1 so the broker acknowledges; QoS 0 gives no ack at all, and a
        # publish into a half-dead socket would look identical to a delivered
        # one. retain=True so a dashboard that subscribes later immediately
        # gets the last known state instead of waiting for the next heartbeat.
        info = self.client.publish(self.topic, json.dumps(payload), qos=1, retain=True)
        info.wait_for_publish(self.timeout)
        if not info.is_published():
            print(f"Publish not acknowledged within {self.timeout}s", file=sys.stderr)
            return False
        print(f"Published status to {self.topic}")
        return True

    def close(self) -> None:
        self.client.disconnect()
        self.client.loop_stop()


# ---- cli ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Report Jetson device status to a server over MQTT or HTTP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default=os.environ.get("STATUS_MQTT_HOST"), help="MQTT broker hostname or IP")
    p.add_argument("--port", type=int, default=int(os.environ.get("STATUS_MQTT_PORT", 1883)), help="MQTT broker port")
    p.add_argument("--topic", default=None, help="MQTT topic (default: devices/<device-id>/status)")
    p.add_argument("--username", default=os.environ.get("STATUS_MQTT_USERNAME"), help="broker username")
    p.add_argument("--password", default=os.environ.get("STATUS_MQTT_PASSWORD"), help="broker password (prefer the env var)")
    p.add_argument("--http", metavar="URL", help="POST JSON to this URL instead of using MQTT")
    p.add_argument("--dry-run", action="store_true", help="print the payload and exit; contacts nothing")
    p.add_argument("--device-id", default=os.environ.get("STATUS_DEVICE_ID") or socket.gethostname(), help="identifies this device")
    p.add_argument("--interval", type=float, default=0, help="seconds between heartbeats; 0 sends one and exits")
    p.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for a send to be acknowledged")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    topic = args.topic or f"devices/{args.device_id}/status"

    if args.dry_run:
        print(json.dumps(collect_status(args.device_id), indent=2))
        return 0

    if not args.http and not args.host:
        print("Need one of --host (MQTT), --http URL, or --dry-run.", file=sys.stderr)
        return 2

    sender = None
    if not args.http:
        try:
            sender = MqttSender(args.host, args.port, topic, args.device_id, args.username, args.password, args.timeout)
        except ImportError:
            print("paho-mqtt is not installed. pip install -r requirements.txt (or use --http).", file=sys.stderr)
            return 1
        except OSError as e:
            print(f"Could not connect to {args.host}:{args.port}: {e}", file=sys.stderr)
            if isinstance(e, ConnectionRefusedError):
                print(
                    "Nothing is listening on that port. This script publishes to an MQTT\n"
                    "broker (mosquitto) -- status_server.py is NOT one, it is another client\n"
                    "of the same broker. Either start a broker (see mosquitto.conf.example)\n"
                    "or skip it entirely with --http http://<server>:8000/status",
                    file=sys.stderr,
                )
            return 1

    ok = True
    try:
        while True:
            status = collect_status(args.device_id)
            ok = send_http(args.http, status, args.timeout) if args.http else sender.send(status)
            if args.interval <= 0:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if sender:
            sender.close()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
