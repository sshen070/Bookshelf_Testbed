# Device Status Heartbeat

Reports how the Jetson itself is doing — uptime, load, memory, disk, temps,
power draw — to a server over MQTT or HTTP.

- `status_client.py` runs **on the device** and sends.
- `status_server.py` runs **on the server** and collects, with a dashboard.

Standalone: imports nothing from `vit_classifier/`, `WAFT/`, or the deprecated
`src/` pipeline, and none of them import this. It has its own dependency (one),
and either half can be copied to another box on its own.

## Setup

```bash
pip install -r requirements.txt      # only needed for MQTT
```

## You need a broker (for the MQTT path)

**Neither script is a broker.** Both are MQTT *clients* of one: the device
publishes to it, the server subscribes to it. If nothing else is running, both
sides fail identically with `Connection refused` on port 1883 — that error means
"no broker", not "the other script isn't up".

```
  jetson1                    edge1                       edge1
  status_client.py  ──publish──>  mosquitto  <──subscribe──  status_server.py
                                  (the broker)
```

Install it on whichever host both can reach — normally the collector:

```bash
sudo apt install -y mosquitto mosquitto-clients
sudo cp mosquitto.conf.example /etc/mosquitto/conf.d/device_status.conf
sudo systemctl restart mosquitto
```

The example config exists because **mosquitto 2.0 changed two defaults**, and
each produces a failure that looks like the broker isn't there: with no
`listener` line it binds to loopback only (so a remote device is refused), and
anonymous clients are denied (`Connection Refused: not authorised`).

Verify before involving these scripts, so you're debugging one thing at a time:

```bash
mosquitto_sub -h <broker> -t 'devices/#' -v     # on the server
mosquitto_pub -h <broker> -t devices/test -m hi # from the device
```

### Or skip the broker entirely

The HTTP path needs no broker and no dependency — the server accepts POSTs
directly:

```bash
python status_server.py --http-port 8000                    # on the server
python status_client.py --http http://<server>:8000/status  # on the device
```

Worth it when there is one collector and one consumer. You give up retained
state, the Last Will, and fan-out to multiple subscribers.

## Client usage

```bash
python status_client.py --dry-run                        # print payload, send nothing
python status_client.py --host 192.168.1.50              # one heartbeat over MQTT
python status_client.py --host 192.168.1.50 --interval 60   # every 60s until stopped
python status_client.py --http http://server:8000/status    # POST JSON instead
```

Publishes to `devices/<hostname>/status` by default (`--topic` to override).
Credentials come from `--username` / `--password`, or `STATUS_MQTT_USERNAME` /
`STATUS_MQTT_PASSWORD` — prefer the env vars, since a password in argv is
visible to every process on the box via `ps`.

Exit status is 0 only when the server acknowledged the send, so this works as a
connectivity check in a script.

## Payload

About 1.7 KB. Everything is read from `/proc` and `/sys` — no `tegrastats`, no
`jetson-stats`, no root.

| Group | Fields |
|---|---|
| identity | `device_id`, `model`, `os` (hostname, kernel, distro, **L4T**, python, arch) |
| liveness | `timestamp`, `uptime_seconds`, `boot_time` |
| load | `cpu` (overall + per core, clocks), `gpu` (load, clock), `load_average`, `memory` (incl. swap), `disk` |
| environment | `temperatures_c`, `power` (rails, `nvpmodel`, `throttle_events`), `battery`, `fan_rpm` |
| connectivity | `network` (per-interface address, link, throughput, wifi RSSI), `cameras` |

```json
{
  "device_id": "jetson1",
  "timestamp": "2026-07-29T00:42:42+00:00",
  "model": "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super",
  "os": {"hostname": "jetson1", "kernel": "5.15.148-tegra",
         "distro": "Ubuntu 22.04.5 LTS", "l4t": "R36.4.7",
         "python": "3.10.12", "arch": "aarch64"},
  "uptime_seconds": 351451.4,
  "boot_time": "2026-07-24T23:05:11+00:00",
  "load_average": [0.21, 0.2, 0.16],
  "cpu": {"used_pct": 19.0, "per_core_pct": [21.0, 16.0, 15.3, 20.8, 21.8, 19.2],
          "cores": 6, "freq_mhz": 730, "max_freq_mhz": 1728},
  "gpu": {"load_pct": 99.9, "freq_mhz": 1020, "max_freq_mhz": 1020},
  "memory": {"total_mb": 7620, "available_mb": 4419, "used_pct": 42.0,
             "swap_total_mb": 3810, "swap_used_pct": 7.8},
  "disk": {"total_gb": 115.7, "free_gb": 69.7, "used_pct": 39.7},
  "temperatures_c": {"cpu-thermal": 41.9, "gpu-thermal": 42.2, "tj-thermal": 42.5},
  "power": {"total_watts": 3.53,
            "rails": {"VDD_IN": {"volts": 5.008, "amps": 0.704, "watts": 3.53}},
            "nvpmodel": 2, "throttle_events": {"oc1": 0, "oc2": 0, "oc3": 78}},
  "battery": null,
  "fan_rpm": 783,
  "network": {"primary_ipv4": "192.168.1.215",
              "interfaces": [
                {"name": "enP8p1s0", "state": "up", "ipv4": "10.11.1.111",
                 "link_speed_mbps": 1000, "rx_kbps": 3.1, "tx_kbps": 2.6},
                {"name": "wlP1p1s0", "state": "up", "ipv4": "192.168.1.215",
                 "rx_kbps": 3.9, "tx_kbps": 22.1,
                 "wifi": {"link_quality": 60.0, "signal_dbm": -50.0}}]},
  "cameras": ["video0"]
}
```

Key names are **append-only**: new collectors never rename or drop an existing
field, because the server keys its health rules off `memory.used_pct`,
`disk.used_pct`, `temperatures_c` and `battery.percent`, and an older consumer
has to keep working after the device is upgraded.

### Rates and percentages are averages, not instants

`cpu.used_pct`, `cpu.per_core_pct` and the per-interface `rx_kbps`/`tx_kbps` are
differenced against the *previous heartbeat*, so at `--interval 60` they
describe the whole minute. That is both cheaper than sleeping for a sampling
window on every beat and more representative than a 100 ms glimpse. Two
consequences worth knowing:

- **The first heartbeat after start has no `rx_kbps`/`tx_kbps`** — there is no
  baseline to difference against yet. CPU is the exception: it takes a one-off
  150 ms sample so the first beat isn't empty.
- A counter that goes backwards (interface bounced, so the counter reset) is
  skipped for that round rather than reported as a negative rate.

### Jetson-specific fields worth watching

- **`gpu.load_pct`** — the kernel's `load` node is **per-mille**, not percent. It
  reads 999 under a full CUDA burn (measured on this board), so it is divided by
  10 here. Read raw, a busy GPU would report `999%`.
- **`power.nvpmodel`** — the active power profile (`2` = MAXN_SUPER here). A board
  that silently booted into a low-power mode is a classic cause of "inference got
  slower and nothing changed."
- **`power.throttle_events`** — cumulative overcurrent throttle counts since boot.
  The absolute number matters less than whether it *climbs* between heartbeats;
  rising counts mean the supply can't hold under load, which shows up as
  mysterious slowdowns long before anything fails. This board booted with
  `oc3` already at 78.
- **`os.l4t`** — the JetPack BSP version, which is what actually explains a
  CUDA/driver mismatch. More useful on a Jetson than the distro alone.
- **`memory.swap_used_pct`** — the Orin shares 8 GB between CPU and GPU, so swap
  climbing is the first sign a model no longer fits.

### `battery` is null here, and that's correct

This board has **no battery**. `/sys/class/power_supply/` is empty on the Orin
Nano dev kit — it's mains-powered, so there is no charge level to report. The
field stays in the payload and reads `null` rather than being dropped or filled
with a plausible-looking `100`, because a server that alerts on battery health
should be able to tell "no battery present" from "battery full".

What replaces it on this board is `power`, read from the on-board INA3221 rail
monitor: actual draw in watts, total (`VDD_IN`) and split across the CPU/GPU and
SoC rails. That is the useful power signal on a mains device — it shows load and
catches a supply that is browning out. If you later run this on something
battery-backed, `battery` populates on its own.

Sensors that don't exist come back `null`, and thermal zones that exist but fail
their read (`cv0`–`cv2` on an idle Orin) are omitted rather than reported as
`0.0` — a fake 0 °C reads as a healthy sensor to an alerting rule.

## Server usage

```bash
python status_server.py --http-port 8000                    # accept POSTs
python status_server.py --mqtt-host localhost               # subscribe
python status_server.py --mqtt-host localhost --http-port 8000 --db status.db
```

Both transports can run at once, into one registry — useful while migrating, or
with a mix of devices. Endpoints:

| Route | What it gives you |
|---|---|
| `GET /` | dashboard: one row per device, auto-refreshing |
| `GET /api/devices` | the same data as JSON |
| `GET /api/history?device=<id>&limit=N` | recent heartbeats (needs `--db`) |
| `GET /healthz` | liveness probe |
| `POST /status` | what `--http` on the client targets |

Devices are never configured server-side — a device exists because it sent
something, so a new Jetson needs no change here to appear.

### Seeing what actually arrived

The console prints a one-line receipt per heartbeat, not the payload:

```
[mqtt] jetson1: up 4d 1h  cpu 2%  gpu 0%  42°C  3.5W  mem 43%  disk 40%
```

The full payload is always received and stored regardless — the line is a
summary, not the extent of what was captured. To see all of it:

| Want | Do |
|---|---|
| the whole payload on the console | `-v` / `--verbose` |
| a dashboard | add `--http-port 8000`, open `http://<server>:8000/` |
| the raw JSON | add `--http-port 8000`, then `curl localhost:8000/api/devices` |
| past heartbeats | add `--db status.db`, then `/api/history?device=<id>` |

**`--mqtt-host` alone serves no web page.** The dashboard and the JSON API are
part of the HTTP server, so without `--http-port` there is nothing listening to
open. Running both together is the normal setup:

```bash
python status_server.py --mqtt-host localhost --http-port 8000 --db status.db
```

`--db` adds append-only SQLite history; without it the server keeps only the
latest heartbeat per device in memory.

### How a device is judged offline

Two independent signals, because each covers the other's blind spot:

- **Last Will**, which is immediate but only fires if the broker noticed the
  drop. Detects a crash within seconds.
- **Staleness** (`--stale-after`, default 180s), which needs no cooperation from
  anything. Catches the cases a will misses — HTTP transport, a broker restart,
  a device wedged but still connected.

Health beyond liveness is derived from the payload: temperature over 80 °C
(`warning`) or 90 °C (`serious`), disk or memory over 90%, battery at or below
20% while discharging. All reasons are reported, not just the worst, so a row
reads `81°C, disk 94%` rather than hiding the second problem.

### Retained messages, and why the server checks the retain flag

Both the heartbeat and the will are published **retained**, so a dashboard that
connects late is told the current state instead of waiting for the next
interval. That creates two traps the server has to handle, and both were real
bugs caught in testing:

- A retained heartbeat replayed at subscribe time is **old**, but arrives now.
  Stamping it with the arrival time makes a device that died last week read
  "last seen 0s ago" — so the server uses the payload's own `timestamp` instead,
  capped at now, and lets staleness apply normally.
- A retained will says only "this device died at *some* point" — it carries no
  date, and the device may be running again. Honouring it would show a live
  device as offline after every server restart.

MQTT makes these distinguishable: a broker sets `RETAIN=0` when delivering to an
already-established subscription and `RETAIN=1` only when replaying stored state
to a new subscriber. So live wills are honoured immediately, replayed ones are
ignored, and staleness reaches the same verdict from evidence. Replays are also
kept out of `--db`, so a restart doesn't duplicate history.

## Notes on the MQTT side

- **QoS 1**, so the broker acknowledges every publish. QoS 0 gives no ack at
  all, which would make a publish into a half-dead socket look exactly like a
  delivered one and leave the exit status meaningless.
- **`retain=True`**, so a dashboard subscribing an hour from now is handed the
  last known state immediately instead of waiting out a whole interval.
- **Last Will** on `devices/<id>/status/lwt`: the broker announces the device as
  offline if it drops without a clean disconnect. A Jetson that loses power
  can't report its own death, and without a will "offline" looks identical to
  "idle".

## Running it as a service

```ini
# /etc/systemd/system/device-status.service
[Unit]
Description=Device status heartbeat
After=network-online.target

[Service]
ExecStart=/usr/bin/python3 /home/jetson1/Bookshelf_Testbed/device_status/status_client.py \
    --host 192.168.1.50 --interval 60
Environment=STATUS_MQTT_PASSWORD=...
Restart=always
RestartSec=10
User=jetson1

[Install]
WantedBy=multi-user.target
```

`Restart=always` covers the case the script deliberately doesn't: it exits
nonzero on a broker that is down at startup rather than blocking forever, and
systemd is the right thing to own that retry.
