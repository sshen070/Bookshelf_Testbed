"""Raise a detection event on a camera -- the hook WAFT calls.

    python3 trigger.py --broker localhost --camera-id spillway --confidence 0.9

Publishes to video/<camera-id>/trigger. The camera answers by attaching its
latest thumbnail and asking the server for permission to stream; it does NOT
start streaming on this alone. That separation is the point: detection is local
and cheap, transmission is shared and expensive, so the decision to spend the
link belongs to whoever can see all the cameras at once.

From WAFT's camera_event_detector.py, the equivalent is three lines:

    import paho.mqtt.publish as publish
    publish.single(f"video/{camera_id}/trigger",
                   json.dumps({"trigger": "waft_motion", "confidence": score,
                               "boxes": boxes, "requested_seconds": 30}),
                   hostname=broker)
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Raise a detection event on a camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--broker", default="localhost")
    p.add_argument("--broker-port", type=int, default=1883)
    p.add_argument("--camera-id", required=True)
    p.add_argument("--trigger", default="waft_motion", help="what detected it")
    p.add_argument("--confidence", type=float, default=0.85)
    p.add_argument("--boxes", help='JSON list of [x1,y1,x2,y2], e.g. "[[10,20,110,140]]"')
    p.add_argument("--labels", help='JSON list of labels, e.g. "[\\"books\\"]"')
    p.add_argument("--seconds", type=float, default=30.0, help="streaming window to request")
    p.add_argument("--repeat", type=int, default=1, help="raise this many events")
    p.add_argument("--interval", type=float, default=5.0, help="seconds between repeats")
    args = p.parse_args(argv)

    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("paho-mqtt is not installed.", file=sys.stderr)
        return 1

    payload = {
        "trigger": args.trigger,
        "confidence": args.confidence,
        "requested_seconds": args.seconds,
    }
    for field, raw in (("boxes", args.boxes), ("labels", args.labels)):
        if raw:
            try:
                payload[field] = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"--{field}: {e}", file=sys.stderr)
                return 2

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"trigger-{int(time.time())}")
    try:
        client.connect(args.broker, args.broker_port, keepalive=30)
    except OSError as e:
        print(f"Could not reach broker {args.broker}:{args.broker_port}: {e}", file=sys.stderr)
        return 1
    client.loop_start()

    topic = f"video/{args.camera_id}/trigger"
    for i in range(args.repeat):
        if i:
            time.sleep(args.interval)
        info = client.publish(topic, json.dumps(payload), qos=1)
        info.wait_for_publish(5)
        print(f"raised {args.trigger} on {args.camera_id} "
              f"(confidence {args.confidence}, requesting {args.seconds:.0f}s)")

    client.disconnect()
    client.loop_stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
