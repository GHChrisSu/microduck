#!/usr/bin/env python3
"""Detect a named YOLO class in the MuJoCo duck camera and optionally approach it.

The first run is perception-only:

    "$DUCK_SIM_RL/.venv/bin/python" scripts/duck_vision_follow.py --target person

Add --drive to enable a deliberately slow, finite simulation-only follow:

    "$DUCK_SIM_RL/.venv/bin/python" scripts/duck_vision_follow.py --target person --drive

The body server sends length-prefixed 640x360 UYVY frames on port 7901. The simulator renders an
upright landscape view, so `--rotate` defaults to 0; the physical camera's mount correction remains
90 degrees. Commands go to the local robotd Unix socket and use its normal deadman: if this process
stops refreshing a command, the duck stops within 500 ms. This script never reaches a physical
robot.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import struct
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from follow_policy import FEATURE_COLUMNS

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
FRAME_BYTES = CAMERA_WIDTH * CAMERA_HEIGHT * 2  # UYVY: two bytes per pixel.
DEFAULT_ROBOT_SOCKET = Path.home() / ".cache/duck-sim/duck-a.sock"
DEFAULT_MODEL = Path.home() / ".cache/duck-sim/vision/yolo26n.pt"
DEFAULT_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"


def read_exact(stream: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = stream.recv(size - len(data))
        if not chunk:
            raise ConnectionError("camera stream closed")
        data.extend(chunk)
    return bytes(data)


def read_camera_frame(stream: socket.socket, rotate: int) -> np.ndarray:
    size = struct.unpack("<I", read_exact(stream, 4))[0]
    if size != FRAME_BYTES:
        raise ValueError(f"expected a 640x360 UYVY frame ({FRAME_BYTES} bytes), got {size}")
    uyvy = np.frombuffer(read_exact(stream, size), dtype=np.uint8).reshape(
        CAMERA_HEIGHT, CAMERA_WIDTH, 2
    )
    bgr = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    if rotate == 90:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
    elif rotate == 270:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


def send_move(stream: socket.socket, vx: float, vyaw: float) -> None:
    message = {
        "jsonrpc": "2.0",
        "method": "robot.move",
        "params": {"vx": vx, "vy": 0.0, "vyaw": vyaw},
    }
    stream.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode())


def send_look(stream: socket.socket, height: float) -> dict:
    """Aim the head camera at a forward point around the person's torso."""
    message = {
        "jsonrpc": "2.0",
        "id": "vision-follow-look",
        "method": "robot.look",
        "params": {"x": 1.0, "y": 0.0, "z": height},
    }
    previous_timeout = stream.gettimeout()
    stream.settimeout(3.0)
    try:
        stream.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode())
        response = bytearray()
        while b"\n" not in response:
            chunk = stream.recv(4096)
            if not chunk:
                raise ConnectionError("robotd closed while setting the camera gaze")
            response.extend(chunk)
        result = json.loads(bytes(response).split(b"\n", 1)[0])
        if "error" in result:
            raise RuntimeError(f"robot.look failed: {result['error']}")
        return result.get("result", {})
    finally:
        stream.settimeout(previous_timeout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use YOLO to find a named class in the simulated duck camera."
    )
    parser.add_argument("--camera-host", default="127.0.0.1")
    parser.add_argument("--camera-port", type=int, default=7901)
    parser.add_argument("--robot-socket", type=Path, default=DEFAULT_ROBOT_SOCKET)
    parser.add_argument(
        "--look-height",
        type=float,
        default=0.15,
        help="aim above the trunk so a standing person's body stays in the camera frame",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Ultralytics model or weights path")
    parser.add_argument(
        "--target",
        nargs="+",
        default=["person"],
        metavar="CLASS",
        help='one or more exact model class names, for example --target person bottle "sports ball"',
    )
    parser.add_argument("--confidence", type=float, default=0.08)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--hz", type=float, default=5.0, help="inference and command refresh rate")
    parser.add_argument("--seconds", type=float, default=15.0, help="run-time limit; 0 follows until Ctrl-C")
    parser.add_argument("--drive", action="store_true", help="enable the simulation-only follow controller")
    parser.add_argument(
        "--record-csv",
        type=Path,
        help="record visible-target observations and rule-controller actions as training demonstrations",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        help="TorchScript high-level follow policy; target-loss search remains the existing safe state machine",
    )
    parser.add_argument("--max-forward", type=float, default=0.40, help="forward speed, capped at 0.40 m/s")
    parser.add_argument(
        "--policy-forward-scale",
        type=float,
        default=1.0,
        help="multiply learned forward output before the max-forward safety cap",
    )
    parser.add_argument("--max-yaw", type=float, default=0.95, help="turn command, capped at 0.95 rad/s")
    parser.add_argument("--min-yaw", type=float, default=0.40, help="minimum turn command outside center tolerance")
    parser.add_argument(
        "--center-tolerance",
        type=float,
        default=0.30,
        help="normalized image offset inside which the duck walks forward",
    )
    parser.add_argument(
        "--center-hysteresis",
        type=float,
        default=0.10,
        help="extra offset needed to leave the centered state, to avoid turn/walk chatter",
    )
    parser.add_argument(
        "--stop-area",
        type=float,
        default=0.09,
        help="hold forward motion when the target box covers this fraction of the image",
    )
    parser.add_argument(
        "--stop-height",
        type=float,
        default=0.58,
        help="hold forward motion when the target box reaches this fraction of image height",
    )
    parser.add_argument(
        "--search-delay",
        type=float,
        default=0.6,
        help="wait this long after losing the target before scanning, in seconds",
    )
    parser.add_argument(
        "--search-yaw",
        type=float,
        default=0.22,
        help="in-place scan speed, capped by --max-yaw, in rad/s",
    )
    parser.add_argument(
        "--search-leg-seconds",
        type=float,
        default=3.0,
        help="duration of each left/right scan sweep, in seconds",
    )
    parser.add_argument(
        "--search-pause-seconds",
        type=float,
        default=0.8,
        help="pause between scan sweeps, in seconds",
    )
    args = parser.parse_args()
    if args.policy_forward_scale <= 0.0:
        parser.error("--policy-forward-scale must be positive")
    if args.hz <= 0 or args.seconds < 0:
        parser.error("--hz must be positive and --seconds cannot be negative")
    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidence must be in (0, 1]")
    if not 0.0 < args.center_tolerance < 1.0:
        parser.error("--center-tolerance must be in (0, 1)")
    if not 0.0 <= args.center_hysteresis < 1.0 - args.center_tolerance:
        parser.error("--center-hysteresis must be non-negative and leave room below 1")
    if not 0.0 < args.stop_area < 1.0:
        parser.error("--stop-area must be in (0, 1)")
    if not 0.0 < args.stop_height < 1.0:
        parser.error("--stop-height must be in (0, 1)")
    if args.max_forward < 0 or args.max_yaw < 0 or args.min_yaw < 0:
        parser.error("speed limits cannot be negative")
    if args.search_delay < 0 or args.search_yaw < 0 or args.search_pause_seconds < 0:
        parser.error("search delay, yaw, and pause cannot be negative")
    if args.search_leg_seconds <= 0:
        parser.error("--search-leg-seconds must be positive")
    if args.record_csv is not None and not args.drive:
        parser.error("--record-csv requires --drive so each sample has a teacher action")
    if args.policy is not None and not args.drive:
        parser.error("--policy requires --drive")
    if args.record_csv is not None and args.policy is not None:
        parser.error("record teacher demonstrations without --policy")
    args.max_forward = min(args.max_forward, 0.40)
    args.max_yaw = min(args.max_yaw, 0.95)
    args.min_yaw = min(args.min_yaw, args.max_yaw)
    args.search_yaw = min(args.search_yaw, args.max_yaw)
    return args


def main() -> int:
    args = parse_args()
    if args.model == str(DEFAULT_MODEL) and not DEFAULT_MODEL.exists():
        DEFAULT_MODEL.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pretrained weights to {DEFAULT_MODEL}.", flush=True)
        with urlopen(DEFAULT_MODEL_URL, timeout=30) as response, DEFAULT_MODEL.open("wb") as weights:
            while chunk := response.read(1024 * 1024):
                weights.write(chunk)
    print(f"Loading {args.model}.", flush=True)
    model = YOLO(args.model)
    names = model.names
    class_ids = {
        class_id
        for class_id, name in names.items()
        if name.casefold() in {target.casefold() for target in args.target}
    }
    missing = [target for target in args.target if target.casefold() not in {n.casefold() for n in names.values()}]
    if missing:
        available = ", ".join(str(name) for name in names.values())
        raise SystemExit(f"unknown target class {missing}; model classes: {available}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    follow_policy = None
    if args.policy is not None:
        # Load TorchScript on CPU first; some PyTorch/MPS builds attempt to
        # materialize serialized scalar tensors as float64 during map_location.
        follow_policy = torch.jit.load(str(args.policy)).to(device).eval()
    print(
        f"Model ready · device {device} · targets {', '.join(args.target)} · "
        f"mode {'LEARNED FOLLOW' if follow_policy is not None else 'FOLLOW' if args.drive else 'DETECT ONLY'}",
        flush=True,
    )

    robot: socket.socket | None = None
    camera: socket.socket | None = None
    records = None
    writer = None
    if args.record_csv is not None:
        args.record_csv.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not args.record_csv.exists() or args.record_csv.stat().st_size == 0
        records = args.record_csv.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            records,
            fieldnames=("timestamp", *FEATURE_COLUMNS, "teacher_vx", "teacher_vyaw"),
        )
        if needs_header:
            writer.writeheader()
    if args.drive:
        if not args.robot_socket.exists():
            raise SystemExit(f"robot socket not found: {args.robot_socket}; start scripts/duck-sim first")
        robot = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        robot.connect(str(args.robot_socket))
        gaze = send_look(robot, args.look_height)
        head = gaze.get("head", {})
        print(
            f"Camera gaze set · head pitch {float(head.get('head_pitch', 0.0)):+.2f} rad",
            flush=True,
        )
        send_move(robot, 0.0, 0.0)

    try:
        camera = socket.create_connection((args.camera_host, args.camera_port), timeout=5.0)
        camera.settimeout(5.0)
        print(
            f"Camera {args.camera_host}:{args.camera_port} · {CAMERA_WIDTH}x{CAMERA_HEIGHT} UYVY · "
            f"rotate {args.rotate}° clockwise · press Ctrl-C to stop",
            flush=True,
        )
        start = last_inference = last_report = 0.0
        period = 1.0 / args.hz
        stop_reason = "time limit" if args.seconds > 0 else "Ctrl-C"
        centered = False
        holding_distance = False
        lost_since: float | None = None
        last_seen_bearing = 0.0
        previous_bearing: float | None = None
        previous_detection_at: float | None = None

        while True:
            frame = read_camera_frame(camera, args.rotate)
            now = time.monotonic()
            if start == 0.0:
                start = now
            if args.seconds > 0 and now - start >= args.seconds:
                break
            if now - last_inference < period:
                continue
            last_inference = now

            result = model.predict(
                source=frame,
                imgsz=args.image_size,
                conf=args.confidence,
                device=device,
                verbose=False,
            )[0]
            height, width = frame.shape[:2]
            found: list[tuple[float, float, float, float, float, float, int]] = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    if class_id not in class_ids:
                        continue
                    x0, y0, x1, y1 = (float(v) for v in box.xyxy[0].tolist())
                    confidence = float(box.conf.item())
                    area = max(0.0, x1 - x0) * max(0.0, y1 - y0) / (width * height)
                    found.append((area, confidence, x0, y0, x1, y1, class_id))

            if found:
                area, confidence, x0, y0, x1, y1, class_id = max(found, key=lambda item: item[0])
                center_x = (x0 + x1) / 2.0
                bearing = 2.0 * center_x / width - 1.0  # negative = left; positive = right.
                height_fraction = max(0.0, y1 - y0) / height
                if previous_bearing is None or previous_detection_at is None:
                    bearing_rate = 0.0
                else:
                    elapsed = max(1e-3, now - previous_detection_at)
                    bearing_rate = max(-4.0, min(4.0, (bearing - previous_bearing) / elapsed))
                previous_bearing = bearing
                previous_detection_at = now
                lost_since = None
                last_seen_bearing = bearing
                vx = vyaw = 0.0
                if args.drive:
                    # These are the controller's internal states before this observation. Giving
                    # them to the learner makes the rule teacher's hysteresis observable.
                    centered_before = centered
                    holding_before = holding_distance
                    # A running person's box changes shape from frame to frame. Keep the
                    # standoff state until the box is clearly smaller, instead of alternating
                    # between walking and stopping around one threshold.
                    close = area >= args.stop_area or height_fraction >= args.stop_height
                    resume = (
                        area >= args.stop_area * 0.75
                        or height_fraction >= args.stop_height * 0.85
                    )
                    if holding_distance:
                        holding_distance = resume
                    else:
                        holding_distance = close
                    if holding_distance:
                        centered = abs(bearing) <= args.center_tolerance
                        if not centered:
                            # Hold distance while continuing to keep the person in view.
                            turn_rate = max(args.min_yaw, args.max_yaw * min(1.0, abs(bearing)))
                            vyaw = -turn_rate * (1.0 if bearing > 0 else -1.0)
                    else:
                        # Enter the forward state near the centre, then tolerate modest box jitter
                        # until the error grows past the wider exit boundary.
                        if centered:
                            centered = abs(bearing) <= args.center_tolerance + args.center_hysteresis
                        else:
                            centered = abs(bearing) <= args.center_tolerance
                    if not holding_distance:
                        # Keep walking while steering toward the target. The walking
                        # policy already turns and translates together; stopping forward
                        # motion whenever the person is off-centre made it spin in place.
                        vx = args.max_forward
                        if not centered:
                            offset = min(1.0, abs(bearing))
                            # robot.move defines positive vyaw as turning left.
                            turn_rate = max(args.min_yaw, args.max_yaw * offset)
                            vyaw = -turn_rate * (1.0 if bearing > 0 else -1.0)
                    teacher_vx, teacher_vyaw = vx, vyaw
                    features = [
                        bearing,
                        area,
                        height_fraction,
                        confidence,
                        bearing_rate,
                        float(centered_before),
                        float(holding_before),
                    ]
                    if writer is not None:
                        writer.writerow(
                            {
                                "timestamp": f"{now:.6f}",
                                **dict(zip(FEATURE_COLUMNS, features)),
                                "teacher_vx": f"{teacher_vx:.6f}",
                                "teacher_vyaw": f"{teacher_vyaw:.6f}",
                            }
                        )
                        assert records is not None
                        records.flush()
                    if follow_policy is not None:
                        with torch.inference_mode():
                            prediction = follow_policy(
                                torch.tensor([features], dtype=torch.float32, device=device)
                            )[0]
                        scaled_vx = float(prediction[0].item()) * args.policy_forward_scale
                        vx = max(0.0, min(args.max_forward, scaled_vx))
                        vyaw = max(-args.max_yaw, min(args.max_yaw, float(prediction[1].item())))
                        # The range gate is a safety backstop while this first learned policy is
                        # only a behaviour-cloned baseline. It cannot command forward motion
                        # after the teacher's conservative visual standoff threshold is crossed.
                        if holding_distance:
                            vx = 0.0
                    assert robot is not None
                    send_move(robot, vx, vyaw)

                if now - last_report >= 0.5:
                    name = names[class_id]
                    action = f"vx={vx:.2f} vyaw={vyaw:.2f}" if args.drive else "detect only"
                    if follow_policy is not None:
                        action = f"learned {action}"
                    if args.drive and holding_distance:
                        action = f"holding distance · {action}"
                    print(
                        f"{name} conf={confidence:.2f} bearing={bearing:+.2f} "
                        f"box-area={area:.1%} box-height={height_fraction:.1%} · {action}",
                        flush=True,
                    )
                    last_report = now
            else:
                if args.drive:
                    centered = False
                    assert robot is not None
                    if lost_since is None:
                        lost_since = now

                    lost_for = now - lost_since
                    turn = 0.0
                    action = "lost · pausing"
                    if lost_for >= args.search_delay:
                        scan_time = lost_for - args.search_delay
                        cycle = args.search_leg_seconds + args.search_pause_seconds
                        leg = int(scan_time // cycle)
                        leg_time = scan_time - leg * cycle
                        if leg_time < args.search_leg_seconds:
                            # Positive image bearing means the target was to the right, so
                            # begin by turning right (negative robot yaw) to reacquire it.
                            first_direction = -1.0 if last_seen_bearing > 0 else 1.0
                            turn = first_direction * (-1.0 if leg % 2 else 1.0) * args.search_yaw
                            action = f"searching {'left' if turn > 0 else 'right'}"
                        else:
                            action = "search sweep pause"
                    send_move(robot, 0.0, turn)
                if now - last_report >= 0.5:
                    message = f"target not detected · {action}" if args.drive else "target not detected"
                    print(message, flush=True)
                    last_report = now

        if args.drive:
            print(f"Stopping: {stop_reason}.", flush=True)
    except KeyboardInterrupt:
        print("\nStopping: Ctrl-C.", flush=True)
    finally:
        if robot is not None:
            send_move(robot, 0.0, 0.0)
            robot.close()
        if camera is not None:
            camera.close()
        if records is not None:
            records.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
