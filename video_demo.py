import argparse
import glob
import json
import platform
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from gesture_runtime.config import load_runtime_config
from gesture_runtime.pipeline import GesturePipeline
from gesture_runtime.visualization import EventSink, RecentEventBuffer, draw_event_panel, draw_landmarks, draw_observation_overlay


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def open_source(source: str) -> Tuple[Optional[cv2.VideoCapture], Optional[np.ndarray], bool]:
    source_path = Path(source)
    if source_path.is_file() and source_path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(source_path))
        if image is None:
            raise RuntimeError("failed to read image source: {}".format(source))
        return None, image, True
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
    else:
        capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        if source.isdigit() and not glob.glob("/dev/video*"):
            if "microsoft" in platform.release().lower() or "wsl" in platform.version().lower():
                raise RuntimeError(
                    "failed to open source: {}. No Linux camera device was found in this WSL environment "
                    "(/dev/video* is missing). Run the script from native Windows Python, or attach/pass "
                    "through a camera to WSL, or use a video/stream URL as --source.".format(source)
                )
        raise RuntimeError("failed to open source: {}".format(source))
    return capture, None, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime hand gesture recognition demo")
    parser.add_argument("--config", type=str, default="configs/gesture_runtime.yaml", help="runtime yaml/json config")
    parser.add_argument("--source", type=str, default="0", help="camera index, video path, or image path")
    parser.add_argument("--backend", type=str, default="", choices=["", "auto", "mediapipe", "legacy"], help="override backend")
    parser.add_argument("--model-path", type=str, default="", help="legacy handpose model path override")
    parser.add_argument("--device", type=str, default="", choices=["", "auto", "cpu", "cuda"], help="override backend device")
    parser.add_argument("--font-path", type=str, default="", help="optional Chinese font path")
    parser.add_argument("--temporal-model-path", type=str, default="", help="optional temporal gesture checkpoint")
    parser.add_argument("--max-hands", type=int, default=0, help="override max hands")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames, 0 means unlimited")
    parser.add_argument("--min-gesture-confidence", type=float, default=-1.0, help="override minimum static confidence")
    parser.add_argument("--min-event-confidence", type=float, default=-1.0, help="override minimum emitted event confidence")
    parser.add_argument("--print-events", dest="print_events", action="store_true", help="print JSON events to stdout")
    parser.add_argument("--no-print-events", dest="print_events", action="store_false", help="disable JSON event output")
    parser.add_argument("--display", dest="display", action="store_true", help="show realtime window")
    parser.add_argument("--no-display", dest="display", action="store_false", help="disable window display")
    parser.add_argument("--mirror", dest="mirror", action="store_true", help="mirror frames horizontally")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false", help="keep original frame orientation")
    parser.add_argument("--event-log", type=str, default="", help="optional JSONL event log path")
    parser.set_defaults(print_events=True, display=True, mirror=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_runtime_config(args.config if args.config else "")
    if args.backend:
        config.backend.name = args.backend
    if args.model_path:
        config.backend.model_path = args.model_path
    if args.device:
        config.backend.device = args.device
    if args.max_hands > 0:
        config.backend.max_hands = args.max_hands
    if args.font_path:
        config.display.font_path = args.font_path
    if args.temporal_model_path:
        config.recognizer.temporal_model_path = args.temporal_model_path
    if args.min_gesture_confidence >= 0.0:
        config.recognizer.min_gesture_confidence = args.min_gesture_confidence
    if args.min_event_confidence >= 0.0:
        config.recognizer.min_event_confidence = args.min_event_confidence
    config.display.print_events = args.print_events
    config.display.display = args.display
    config.display.mirror = args.mirror
    if args.event_log:
        config.display.event_log = args.event_log

    pipeline = None
    sink = EventSink(print_events=config.display.print_events, event_log=config.display.event_log)
    recent_events = RecentEventBuffer(maxlen=6)
    capture = None

    try:
        pipeline = GesturePipeline(config)
        capture, image_frame, single_image = open_source(args.source)
        frame_index = 0
        while True:
            if single_image:
                frame = image_frame.copy()
                if frame_index > 0:
                    break
            else:
                ret, frame = capture.read()
                if not ret:
                    break
            if config.display.mirror:
                frame = cv2.flip(frame, 1)
            frame_index += 1
            timestamp = time.time()
            observations, events = pipeline.process_frame(frame, timestamp=timestamp)
            for observation in observations:
                draw_landmarks(frame, observation.landmarks)
                draw_observation_overlay(frame, observation, font_path=config.display.font_path)
            for event in events:
                if event["confidence"] < config.recognizer.min_event_confidence:
                    continue
                event["frame_index"] = frame_index
                sink.emit(event)
                recent_events.append(event)
            draw_event_panel(frame, recent_events.as_list(), font_path=config.display.font_path)
            cv2.putText(
                frame,
                "Q: quit",
                (frame.shape[1] - 100, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if config.display.display:
                cv2.imshow("Hand Gesture Output", frame)
                wait_delay = 0 if single_image else 1
                key = cv2.waitKey(wait_delay) & 0xFF
                if key == ord("q") or key == 27:
                    break
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        sink.close()
        if pipeline is not None:
            pipeline.close()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
