# Gesture Pipeline Refactor

## Why this refactor exists

The original `handpose_x` project only predicts 21 hand landmarks after an external hand box has already been found.
That means dynamic gesture failure is usually caused by unstable hand detection, not only by gesture thresholds.

This refactor rebuilds the runtime into four layers:

1. `backend`
   - `mediapipe`: recommended production/default backend when `mediapipe` is installed.
   - `legacy`: current skin-segmentation + ReXNet hand landmark model.

2. `tracker`
   - stable track id assignment
   - landmark smoothing
   - motion continuity

3. `recognizer`
   - static pose analysis
   - rule-based dynamic gesture state machine
   - optional temporal model fusion

4. `training`
   - a future trainable temporal gesture model for sequence classification

## Runtime

Recommended:

```bash
python video_demo.py --source 0 --display --backend auto
```

If `mediapipe` is installed, `auto` chooses it first.
If not installed, it falls back to the legacy backend.

The recommended project runtime is Python 3.11 with `mediapipe==0.10.11`.
If upgrading Python or MediaPipe, re-check import and realtime camera behavior before using the runtime.

## Supported event output

- `swipe_left`
- `swipe_right`
- `swipe_up`
- `swipe_down`
- `zoom_in`
- `zoom_out`
- `rotate_clockwise`
- `rotate_counterclockwise`
- `cross`
- `palm_swipe_left`
- `open_then_fist`
- `fist_hold`
- static gestures: `one`, `ok`, `fist`, `pinch`, `palm_open`, `love`

All events keep both English and Chinese labels and include `action_hints`.

## Temporal training dataset format

The future dynamic gesture dataset should be provided as `json` or `jsonl`.

One sample example:

```json
{
  "label": "swipe_left",
  "frames": [
    {
      "frame_index": 1,
      "landmarks": [[100.0, 120.0], [110.0, 130.0]]
    }
  ]
}
```

Notes:

- each frame must contain 21 landmark points
- each landmark is `[x, y]`
- `label` must match one of the temporal classes in `gesture_runtime/labels.py`
- training uses the temporal sequence model in `train_dynamic_gesture.py`

## Training

```bash
python train_dynamic_gesture.py \
  --train-manifest datasets/dynamic_gesture/train.jsonl \
  --val-manifest datasets/dynamic_gesture/val.jsonl \
  --output-dir gesture_checkpoints
```
