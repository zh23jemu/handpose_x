from importlib.util import find_spec

from gesture_runtime.backends.legacy_backend import LegacyHandPoseBackend
from gesture_runtime.backends.mediapipe_backend import MediaPipeHandBackend
from gesture_runtime.config import BackendConfig


def create_backend(config: BackendConfig):
    backend_name = config.name.lower()
    if backend_name == "auto":
        if find_spec("mediapipe") is not None:
            return MediaPipeHandBackend(config)
        return LegacyHandPoseBackend(config)
    if backend_name in {"mediapipe", "mp"}:
        return MediaPipeHandBackend(config)
    if backend_name in {"legacy", "skin", "handpose_x"}:
        return LegacyHandPoseBackend(config)
    raise ValueError("unsupported backend: {}".format(config.name))
