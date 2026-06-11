from abc import ABC, abstractmethod
from typing import List

import numpy as np

from gesture_runtime.types import HandDetection


class HandPoseBackend(ABC):
    name = "base"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[HandDetection]:
        raise NotImplementedError

    def close(self) -> None:
        return None
