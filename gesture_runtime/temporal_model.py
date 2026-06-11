import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gesture_runtime.labels import TEMPORAL_GESTURE_CLASSES


def _distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def _angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
    vec_ba = point_a - point_b
    vec_bc = point_c - point_b
    norm_ba = np.linalg.norm(vec_ba)
    norm_bc = np.linalg.norm(vec_bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 180.0
    cosine = float(np.dot(vec_ba, vec_bc) / (norm_ba * norm_bc))
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    landmarks = np.asarray(landmarks, dtype=np.float32)
    wrist = landmarks[0]
    centered = landmarks - wrist
    palm_anchor = np.mean(landmarks[[0, 5, 9, 13, 17]], axis=0)
    scale = max(
        1e-6,
        _distance(palm_anchor, landmarks[5]),
        _distance(palm_anchor, landmarks[17]),
        _distance(landmarks[0], landmarks[9]),
    )
    normalized = centered / scale
    axis = landmarks[17] - landmarks[5]
    angle = float(np.arctan2(axis[1], axis[0]))
    rotation = np.array(
        [
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle), np.cos(-angle)],
        ],
        dtype=np.float32,
    )
    return normalized @ rotation.T


def extract_frame_features(landmarks: np.ndarray) -> np.ndarray:
    normalized = normalize_landmarks(landmarks)
    palm_size = max(
        1e-6,
        _distance(landmarks[0], landmarks[5]),
        _distance(landmarks[0], landmarks[9]),
        _distance(landmarks[0], landmarks[17]),
    )
    pinch_ratio = _distance(landmarks[4], landmarks[8]) / palm_size
    finger_angles = []
    finger_defs = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    for mcp, pip, dip, tip in finger_defs:
        finger_angles.append(_angle(landmarks[mcp], landmarks[pip], landmarks[dip]) / 180.0)
        finger_angles.append(_angle(landmarks[pip], landmarks[dip], landmarks[tip]) / 180.0)
    tip_distances = [
        _distance(landmarks[index], landmarks[0]) / palm_size
        for index in (4, 8, 12, 16, 20)
    ]
    palm_axis = landmarks[17] - landmarks[5]
    palm_angle = float(np.arctan2(palm_axis[1], palm_axis[0]) / np.pi)
    feature = np.concatenate(
        [
            normalized.reshape(-1),
            np.asarray(finger_angles, dtype=np.float32),
            np.asarray(tip_distances, dtype=np.float32),
            np.asarray([pinch_ratio, palm_angle], dtype=np.float32),
        ]
    )
    return feature.astype(np.float32)


def stack_sequence_features(frames: Sequence[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("frames sequence is empty")
    base_features = [extract_frame_features(frame) for frame in frames]
    stacked = []
    previous = None
    for feature in base_features:
        if previous is None:
            velocity = np.zeros_like(feature)
        else:
            velocity = feature - previous
        stacked.append(np.concatenate([feature, velocity], axis=0))
        previous = feature
    return np.asarray(stacked, dtype=np.float32)


def temporal_class_to_index() -> Dict[str, int]:
    return {label: index for index, label in enumerate(TEMPORAL_GESTURE_CLASSES)}


def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    if len(sequence) == target_length:
        return sequence.astype(np.float32)
    if len(sequence) == 1:
        return np.repeat(sequence.astype(np.float32), target_length, axis=0)
    indices = np.linspace(0, len(sequence) - 1, target_length)
    lower = np.floor(indices).astype(np.int32)
    upper = np.ceil(indices).astype(np.int32)
    alpha = (indices - lower).astype(np.float32)
    result = []
    for low, up, weight in zip(lower, upper, alpha):
        if low == up:
            result.append(sequence[low])
        else:
            result.append((1.0 - weight) * sequence[low] + weight * sequence[up])
    return np.asarray(result, dtype=np.float32)


class TemporalGestureNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 192, num_layers: int = 3, num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(sequence)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)


class GestureSequenceDataset(Dataset):
    def __init__(self, manifest_path: str, window_size: int):
        self.window_size = window_size
        self.label_to_index = temporal_class_to_index()
        self.samples = self._load_manifest(manifest_path)

    def _load_manifest(self, manifest_path: str) -> List[Dict[str, object]]:
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError("manifest not found: {}".format(manifest_path))
        if path.suffix.lower() == ".jsonl":
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            lines = payload["samples"] if isinstance(payload, dict) else payload
        samples = []
        for item in lines:
            label = item["label"]
            if label not in self.label_to_index:
                continue
            if "frames" in item:
                frames = item["frames"]
            elif "sequence_path" in item:
                frames = json.loads(Path(item["sequence_path"]).read_text(encoding="utf-8"))["frames"]
            else:
                continue
            samples.append({"label": label, "frames": frames})
        if not samples:
            raise RuntimeError("no valid gesture samples found in {}".format(manifest_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        frames = [np.asarray(frame["landmarks"], dtype=np.float32) for frame in sample["frames"]]
        sequence = stack_sequence_features(frames)
        sequence = resample_sequence(sequence, self.window_size)
        label_index = self.label_to_index[sample["label"]]
        return torch.from_numpy(sequence), torch.tensor(label_index, dtype=torch.long)


@dataclass
class TemporalTrainConfig:
    train_manifest: str
    val_manifest: str
    output_dir: str
    window_size: int = 32
    batch_size: int = 16
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 192
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.15
    device: str = "auto"
    num_workers: int = 0
    seed: int = 42


def build_temporal_model(window_size: int, hidden_dim: int = 192, num_layers: int = 3, num_heads: int = 4, dropout: float = 0.15) -> TemporalGestureNet:
    dummy = np.zeros((window_size, 118), dtype=np.float32)
    input_dim = dummy.shape[1]
    return TemporalGestureNet(
        input_dim=input_dim,
        num_classes=len(TEMPORAL_GESTURE_CLASSES),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )


class TemporalGestureClassifier:
    def __init__(self, checkpoint_path: str, device: str = "auto", window_size: int = 32):
        if device == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = device
        self.device = torch.device(device_name)
        payload = torch.load(checkpoint_path, map_location=self.device)
        meta = payload.get("meta", {})
        self.window_size = int(meta.get("window_size", window_size))
        self.class_names = list(meta.get("class_names", TEMPORAL_GESTURE_CLASSES))
        self.model = build_temporal_model(
            window_size=self.window_size,
            hidden_dim=int(meta.get("hidden_dim", 192)),
            num_layers=int(meta.get("num_layers", 3)),
            num_heads=int(meta.get("num_heads", 4)),
            dropout=float(meta.get("dropout", 0.15)),
        )
        self.model.load_state_dict(payload["model"])
        self.model.to(self.device).eval()

    def predict(self, sequence: np.ndarray) -> Optional[Tuple[str, float, Dict[str, float]]]:
        if len(sequence) < 4:
            return None
        sequence = resample_sequence(sequence, self.window_size)
        tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor.float())
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        best_index = int(np.argmax(probs))
        scores = {label: float(probs[idx]) for idx, label in enumerate(self.class_names)}
        return self.class_names[best_index], float(probs[best_index]), scores


def train_temporal_model(config: TemporalTrainConfig) -> Path:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if (config.device == "auto" and torch.cuda.is_available()) else config.device if config.device != "auto" else "cpu")
    train_dataset = GestureSequenceDataset(config.train_manifest, config.window_size)
    val_dataset = GestureSequenceDataset(config.val_manifest, config.window_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    model = build_temporal_model(
        window_size=config.window_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_temporal_gesture.pth"
    best_accuracy = -1.0

    for epoch in range(config.epochs):
        model.train()
        for features, labels in train_loader:
            features = features.to(device).float()
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device).float()
                labels = labels.to(device)
                preds = model(features).argmax(dim=1)
                total += int(labels.numel())
                correct += int((preds == labels).sum().item())
        accuracy = correct / max(1, total)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {
                    "model": model.state_dict(),
                    "meta": {
                        "window_size": config.window_size,
                        "class_names": TEMPORAL_GESTURE_CLASSES,
                        "hidden_dim": config.hidden_dim,
                        "num_layers": config.num_layers,
                        "num_heads": config.num_heads,
                        "dropout": config.dropout,
                        "best_accuracy": accuracy,
                    },
                },
                best_path,
            )
    return best_path
