from typing import Callable, Dict

import torch
from torchvision.models import shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0

from models.mobilenetv2 import MobileNetV2
from models.resnet import resnet18, resnet34, resnet50, resnet101
from models.rexnetv1 import ReXNetV1
from models.shufflenet import ShuffleNet
from models.shufflenetv2 import ShuffleNetV2
from models.squeezenet import squeezenet1_0, squeezenet1_1


def build_keypoint_model(model_name: str, num_classes: int = 42, img_size: int = 256) -> torch.nn.Module:
    """按项目已有模型命名创建 21 关键点回归网络。

    说明：
    - 这里复用 `train.py` / `inference.py` 中已经支持的骨干网络集合；
    - 分析脚本只需要推理与参数统计，因此默认不加载 ImageNet 预训练权重；
    - 返回模型的输出维度固定为 `num_classes`，默认 42，即 21 个关键点的 x/y 坐标。
    """
    builders: Dict[str, Callable[[], torch.nn.Module]] = {
        "resnet_18": lambda: resnet18(pretrained=False, num_classes=num_classes, img_size=img_size),
        "resnet_34": lambda: resnet34(pretrained=False, num_classes=num_classes, img_size=img_size),
        "resnet_50": lambda: resnet50(pretrained=False, num_classes=num_classes, img_size=img_size),
        "resnet_101": lambda: resnet101(pretrained=False, num_classes=num_classes, img_size=img_size),
        "squeezenet1_0": lambda: squeezenet1_0(pretrained=False, num_classes=num_classes),
        "squeezenet1_1": lambda: squeezenet1_1(pretrained=False, num_classes=num_classes),
        "shufflenetv2": lambda: ShuffleNetV2(ratio=1.0, num_classes=num_classes),
        "shufflenet": lambda: ShuffleNet(num_blocks=[2, 4, 2], num_classes=num_classes, groups=3),
        "shufflenet_v2_x1_5": lambda: shufflenet_v2_x1_5(pretrained=False, num_classes=num_classes),
        "shufflenet_v2_x1_0": lambda: shufflenet_v2_x1_0(pretrained=False, num_classes=num_classes),
        "shufflenet_v2_x2_0": lambda: shufflenet_v2_x2_0(pretrained=False, num_classes=num_classes),
        "mobilenetv2": lambda: MobileNetV2(num_classes=num_classes),
        "ReXNetV1": lambda: ReXNetV1(num_classes=num_classes),
    }
    if model_name not in builders:
        supported = ", ".join(sorted(builders))
        raise ValueError("不支持的模型名称：{}。可选模型：{}".format(model_name, supported))
    return builders[model_name]()


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """加载关键点模型权重，兼容纯 state_dict 与常见 checkpoint 字典。

    训练脚本通常保存 `model.state_dict()`；部分外部训练流程可能会保存
    `{"model": state_dict}`、`{"state_dict": state_dict}` 或
    `{"model_state_dict": state_dict}`。这里做轻量兼容，避免分析脚本因为
    checkpoint 包装格式不同而中断。
    """
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "model" in payload:
        payload = payload["model"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    elif isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    model.load_state_dict(payload)


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型参数总量，用于速度-体积-精度权衡图。"""
    return int(sum(parameter.numel() for parameter in model.parameters()))
