"""Model registry used by the training entry point."""

from collections.abc import Callable

from torch import nn

from .mobilenet_v2 import mobile_half, mobilenet
from .mobilenet_v2_imagenet import mobilenet_v2 as mobilenet_v2_imagenet
from .resnet_cifar import (
    resnet8,
    resnet8x4,
    resnet14,
    resnet20,
    resnet32,
    resnet32x4,
    resnet44,
    resnet56,
    resnet110,
)
from .resnet import ResNet10
from .resnet_tiny_imagenet import (
    ResNet18 as ResNet18TinyImageNet,
    ResNet34 as ResNet34TinyImageNet,
    ResNet50 as ResNet50TinyImageNet,
)
from .shufflenet_v1 import ShuffleV1
from .shufflenet_v2 import ShuffleV2
from .shufflenet_v2_imagenet import shufflenet_v2_x1_0
from .vgg import vgg8_bn, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .wide_resnet import wrn_28_4
from .wide_resnet_cifar import (
    wrn_16_1,
    wrn_16_2,
    wrn_40_1,
    wrn_40_2,
)

ModelBuilder = Callable[..., nn.Module]

MODEL_REGISTRY: dict[str, ModelBuilder] = {
    # CIFAR ResNet backbones.
    "resnet8": resnet8,
    "resnet14": resnet14,
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet44": resnet44,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "resnet8x4": resnet8x4,
    "resnet32x4": resnet32x4,
    # Four-stage ResNet backbones for CIFAR-sized inputs.
    "resnet10": ResNet10,
    "resnet18_tiny_imagenet": ResNet18TinyImageNet,
    "resnet34_tiny_imagenet": ResNet34TinyImageNet,
    "resnet50_tiny_imagenet": ResNet50TinyImageNet,
    # Wide ResNet backbones.
    "wrn_16_1": wrn_16_1,
    "wrn_16_2": wrn_16_2,
    "wrn_28_4": wrn_28_4,
    "wrn_40_1": wrn_40_1,
    "wrn_40_2": wrn_40_2,
    # VGG backbones.
    "vgg8": vgg8_bn,
    "vgg11": vgg11_bn,
    "vgg13": vgg13_bn,
    "vgg16": vgg16_bn,
    "vgg19": vgg19_bn,
    # MobileNetV2 backbones.
    "mobilenet_v2_half": mobile_half,
    "mobilenet_v2": mobilenet,
    "mobilenet_v2_imagenet": mobilenet_v2_imagenet,
    # ShuffleNet backbones.
    "shufflenet_v1": ShuffleV1,
    "shufflenet_v2": ShuffleV2,
    "shufflenet_v2_imagenet": shufflenet_v2_x1_0,
}

# Backward-compatible alias used by the current training script.
model_dict = MODEL_REGISTRY


def build_model(name: str, *, num_classes: int, **kwargs) -> nn.Module:
    """Build a registered model with a consistent error message."""
    try:
        builder = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model '{name}'. Available models: {available}."
        ) from exc

    return builder(num_classes=num_classes, **kwargs)
