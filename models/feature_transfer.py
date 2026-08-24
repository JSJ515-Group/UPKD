"""Feature-transfer wrapper and multi-scale feature loss."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from modules.feature_transfer import FeatureTransfer

from .registry import build_model


def _resolve_student_name(model_name: str, num_classes: int) -> str:
    """Map a command-line model name to a registered backbone name."""
    name = model_name.lower()

    if num_classes == 200:
        aliases = {
            "resnet18": "resnet18_imagenet",
            "resnet34": "resnet34_imagenet",
            "resnet50": "resnet50_imagenet",
            "mobilenetv2": "mobilenet_v2_imagenet",
            "mobile": "mobilenet_v2_imagenet",
            "shufflev1": "shufflenet_v1",
            "shufflev2": "shufflenet_v2_imagenet",
        }
    else:
        aliases = {
            "resnet18": "resnet18_cifar",
            "resnet34": "resnet34_cifar",
            "resnet50": "resnet50_cifar",
            "mobilenetv2": "mobilenet_v2_half",
            "mobile": "mobilenet_v2_half",
            "shufflev1": "shufflenet_v1",
            "shufflev2": "shufflenet_v2",
        }

    return aliases.get(name, name)


def _supports_feature_output(model: nn.Module) -> bool:
    forward = getattr(model, "forward", None)
    code = getattr(forward, "__code__", None)
    return code is not None and "is_feat" in code.co_varnames


def _extract_features_with_hooks(
    student: nn.Module,
    images: Tensor,
) -> tuple[list[Tensor], Tensor]:
    """Extract stage features from a backbone without ``is_feat`` support."""
    hook_modules: list[nn.Module] = []

    stage_names = ("layer1", "layer2", "layer3", "layer4")
    if all(hasattr(student, name) for name in stage_names):
        hook_modules = [getattr(student, name) for name in stage_names]
    elif hasattr(student, "features"):
        feature_layers = student.features
        candidate_indices = [3, 6, 13, len(feature_layers) - 1]
        hook_modules = [
            feature_layers[index]
            for index in candidate_indices
            if 0 <= index < len(feature_layers)
        ]

    if not hook_modules:
        raise ValueError(
            f"Model {type(student).__name__} does not expose feature outputs "
            "or a supported stage structure."
        )

    features: list[Tensor] = []

    def save_output(_module, _inputs, output) -> None:
        if isinstance(output, Tensor):
            features.append(output)

    handles = [module.register_forward_hook(save_output) for module in hook_modules]
    try:
        logits = student(images)
    finally:
        for handle in handles:
            handle.remove()

    spatial_features = [feature for feature in features if feature.ndim == 4]
    if not spatial_features:
        raise ValueError(
            f"No spatial feature maps were extracted from "
            f"{type(student).__name__}."
        )

    return spatial_features, logits


def _forward_with_features(
    student: nn.Module,
    images: Tensor,
) -> tuple[list[Tensor], Tensor]:
    if _supports_feature_output(student):
        output = student(images, is_feat=True)
        if not isinstance(output, (list, tuple)) or len(output) < 2:
            raise ValueError(
                "A feature-aware backbone must return (features, logits)."
            )
        features, logits = output[0], output[1]
    else:
        features, logits = _extract_features_with_hooks(student, images)

    spatial_features = [
        feature
        for feature in features
        if isinstance(feature, Tensor) and feature.ndim == 4
    ]
    if not spatial_features:
        raise ValueError(
            f"Model {type(student).__name__} returned no spatial feature maps."
        )

    return spatial_features, logits


def _build_student(model_name: str, num_classes: int) -> nn.Module:
    registered_name = _resolve_student_name(model_name, num_classes)
    return build_model(registered_name, num_classes=num_classes)


def _infer_spatial_features(
    student: nn.Module,
    num_classes: int,
) -> list[Tensor]:
    image_size = 64 if num_classes == 200 else 32
    device = next(student.parameters()).device
    was_training = student.training
    student.eval()

    with torch.no_grad():
        dummy_images = torch.randn(2, 3, image_size, image_size, device=device)
        features, _ = _forward_with_features(student, dummy_images)

    student.train(was_training)
    return features


def _teacher_channel_template(teacher_name: str | None) -> list[int] | None:
    if teacher_name is None:
        return None

    name = teacher_name.lower()

    if name in {"resnet20", "resnet32", "resnet44", "resnet56", "resnet110"}:
        return [16, 32, 64]
    if name in {"resnet8x4", "resnet32x4"}:
        return [64, 128, 256]
    if name in {"wrn_16_1", "wrn_40_1"}:
        return [16, 32, 64]
    if name in {"wrn_16_2", "wrn_40_2", "wrn_28_4"}:
        return [32, 64, 128]
    if name in {"vgg8", "vgg11", "vgg13", "vgg16", "vgg19"}:
        return [128, 256, 512, 512]
    if name in {"resnet18", "resnet34", "resnet18v2", "resnet34v2"}:
        return [64, 128, 256, 512]
    if name in {"resnet50", "resnet50v2"}:
        return [256, 512, 1024, 2048]
    if "mobile" in name:
        return [24, 32, 96, 320]
    if "shuffle" in name:
        return [64, 128, 256, 256]

    return None


def _default_feature_count(
    model_name: str,
    inferred_features: Sequence[Tensor],
) -> int:
    name = model_name.lower()

    if name.startswith("resnet") and name not in {
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet18v2",
        "resnet34v2",
        "resnet50v2",
    }:
        return min(3, len(inferred_features))
    if "x4" in name or "wrn" in name:
        return min(3, len(inferred_features))

    return min(4, len(inferred_features))


def _select_student_features(
    model_name: str,
    features: Sequence[Tensor],
    teacher_name: str | None,
) -> list[Tensor]:
    spatial_features = [feature for feature in features if feature.ndim == 4]
    teacher_channels = _teacher_channel_template(teacher_name)

    if teacher_channels is None:
        feature_count = _default_feature_count(model_name, spatial_features)
    else:
        feature_count = min(len(teacher_channels), len(spatial_features))

    return spatial_features[-feature_count:]


class FeatureTransferModel(nn.Module):
    """Attach top-down feature-transfer blocks to a student backbone."""

    def __init__(
        self,
        student: nn.Module,
        model_name: str,
        teacher_name: str | None,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.student = student
        self.model_name = model_name
        self.teacher_name = teacher_name
        self.num_classes = num_classes

        inferred_features = _infer_spatial_features(student, num_classes)
        selected_features = _select_student_features(
            model_name,
            inferred_features,
            teacher_name,
        )

        self.in_channels = [feature.shape[1] for feature in selected_features]
        self.shapes = [feature.shape[-1] for feature in selected_features]

        teacher_channels = _teacher_channel_template(teacher_name)
        if teacher_channels is None:
            self.out_channels = list(self.in_channels)
        else:
            self.out_channels = teacher_channels[-len(self.in_channels) :]

        mid_channel = min(512, self.in_channels[-1])
        blocks = nn.ModuleList(
            FeatureTransfer(
                in_channel=in_channel,
                mid_channel=mid_channel,
                out_channel=self.out_channels[index],
                fuse=index < len(self.in_channels) - 1,
            )
            for index, in_channel in enumerate(self.in_channels)
        )
        self.feature_transfer_blocks = blocks[::-1]

    def forward(self, images: Tensor) -> tuple[list[Tensor], Tensor]:
        features, logits = _forward_with_features(self.student, images)
        features = _select_student_features(
            self.model_name,
            features,
            self.teacher_name,
        )

        if len(features) != len(self.feature_transfer_blocks):
            shapes = [tuple(feature.shape) for feature in features]
            raise ValueError(
                "Feature count changed after initialization: "
                f"expected {len(self.feature_transfer_blocks)}, "
                f"received {len(features)} with shapes {shapes}."
            )

        reversed_features = features[::-1]
        transferred_features: list[Tensor] = []

        output, context = self.feature_transfer_blocks[0](reversed_features[0])
        transferred_features.append(output)

        for feature, block in zip(
            reversed_features[1:],
            self.feature_transfer_blocks[1:],
        ):
            output, context = block(feature, context)
            transferred_features.insert(0, output)

        return transferred_features, logits


def build_feature_transfer_model(
    model: str,
    num_classes: int,
    teacher: str | None = None,
) -> FeatureTransferModel:
    """Build a student backbone with feature-transfer blocks."""
    student = _build_student(model, num_classes)
    return FeatureTransferModel(
        student=student,
        model_name=model,
        teacher_name=teacher,
        num_classes=num_classes,
    )


def multi_scale_feature_loss(
    student_features: Sequence[Tensor],
    teacher_features: Sequence[Tensor],
) -> Tensor:
    """Compute weighted MSE at the original and pooled spatial scales."""
    total_loss: Tensor | float = 0.0

    for student_feature, teacher_feature in zip(
        student_features,
        teacher_features,
    ):
        height = student_feature.shape[-2]
        feature_loss = F.mse_loss(student_feature, teacher_feature)
        scale_weight = 1.0
        weight_sum = 1.0

        for output_size in (4, 2, 1):
            if output_size >= height:
                continue

            pooled_student = F.adaptive_avg_pool2d(
                student_feature,
                (output_size, output_size),
            )
            pooled_teacher = F.adaptive_avg_pool2d(
                teacher_feature,
                (output_size, output_size),
            )
            scale_weight /= 2.0
            feature_loss = feature_loss + scale_weight * F.mse_loss(
                pooled_student,
                pooled_teacher,
            )
            weight_sum += scale_weight

        total_loss = total_loss + feature_loss / weight_sum

    if isinstance(total_loss, float):
        raise ValueError("At least one feature pair is required.")

    return total_loss
