import torch
import torch.nn.functional as F

from models import multi_scale_feature_loss


def model_supports_features(model):
    forward = getattr(model, "forward", None)
    code = getattr(forward, "__code__", None)
    return code is not None and "is_feat" in code.co_varnames


def extract_resnet_features(model, inputs):
    modules = [
        getattr(model, name)
        for name in ("layer1", "layer2", "layer3", "layer4")
        if hasattr(model, name)
    ]
    if not modules:
        raise ValueError(
            f"{type(model).__name__} has no feature output or ResNet stages."
        )

    features = []

    def save_feature(_module, _inputs, output):
        features.append(output)

    handles = [module.register_forward_hook(save_feature) for module in modules]
    try:
        logits = model(inputs)
    finally:
        for handle in handles:
            handle.remove()

    features = [
        feature for feature in features
        if isinstance(feature, torch.Tensor) and feature.dim() == 4
    ]
    if not features:
        raise ValueError(
            f"No spatial features were extracted from {type(model).__name__}."
        )
    return features, logits


def forward_teacher(teacher, inputs, preact=False):
    if model_supports_features(teacher):
        try:
            output = teacher(inputs, is_feat=True, preact=preact)
        except TypeError:
            output = teacher(inputs, is_feat=True)

        if not isinstance(output, (list, tuple)) or len(output) < 2:
            raise ValueError(
                "A feature-aware teacher must return (features, logits)."
            )
        features, logits = output[0], output[1]
    else:
        features, logits = extract_resnet_features(teacher, inputs)

    if not isinstance(features, list):
        features = [features]

    features = [
        feature for feature in features
        if isinstance(feature, torch.Tensor) and feature.dim() == 4
    ]
    return features, logits


def get_last_feature(features):
    return features[-1] if isinstance(features, list) else features


def align_logits(teacher_logits, target_dim):
    if teacher_logits.size(1) == target_dim:
        return teacher_logits
    if teacher_logits.size(1) > target_dim:
        return teacher_logits[:, :target_dim]
    return F.pad(teacher_logits, (0, target_dim - teacher_logits.size(1)))


def match_channels(feature, target_channels):
    if feature.size(1) == target_channels:
        return feature
    if feature.size(1) > target_channels:
        return feature[:, :target_channels]
    padding = target_channels - feature.size(1)
    return F.pad(feature, (0, 0, 0, 0, 0, padding), value=0.0)


def vectorize_feature(feature):
    if feature.dim() == 4:
        feature = F.adaptive_avg_pool2d(feature, 1).view(feature.size(0), -1)
    return F.normalize(feature, dim=1)


def match_vector_dimensions(first, second):
    target_dim = min(first.size(1), second.size(1))
    if first.size(1) != target_dim:
        first = F.adaptive_avg_pool1d(
            first.unsqueeze(1), target_dim
        ).squeeze(1)
    if second.size(1) != target_dim:
        second = F.adaptive_avg_pool1d(
            second.unsqueeze(1), target_dim
        ).squeeze(1)
    return first, second


def feature_transfer_loss(student_features, teacher_features, epoch, args):
    count = len(student_features)
    if len(teacher_features) == count + 1:
        teacher_features = teacher_features[1:]
    elif len(teacher_features) > count:
        teacher_features = teacher_features[-count:]

    aligned_teacher = []
    for student_feature, teacher_feature in zip(
        student_features, teacher_features
    ):
        if teacher_feature.shape[-2:] != student_feature.shape[-2:]:
            teacher_feature = F.interpolate(
                teacher_feature,
                size=student_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        aligned_teacher.append(
            match_channels(teacher_feature, student_feature.size(1))
        )

    loss = multi_scale_feature_loss(student_features, aligned_teacher)
    warmup = min(1.0, float(epoch + 1) / float(args.feature_warmup_epochs))
    return loss * warmup, aligned_teacher


def align_feature_lists(student_features, teacher_features):
    aligned_student, aligned_teacher = [], []
    pair_count = min(len(student_features), len(teacher_features))
    student_features = student_features[-pair_count:]
    teacher_features = teacher_features[-pair_count:]

    for student_feature, teacher_feature in zip(
        student_features, teacher_features
    ):
        if student_feature.dim() == 4 or teacher_feature.dim() == 4:
            if student_feature.dim() == 2:
                student_feature = student_feature.unsqueeze(-1).unsqueeze(-1)
            if teacher_feature.dim() == 2:
                teacher_feature = teacher_feature.unsqueeze(-1).unsqueeze(-1)

            if teacher_feature.shape[-2:] != student_feature.shape[-2:]:
                teacher_feature = F.interpolate(
                    teacher_feature,
                    size=student_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            teacher_feature = match_channels(
                teacher_feature, student_feature.size(1)
            )
        elif student_feature.dim() == 2 and teacher_feature.dim() == 2:
            if teacher_feature.size(1) > student_feature.size(1):
                teacher_feature = teacher_feature[:, :student_feature.size(1)]
            elif teacher_feature.size(1) < student_feature.size(1):
                teacher_feature = F.pad(
                    teacher_feature,
                    (0, student_feature.size(1) - teacher_feature.size(1)),
                )
        else:
            raise ValueError(
                "Unsupported feature ranks: "
                f"{student_feature.shape} and {teacher_feature.shape}."
            )

        aligned_student.append(student_feature)
        aligned_teacher.append(teacher_feature)

    return aligned_student, aligned_teacher
