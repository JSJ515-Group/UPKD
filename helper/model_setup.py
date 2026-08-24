import importlib
import os

import torch

from models import build_feature_transfer_model, build_model, model_dict
from setting import teacher_model_config_dict


def _dataset_info(dataset):
    if dataset == "cifar100":
        return 100, "cifar100"
    if dataset == "cifar10":
        return 10, "cifar10"
    if dataset == "tiny_imagenet":
        return 200, "tiny_imagenet"
    raise ValueError(f"Unsupported dataset: {dataset}")


def _student_name(model_name, num_classes):
    name = model_name.lower()
    aliases = {
        "resnet18": (
            "resnet18_imagenet" if num_classes == 200 else "resnet18_cifar"
        ),
        "resnet34": (
            "resnet34_imagenet" if num_classes == 200 else "resnet34_cifar"
        ),
        "resnet50": (
            "resnet50_imagenet" if num_classes == 200 else "resnet50_cifar"
        ),
        "mobilenetv2": (
            "mobilenet_v2_imagenet" if num_classes == 200
            else "mobilenet_v2_half"
        ),
        "mobile": (
            "mobilenet_v2_imagenet" if num_classes == 200
            else "mobilenet_v2_half"
        ),
        "shufflev1": "shufflenet_v1",
        "shufflev2": (
            "shufflenet_v2_imagenet" if num_classes == 200
            else "shufflenet_v2"
        ),
    }
    return aliases.get(name, name)


def _checkpoint_state(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model", "state_dict", "model_state_dict", "net"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def _clean_state_dict(state_dict, model):
    target_keys = set(model.state_dict())
    uses_linear = "linear.weight" in target_keys
    uses_fc = "fc.weight" in target_keys
    uses_shortcut = any(".shortcut." in key for key in target_keys)
    uses_downsample = any(".downsample." in key for key in target_keys)
    cleaned = {}

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue

        removing = True
        while removing:
            removing = False
            for prefix in ("module.", "model.", "teacher.", "net."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    removing = True

        if uses_linear:
            if key == "fc.weight":
                key = "linear.weight"
            elif key == "fc.bias":
                key = "linear.bias"
        elif uses_fc:
            if key == "linear.weight":
                key = "fc.weight"
            elif key == "linear.bias":
                key = "fc.bias"

        if uses_shortcut:
            key = key.replace(".downsample.0.", ".shortcut.0.")
            key = key.replace(".downsample.1.", ".shortcut.1.")
        elif uses_downsample:
            key = key.replace(".shortcut.0.", ".downsample.0.")
            key = key.replace(".shortcut.1.", ".downsample.1.")

        cleaned[key] = value

    return cleaned


def _build_tiny_teacher(model_name, num_classes):
    name = model_name.lower()

    if name in {"resnet18", "resnet34", "resnet50"}:
        from models import resnetv2_org

        constructors = {
            "resnet18": resnetv2_org.ResNet18,
            "resnet34": resnetv2_org.ResNet34,
            "resnet50": resnetv2_org.ResNet50,
        }
        return constructors[name](num_classes=num_classes)

    if name in model_dict:
        return model_dict[name](num_classes=num_classes)

    if name.startswith("wrn_"):
        parts = name.split("_")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid WideResNet name '{model_name}'. "
                "Use a name such as 'wrn_40_2'."
            )
        from models.wide_resnet_cifar import wrn

        return wrn(
            depth=int(parts[1]),
            widen_factor=int(parts[2]),
            num_classes=num_classes,
        )

    raise ValueError(f"Unsupported Tiny-ImageNet teacher: {model_name}")


def _load_tiny_teacher(model_name, checkpoint_path, num_classes, device):
    teacher = _build_tiny_teacher(model_name, num_classes)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _clean_state_dict(_checkpoint_state(checkpoint), teacher)

    try:
        teacher.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Teacher '{model_name}' is incompatible with checkpoint "
            f"'{checkpoint_path}'.\n{error}"
        ) from error

    return teacher.to(device).eval()


def _load_configured_teacher(dataset, teacher_name, device):
    config = teacher_model_config_dict[dataset][teacher_name]
    module = importlib.import_module(config["module"])
    constructor = getattr(module, config["class"])
    kwargs = config.get("kwargs", {})
    teacher = constructor(num_classes=config["num_classes"], **kwargs)

    if "ckpt" in config:
        checkpoint = torch.load(config["ckpt"], map_location="cpu")
        state_dict = _checkpoint_state(checkpoint)
        cleaned = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[7:]

            if config["module"] == "models.resnet":
                if key == "fc.weight":
                    key = "linear.weight"
                elif key == "fc.bias":
                    key = "linear.bias"
                key = key.replace(".downsample.0.", ".shortcut.0.")
                key = key.replace(".downsample.1.", ".shortcut.1.")

            cleaned[key] = value

        teacher.load_state_dict(cleaned, strict=True)

    return teacher.to(device).eval()


def build_models(args, device):
    num_classes, teacher_dataset = _dataset_info(args.dataset)

    if args.use_feature_transfer:
        student = build_feature_transfer_model(
            args.model_s,
            num_classes=num_classes,
            teacher=args.teacher,
        )
    else:
        student = build_model(
            _student_name(args.model_s, num_classes),
            num_classes=num_classes,
        )
    student = student.to(device)

    if teacher_dataset == "tiny_imagenet":
        checkpoint_path = teacher_model_config_dict[
            teacher_dataset
        ][args.teacher]["ckpt"]
        teacher = _load_tiny_teacher(
            args.teacher,
            checkpoint_path,
            num_classes,
            device,
        )
    else:
        teacher = _load_configured_teacher(
            teacher_dataset,
            args.teacher,
            device,
        )

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    return student, teacher
