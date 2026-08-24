"""
Tiny-ImageNet loader using the original image folders.

Important:
1. Labels are assigned using alphabetically sorted WNIDs.
2. This mapping matches the downloaded VGG13 teacher checkpoint.
3. The public API remains compatible with train.py:
       get_tiny_imagenet_dataloaders(...)
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ============================================================
# 1. Dataset statistics and transforms
# ============================================================

TINY_IMAGENET_MEAN = [0.485, 0.456, 0.406]
TINY_IMAGENET_STD = [0.229, 0.224, 0.225]


transform_train = transforms.Compose([
    transforms.RandomCrop(
        size=64,
        padding=8,
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=TINY_IMAGENET_MEAN,
        std=TINY_IMAGENET_STD,
    ),
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=TINY_IMAGENET_MEAN,
        std=TINY_IMAGENET_STD,
    ),
])


# ============================================================
# 2. Tiny-ImageNet raw-folder dataset
# ============================================================

class TinyImageNetRaw(Dataset):
    """
    Read Tiny-ImageNet directly from:

        tiny-imagenet-200/
        ├── train/
        │   ├── n01443537/
        │   │   └── images/
        │   └── ...
        ├── val/
        │   ├── images/
        │   └── val_annotations.txt
        └── wnids.txt

    The critical point is:

        class_order = sorted(wnids)

    This matches the tested VGG13 teacher checkpoint.
    """

    def __init__(
        self,
        root: str = "./data/tiny-imagenet-200",
        split: str = "train",
        transform=None,
        return_index: bool = False,
    ):
        super().__init__()

        if split not in {"train", "val"}:
            raise ValueError(
                f"split must be 'train' or 'val', got: {split}"
            )

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.return_index = return_index

        self.wnids_file = self.root / "wnids.txt"
        self.train_dir = self.root / "train"
        self.val_dir = self.root / "val"
        self.val_images_dir = self.val_dir / "images"
        self.val_annotations_file = (
            self.val_dir / "val_annotations.txt"
        )

        self._check_basic_paths()

        # ----------------------------------------------------
        # Critical mapping:
        # alphabetically sorted WNIDs
        # ----------------------------------------------------
        with open(
            self.wnids_file,
            "r",
            encoding="utf-8",
        ) as file:
            wnids = [
                line.strip()
                for line in file
                if line.strip()
            ]

        if len(wnids) != 200:
            raise RuntimeError(
                "Tiny-ImageNet must contain 200 classes, "
                f"but wnids.txt contains {len(wnids)}"
            )

        if len(set(wnids)) != 200:
            raise RuntimeError(
                "Duplicate WNIDs were found in wnids.txt"
            )

        self.classes: List[str] = sorted(wnids)

        self.class_to_idx = {
            wnid: index
            for index, wnid in enumerate(self.classes)
        }

        self.idx_to_class = {
            index: wnid
            for wnid, index in self.class_to_idx.items()
        }

        self.samples: List[Tuple[Path, int]] = []

        if self.split == "train":
            self._build_train_samples()
        else:
            self._build_val_samples()

        expected_samples = (
            100000 if self.split == "train" else 10000
        )

        if len(self.samples) != expected_samples:
            raise RuntimeError(
                f"{self.split} sample count is incorrect: "
                f"expected {expected_samples}, "
                f"found {len(self.samples)}"
            )

        # Common torchvision-compatible attributes
        self.targets = [
            label for _, label in self.samples
        ]

    def _check_basic_paths(self):
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Tiny-ImageNet root was not found: {self.root}"
            )

        if not self.wnids_file.is_file():
            raise FileNotFoundError(
                f"wnids.txt was not found: {self.wnids_file}"
            )

        if not self.train_dir.is_dir():
            raise FileNotFoundError(
                f"Train directory was not found: {self.train_dir}"
            )

        if not self.val_dir.is_dir():
            raise FileNotFoundError(
                f"Validation directory was not found: {self.val_dir}"
            )

    def _build_train_samples(self):
        """
        Build:
            train/<wnid>/images/*.JPEG

        WNIDs are traversed in sorted order.
        """
        for wnid in self.classes:
            class_dir = self.train_dir / wnid
            image_dir = class_dir / "images"

            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Training class directory missing: {class_dir}"
                )

            if not image_dir.is_dir():
                raise FileNotFoundError(
                    f"Training image directory missing: {image_dir}"
                )

            label = self.class_to_idx[wnid]

            image_paths = sorted([
                path
                for path in image_dir.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in {".jpeg", ".jpg", ".png"}
                )
            ])

            if len(image_paths) != 500:
                raise RuntimeError(
                    f"Class {wnid} should contain 500 images, "
                    f"but found {len(image_paths)}"
                )

            for image_path in image_paths:
                self.samples.append(
                    (image_path, label)
                )

    def _build_val_samples(self):
        """
        Build validation samples using val_annotations.txt.
        """
        if not self.val_images_dir.is_dir():
            raise FileNotFoundError(
                "Validation image directory was not found: "
                f"{self.val_images_dir}"
            )

        if not self.val_annotations_file.is_file():
            raise FileNotFoundError(
                "Validation annotation file was not found: "
                f"{self.val_annotations_file}"
            )

        annotation_records = []

        with open(
            self.val_annotations_file,
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                parts = line.strip().split("\t")

                if len(parts) < 2:
                    raise RuntimeError(
                        "Invalid validation annotation at "
                        f"line {line_number}: {line!r}"
                    )

                image_name = parts[0]
                wnid = parts[1]

                if wnid not in self.class_to_idx:
                    raise KeyError(
                        f"Validation WNID {wnid} is not "
                        "present in wnids.txt"
                    )

                image_path = (
                    self.val_images_dir / image_name
                )

                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Validation image missing: {image_path}"
                    )

                label = self.class_to_idx[wnid]

                annotation_records.append(
                    (image_name, image_path, label)
                )

        # Stable and deterministic validation order
        annotation_records.sort(
            key=lambda item: item[0]
        )

        self.samples = [
            (image_path, label)
            for _, image_path, label
            in annotation_records
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        if self.return_index:
            return image, label, index

        return image, label


# ============================================================
# 3. Public DataLoader function used by train.py
# ============================================================

def get_tiny_imagenet_dataloaders(
    batch_size: int = 64,
    num_workers: int = 8,
    is_instance: bool = False,
    data_folder: str = "./data/tiny-imagenet-200",
):
    """
    Return Tiny-ImageNet train and validation loaders.

    Compatible with the existing call in train.py:

        train_loader, val_loader = (
            get_tiny_imagenet_dataloaders(
                batch_size=64,
                num_workers=4,
            )
        )
    """

    train_set = TinyImageNetRaw(
        root=data_folder,
        split="train",
        transform=transform_train,
        return_index=is_instance,
    )

    val_set = TinyImageNetRaw(
        root=data_folder,
        split="val",
        transform=transform_test,
        return_index=False,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(
            num_workers > 0
        ),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(
            num_workers > 0
        ),
    )

    if is_instance:
        return (
            train_loader,
            val_loader,
            len(train_set),
        )

    return train_loader, val_loader


# Compatibility alias for repositories using the old spelling
get_tinyimagenet_dataloaders = (
    get_tiny_imagenet_dataloaders
)


# ============================================================
# 4. Standalone smoke test
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the sorted-WNID Tiny-ImageNet loader"
        )
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/tiny-imagenet-200",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_loader, val_loader = (
        get_tiny_imagenet_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_folder=args.data_root,
        )
    )

    train_set = train_loader.dataset
    val_set = val_loader.dataset

    print("\n" + "=" * 70)
    print("TINY-IMAGENET SORTED-MAPPING LOADER TEST")
    print("=" * 70)

    print(f"Data root          : {train_set.root}")
    print(f"Train samples      : {len(train_set)}")
    print(f"Validation samples : {len(val_set)}")
    print(f"Number of classes  : {len(train_set.classes)}")

    print("\nFirst 10 sorted classes:")
    for index, wnid in enumerate(
        train_set.classes[:10]
    ):
        print(f"{index:3d}: {wnid}")

    print("\nLast 5 sorted classes:")
    start = len(train_set.classes) - 5

    for offset, wnid in enumerate(
        train_set.classes[-5:]
    ):
        print(f"{start + offset:3d}: {wnid}")

    if (
        train_set.class_to_idx
        != val_set.class_to_idx
    ):
        raise RuntimeError(
            "Train and validation class mappings differ"
        )

    print("\nTrain/validation class mapping: matched")

    train_images, train_labels = next(
        iter(train_loader)
    )

    val_images, val_labels = next(
        iter(val_loader)
    )

    print("\nFirst train batch:")
    print(
        f"Images shape : "
        f"{tuple(train_images.shape)}"
    )
    print(
        f"Labels shape : "
        f"{tuple(train_labels.shape)}"
    )
    print(
        f"Label range  : "
        f"[{train_labels.min().item()}, "
        f"{train_labels.max().item()}]"
    )
    print(
        f"Image dtype  : "
        f"{train_images.dtype}"
    )

    print("\nFirst validation batch:")
    print(
        f"Images shape : "
        f"{tuple(val_images.shape)}"
    )
    print(
        f"Labels shape : "
        f"{tuple(val_labels.shape)}"
    )
    print(
        f"Label range  : "
        f"[{val_labels.min().item()}, "
        f"{val_labels.max().item()}]"
    )
    print(
        f"Image dtype  : "
        f"{val_images.dtype}"
    )

    print("\nLoader test passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()