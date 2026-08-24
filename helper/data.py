def build_dataloaders(args):
    print(f"Dataset: {args.dataset}")

    if args.dataset == "cifar100":
        from dataset.cifar100 import get_cifar100_dataloaders

        return get_cifar100_dataloaders(
            data_folder=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    if args.dataset == "cifar10":
        from dataset.cifar10 import get_cifar10_dataloaders

        return get_cifar10_dataloaders(
            data_folder=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    if args.dataset == "tiny_imagenet":
        from dataset.tiny_imagenet import get_tiny_imagenet_dataloaders

        return get_tiny_imagenet_dataloaders(
            data_folder=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    raise ValueError(f"Unsupported dataset: {args.dataset}")
