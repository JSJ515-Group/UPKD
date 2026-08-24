teacher_model_config_dict = {
    'cifar100': {
        'resnet56': {
            'module': 'models.resnet_cifar',
            'class': 'resnet56',
            'num_classes': 100,
            'ckpt': './save/teachers/CIFAR100/resnet56_best.pth'
        },
        'resnet110': {
            'module': 'models.resnet_cifar',
            'class': 'resnet110',
            'num_classes': 100,
            'ckpt': './save/teachers/CIFAR100/resnet110_best.pth'
        },
        'resnet32x4': {
            'module': 'models.resnet_cifar',
            'class': 'resnet32x4',
            'num_classes': 100,
            'ckpt': './save/teachers/CIFAR100/resnet32x4_best.pth'
        },
        'wrn_40_2': {
            'module': 'models.wide_resnet_cifar',
            'class': 'WideResNet',
            'kwargs': {'depth': 40, 'widen_factor': 2, 'dropRate': 0.0},
            'num_classes': 100,
            'ckpt': './save/teachers/CIFAR100/wrn_40_2_best.pth'
        },
        'vgg13': {
            'module': 'models.vgg',
            'class': 'vgg13_bn',
            'num_classes': 100,
            'ckpt': './save/teachers/CIFAR100/vgg13_best.pth'
        },
    },

    'cifar10': {
        'resnet34': {
            'module': 'models.resnet',
            'class': 'ResNet34',
            'num_classes': 10,
            'ckpt': './save/teachers/CIFAR10/cifar10_resnet34.pth'
        },
        'wrn_40_2': {
            'module': 'models.wide_resnet_cifar',
            'class': 'WideResNet',
            'kwargs': {'depth': 40, 'widen_factor': 2, 'dropRate': 0.0},
            'num_classes': 10,
            'ckpt': './save/teachers/CIFAR10/cifar10_wrn40_2.pth'
        },
    },

    'tiny_imagenet': {
        'wrn_40_2': {
                'ckpt': 'save/teachers/Tiny-ImageNet/wrn_40_2_best.pth',
        },

        'vgg13': {
                'ckpt': 'save/teachers/Tiny-ImageNet/vgg13_best.pth',
        },
    }
}