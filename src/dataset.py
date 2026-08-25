import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

FASHION_CLASSES = [
    "t-shirt/top", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle boot",
]


def get_transforms(train=True, dataset="cifar10"):
    if dataset == "fashion_mnist":
        # upsample to 32x32 and repeat to 3 channels so the same model works
        base = [transforms.Resize(32), transforms.Grayscale(num_output_channels=3)]
        aug = [transforms.RandomHorizontalFlip()] if train else []
        return transforms.Compose(
            base + aug + [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.2860] * 3, std=[0.3530] * 3),
            ]
        )

    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def get_dataloaders(data_dir, batch_size=64, num_workers=2, dataset="cifar10"):
    if dataset == "fashion_mnist":
        ds_cls = datasets.FashionMNIST
    elif dataset == "cifar10":
        ds_cls = datasets.CIFAR10
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    train_dataset = ds_cls(
        root=data_dir, train=True, download=True,
        transform=get_transforms(train=True, dataset=dataset),
    )
    val_dataset = ds_cls(
        root=data_dir, train=False, download=True,
        transform=get_transforms(train=False, dataset=dataset),
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, val_loader


def class_names(dataset="cifar10"):
    return FASHION_CLASSES if dataset == "fashion_mnist" else CIFAR10_CLASSES
