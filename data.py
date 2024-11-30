from torchvision import datasets, transforms
from torch.utils.data import Subset, random_split, DataLoader


def get_data_sets(validation_split: float):
    # MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    full_train_dataset = Subset(
        datasets.MNIST(
            root="./data", train=True, transform=transform, download=True
        ),
        range(1000),
    )

    # Split into training and validation datasets
    train_size = int((1 - validation_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    return train_dataset, val_dataset


def get_data_loaders(batch_size: int, validation_split: float):
    train_dataset, val_dataset = get_data_sets(validation_split)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader
