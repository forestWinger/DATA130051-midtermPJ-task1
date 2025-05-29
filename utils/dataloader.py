import torch as th
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(dataset, batch_size=64, val_split=0.2):
    """
    Get train and validation dataloaders for different datasets.

    Args:
        dataset(torchvision.datasets): The dataset to split.
        batch_size (int): Batch size for the dataloaders.
        val_split (float): Fraction of the dataset to use for validation.
        
    Returns:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
    """
    generator = th.Generator().manual_seed(42)  # For reproducibility
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader