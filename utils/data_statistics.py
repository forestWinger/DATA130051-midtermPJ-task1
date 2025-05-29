from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from PIL import Image

def pil_to_rgb(img):
    return img.convert('RGB')

def compute_mean_std(dataset):
    """
    Compute the mean and standard deviation of a dataset.
    Args:
        dataset: A PyTorch dataset.
    Returns:
        mean: A tensor containing the mean of each channel.
        std: A tensor containing the standard deviation of each channel.
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    n_images = 0
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)

    for images, _ in tqdm(loader):
        n_images += images.size(0)
        channel_sum += images.sum(dim=[0, 2, 3])  # sum over batch, height, width
        channel_squared_sum += (images ** 2).sum(dim=[0, 2, 3])

    # Calculate mean and std
    mean = channel_sum / (n_images * 224 * 224)
    std = (channel_squared_sum / (n_images * 224 * 224) - mean ** 2).sqrt()

    return mean, std

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Lambda(pil_to_rgb),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    dataset = datasets.Caltech101(root='./data', download=False, transform=transform)
    mean, std = compute_mean_std(dataset)
    print(f"Caltech-101 's mean: {mean}")
    print(f"Caltech-101 's std:  {std}")