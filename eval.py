import torch as th
import torch.nn as nn
from torchvision import datasets, transforms
import model
from utils.dataloader import get_dataloaders
from PIL import Image
import argparse
import os

def pil_to_rgb(img):
    return img.convert('RGB')

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/config_1.yaml')
    args = parser.parse_args()
    config_basename = os.path.splitext(os.path.basename(args.config))[0]

    # Load the model
    myresnet = model.resnet18(pretrained=True, finetuning=False, root=f'./saved_models/best_model_{config_basename}.pth')
    print(f"Using config: {config_basename}")

    # Load the test dataset
    data_root = './data'

    transform = transforms.Compose([
        transforms.Lambda(pil_to_rgb),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.534, 0.507, 0.476],  # Caltech-101 dataset mean
                            std=[0.299, 0.293, 0.305])    # Caltech-101 dataset std
    ])

    dataset = datasets.Caltech101(root=data_root, download=False, transform=transform)
    _ , val_loader = get_dataloaders(dataset, batch_size=64, val_split=0.2) 
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    myresnet.to(device)
    myresnet.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with th.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = myresnet(inputs)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    print(f"Validation Accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    evaluate()