import torch as th
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import model
from utils.dataloader import get_dataloaders
from PIL import Image
import argparse
import yaml
import os
import time

def pil_to_rgb(img):
    return img.convert('RGB')

def train():
    """
    Train a ResNet-18 model on the Caltech-101 dataset.
    The model is pretrained on ImageNet and then finetuned on Caltech-101.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/config_1.yaml')
    args = parser.parse_args()
    config_basename = os.path.splitext(os.path.basename(args.config))[0]
    print(f"Using config: {config_basename}")

    writer = SummaryWriter(log_dir=f"runs/{config_basename}/" + time.strftime("%Y%m%d-%H%M%S"))
    print(f"TensorBoard logs will be saved to: {writer.log_dir}")

    with open(args.config, 'r') as f:
        params = yaml.safe_load(f)
        print(f"Parameters: {params}")

    myresnet = model.resnet18(pretrained = params["finetuning"])

    # Finetuning the backbone, training a new fully connected layer.
    fc_params = list(myresnet.fc.parameters())
    backbone_params = [p for name, p in myresnet.named_parameters() if "fc" not in name]

    optimizer = th.optim.SGD([
        {"params": backbone_params, "lr": params["finetuning_lr"]},     # Backbone, smaller lr
        {"params": fc_params, "lr": params["new_fc_lr"]}                # New fully connected layer, larger lr
    ], momentum=params["momentum"], weight_decay=params["weight_decay"])

    # Training in GPU
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    myresnet.to(device)

    dummy_input = th.randn(1, 3, 224, 224).to(device)
    writer.add_graph(myresnet, dummy_input)

    # load data
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
    train_loader, val_loader = get_dataloaders(dataset, batch_size=64, val_split=0.2)

    # Training loop
    best_val_loss = float('inf')
    patience = 2  # Early stopping patience
    for epoch in range(params["num_epochs"]):
        myresnet.train()
        train_loss_sum = 0.0
        train_samples = 0
        train_correct = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = myresnet(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)
            _, preds = outputs.max(1)
            train_correct += (preds == labels).sum().item()

        avg_train_loss = train_loss_sum / train_samples
        train_accuracy = train_correct / train_samples

        # Validation
        myresnet.eval()
        val_loss_sum = 0.0
        val_samples = 0
        val_correct = 0
        with th.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = myresnet(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_loss_sum += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
        avg_val_loss = val_loss_sum / val_samples
        val_accuracy = val_correct / val_samples

        print(f"Epoch [{epoch+1}/{params['num_epochs']}], "
            f"TrainLoss: {avg_train_loss:.4f}, TrainAcc: {train_accuracy:.4f}, "
            f"ValLoss: {avg_val_loss:.4f}, ValAcc: {val_accuracy:.4f}")
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            th.save(myresnet.state_dict(), f'./saved_models/best_model_{config_basename}.pth')
        else:
            trigger_times += 1
            print(f'No improvement. Early stopping trigger: {trigger_times}/{patience}')
        if trigger_times >= patience:
            print('Early stopping!')
            break
    
    writer.close()
        
if __name__ == "__main__":
    train()