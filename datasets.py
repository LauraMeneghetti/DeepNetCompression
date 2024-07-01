import torch
from torchvision import datasets, transforms

def get_dataloaders(dataset_name, batch_size, input_size, data_dir):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=val_transform)
    
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=val_transform)

    elif dataset_name == 'stl10':
        train_dataset = datasets.STL10(root=data_dir, split="train", download=False, transform=train_transform)
        test_dataset = datasets.STL10(root=data_dir, split="test", download=False, transform=val_transform)
    
    else:
        raise ValueError("Unknown dataset")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset