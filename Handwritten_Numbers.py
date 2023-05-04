# Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms

def load_data_one_hot():
    # Define a custom Flatten transform
    class Flatten:
        def __call__(self, tensor):
            return tensor.view(tensor.shape[0], -1)

    # Define the data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize using mean and std from the MNIST dataset
        Flatten()
    ])

    # Define a custom MNIST dataset class with one-hot encoded labels
    class OneHotMNIST(torchvision.datasets.MNIST):
        def __getitem__(self, index):
            img, target = super().__getitem__(index)
            one_hot_target = torch.zeros(10)
            one_hot_target[target] = 1
            return img, one_hot_target

    # Download and load the MNIST dataset
    train_set = OneHotMNIST(
        root='./data', train=True, download=True, transform=transform)
    test_set = OneHotMNIST(
        root='./data', train=False, download=True, transform=transform)

    # Create data loaders for training and testing
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=640, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=False, num_workers=0)

    # Test the data loader
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    print("Image tensor shape:", images.shape)
    print("Labels tensor shape:", labels.shape)

    return {"train": train_loader, "test": test_loader}


def load_data():
    # Define a custom Flatten transform
    class Flatten:
        def __call__(self, tensor):
            return tensor.view(tensor.shape[0], -1)

    # Define the data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize using mean and std from the MNIST dataset
        Flatten()
    ])

    # Define a custom MNIST dataset class with numerical labels
    class NumericMNIST(torchvision.datasets.MNIST):
        def __getitem__(self, index):
            img, target = super().__getitem__(index)
            return img, target + 1  # Add 1 to the target to make it between 1 and 10

    # Download and load the MNIST dataset
    train_set = NumericMNIST(
        root='./data', train=True, download=True, transform=transform)
    test_set = NumericMNIST(
        root='./data', train=False, download=True, transform=transform)

    # Create data loaders for training and testing
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=640, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=False, num_workers=0)

    # Test the data loader
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    print("Image tensor shape:", images.shape)
    print("Labels tensor shape:", labels.shape)

    return {"train": train_loader, "test": test_loader}
