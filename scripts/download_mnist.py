from pathlib import Path

from torchvision import datasets, transforms

def main():

    data_dir = Path("data")

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

if __name__ == "__main__":
    main()