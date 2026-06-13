import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.AlexNet import AlexNetMNIST
from Layers.Helpers import MNISTData


def main():
    batch_size = 4
    iterations = 10

    data_layer = MNISTData(batch_size=batch_size)

    model = AlexNetMNIST()
    model.data_layer = data_layer
    model.train(iterations)

    print(f"Final loss: {model.loss[-1]}")

    test_images, test_labels = data_layer.get_test_set()
    predictions = model.test(test_images[:batch_size])

    print(f"Predictions: {predictions}")

if __name__ == "__main__":
    main()
