import sys
from pathlib import Path

from matplotlib.pylab import rint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.AlexNet import AlexNetMNIST
from Layers.Helpers import MNISTData, calculate_accuracy


def main():
    batch_size = 32
    epochs = 1

    data_layer = MNISTData(batch_size=batch_size)

    val_size = 10000

    val_images = data_layer.train[-val_size:]
    val_labels = data_layer.labels[-val_size:]

    data_layer.train = data_layer.train[:-val_size]
    data_layer.labels = data_layer.labels[:-val_size]

    data_layer._current_forward_idx_iterator = data_layer._forward_idx_iterator()

    iterations_per_epoch = data_layer.train.shape[0] // batch_size
    total_iterations = iterations_per_epoch * epochs

    print(f"Iterations per epoch: {iterations_per_epoch}")
    print(f"Total iterations: {total_iterations}")

    print(f"Training samples: {data_layer.train.shape[0]}")
    print(f"Validation samples: {val_images.shape[0]}")
    print(f"Test samples: {data_layer.test.shape[0]}")

    print(f"Training images shape: {data_layer.train.shape}")
    print(f"Training labels shape: {data_layer.labels.shape}")
    print(f"Validation images shape: {val_images.shape}")
    print(f"Validation labels shape: {val_labels.shape}")
    print(f"Test images shape: {data_layer.test.shape}")
    print(f"Test labels shape: {data_layer.testLabels.shape}")

    model = AlexNetMNIST()
    model.data_layer = data_layer
    for iteration in range(100):
        loss = model.forward()
        train_predictions = model.loss_layer.prediction_tensor
        train_labels = model.label_tensor
        train_accuracy = calculate_accuracy(train_predictions, train_labels)
        model.loss.append(loss)
        model.backward()
        print(f"Iteration {iteration + 1}, train loss: {loss / batch_size}, train accuracy: {train_accuracy}")

    print(f"Final loss: {model.loss[-1]}")

    test_images, test_labels = data_layer.get_test_set()
    predictions = model.test(test_images[:batch_size])

    print(f"Predictions: {predictions}")

if __name__ == "__main__":
    main()
