import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from Models.AlexNet import AlexNetMNIST
from Layers.Helpers import MNISTData

np.random.seed(0)

data_layer = MNISTData(batch_size=2)

model = AlexNetMNIST()
model.data_layer = data_layer
model.phase(False)

loss = model.forward()
print(f"Loss: {loss}")

model.backward()

for i, layer in enumerate(model.layers):
    if hasattr(layer, "gradient_weights"):
        print(i, type(layer).__name__, "grad weights", layer.gradient_weights.shape, np.linalg.norm(layer.gradient_weights))
    if hasattr(layer, "gradient_bias"):
        print(i, type(layer).__name__, "grad bias", layer.gradient_bias.shape, np.linalg.norm(layer.gradient_bias))