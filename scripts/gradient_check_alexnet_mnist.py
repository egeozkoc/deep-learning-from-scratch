import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from Models.AlexNet import AlexNetMNIST
from Layers.Helpers import gradient_check_weights
from Layers.Helpers import MNISTData


data_layer = MNISTData(batch_size=2)

model = AlexNetMNIST()

x, y = data_layer.next()

activation = x
final_fc_index = 10

for layer in model.layers[:final_fc_index]:
    activation = layer.forward(activation)

final_fc = model.layers[final_fc_index]
softmax = model.layers[final_fc_index + 1]

final_fc.optimizer = None

diff = gradient_check_weights(
    [final_fc, softmax, model.loss_layer],
    activation,
    y,
    bias=False,
)

print("max diff:", np.max(diff))
print("mean diff:", np.mean(diff))

