from Layers.Conv import Conv
from Layers.ReLU import ReLU
from Layers.Pooling import Pooling
from Layers.Flatten import Flatten
from Layers.FullyConnected import FullyConnected
from Layers.Dropout import Dropout
from Layers.SoftMax import SoftMax
from Optimization.Optimizers import Adam
from NeuralNetwork import NeuralNetwork

class AlexNet(NeuralNetwork):
    def __init__(self):
        super().__init__(optimizer=Adam(learning_rate=0.001, mu=0.9, rho=0.999))

        self.append_layer(Conv(num_kernels=64, convolution_shape=(3, 11, 11), stride_shape=(4, 4)))
        self.append_layer(ReLU())
        self.append_layer(Pooling(stride_shape=(2, 2), pooling_shape=(3, 3)))

        self.append_layer(Conv(num_kernels=192, convolution_shape=(64, 5, 5), stride_shape=(1, 1)))
        self.append_layer(ReLU())
        self.append_layer(Pooling(stride_shape=(2, 2), pooling_shape=(3, 3)))

        self.append_layer(Conv(num_kernels=384, convolution_shape=(192, 3, 3), stride_shape=(1, 1)))
        self.append_layer(ReLU())

        self.append_layer(Conv(num_kernels=256, convolution_shape=(384, 3, 3), stride_shape=(1, 1)))
        self.append_layer(ReLU())

        self.append_layer(Conv(num_kernels=256, convolution_shape=(256, 3, 3), stride_shape=(1, 1)))
        self.append_layer(ReLU())
        self.append_layer(Pooling(stride_shape=(2, 2), pooling_shape=(3, 3)))

        self.append_layer(Flatten())

        self.append_layer(FullyConnected(256 * 6 * 6, 4096))
        self.append_layer(ReLU())
        self.append_layer(Dropout(0.5))

        self.append_layer(FullyConnected(4096, 4096))
        self.append_layer(ReLU())
        self.append_layer(Dropout(0.5))

        self.append_layer(FullyConnected(4096, 10))
        self.append_layer(SoftMax())


class AlexNetMNIST(NeuralNetwork):
    def __init__(self):
        super().__init__(optimizer=Adam(learning_rate=0.001, mu=0.9, rho=0.999))

        self.append_layer(Conv(num_kernels=32, convolution_shape=(1, 5, 5), stride_shape=(1, 1)))
        self.append_layer(ReLU())
        self.append_layer(Pooling(stride_shape=(2, 2), pooling_shape=(2, 2)))

        self.append_layer(Conv(num_kernels=64, convolution_shape=(32, 5, 5), stride_shape=(1, 1)))
        self.append_layer(ReLU())
        self.append_layer(Pooling(stride_shape=(2, 2), pooling_shape=(2, 2)))

        self.append_layer(Flatten())
        self.append_layer(FullyConnected(64 * 7 * 7, 256))
        self.append_layer(ReLU())
        self.append_layer(Dropout(0.5))
        self.append_layer(FullyConnected(256, 10))
        self.append_layer(SoftMax())
        
        
