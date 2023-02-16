import casadi as cs
import tensorflow as tf
from utils import activation
from tensorflow import keras

class MLP():
    def __init__(self, my_model,num_layers=4):
        self.model = keras.models.load_model(my_model)
        self.num_hidden = None

        self.weights = []
        self.bias = []
        self.act = []

        self.get_config(num_layers=num_layers)

    def get_config(self, num_layers):
        for i in range(1,num_layers):
            layer = self.model.get_layer(index=i)
            w,b = layer.get_weights()
            self.weights.append(w.T)
            self.bias.append(b)
            self.act.append(layer.get_config()['activation'])

    def forward(self, x):
        for i in range(len(self.weights)):
            x = cs.mtimes(self.weights[i],x) + self.bias[i]
            x = activation(x, self.act[i])
        return x

