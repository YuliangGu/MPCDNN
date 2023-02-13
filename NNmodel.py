import casadi as cs
import tensorflow as tf
from utils import activation
from tensorflow import keras

class MLP():
    def __init__(self, my_model):
        self.model = keras.models.load_model(my_model)
        # self.model = model
        self.num_hidden = None

        self.weights = []
        self.bias = []
        self.act = []

        self.get_weights_bias()
        self.get_act()

    def get_weights_bias(self):
        weights_ = self.model.get_weights()
        self.num_hidden = int(len(weights_)/2)
        for i in range(0, len(weights_), 2):
            # take the transpose here
            self.weights.append(weights_[i].T)
            self.bias.append(weights_[i+1])
    
    def get_act(self):  
        for i in range(self.num_hidden):
            activation = self.model.get_layer(index=i+1).get_config()['activation']
            self.act.append(activation)

    def forward(self, x):
        for i in range(self.num_hidden):
            x = cs.mtimes(self.weights[i],x) + self.bias[i]
            x = activation(x, self.act[i])
        return x

