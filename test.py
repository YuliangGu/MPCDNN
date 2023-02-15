from tensorflow import keras
from NNmodel import MLP
from quadrotor import Quad
from quadrotorOCP import QuadOpt
import casadi as cs
import numpy as np

def get_model():
    inputs = keras.Input(shape=17,)
    _ = keras.layers.Dense(17)(inputs)
    outputs = keras.layers.Dense(6)(_)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

keras_model = get_model()
mlp = MLP(keras_model)

quad = Quad()
dnn = {}
dnn['model'] = mlp
dnn['conf'] = {'reduced_state':True, 'full_state':False}

quadopt = QuadOpt(quad,residual=dnn)
x_r = np.ones(13)
u_r = np.ones(4)
quadopt.set_reference_state(X_ref=x_r, u_ref=u_r)

a = quadopt.discretize_f_and_q(m=4)

print(a)