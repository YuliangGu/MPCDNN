from tensorflow import keras
from NNmodel import MLP
from quadrotor import Quad
from quadrotorOCP import QuadOpt
import casadi as cs
import numpy as np


model = keras.models.load_model('model_mlp')
# x,y = generate_data(npzfile='./data_2learn/test_5.npz')
# y_pred = model(x)

x = cs.MX.sym('x', 17)
mlp = MLP('model_mlp')
a = mlp.forward(x)
test = np.random.random(17)
f = cs.Function('f',[x],[a],['x'],['r'])
test_cs = f(x=test)
print(test_cs)

test = np.reshape(test,(1,17))
test_pred = model.predict(test)
print(test_pred)
# x_o = mlp.forward(x_t)

# f = cs.Function([x_t],[x_o])
# quad = Quad()
# dnn = {}
# dnn['model'] = mlp
# dnn['conf'] = {'reduced =_state':False, 'full_state':True}
# quadopt = QuadOpt(quad,residual=dnn)


# x_r = np.ones(13)
# u_r = np.ones(4)
# quadopt.set_reference_state(X_ref=x_r, u_ref=u_r)

# a = quadopt.discretize_f_and_q(m=4)

# print(a)