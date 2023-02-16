# import tensorflow as tf
import numpy as np
from quadrotor import Quad
from tensorflow import keras
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

def generate_data(npzfile):
    npzfile = np.load(npzfile)
    X = npzfile['arr_0']
    u = npzfile['arr_1']
    quad = Quad()
    x_nominal = np.zeros(X.shape)
    for i in range(u.shape[0]-1):
        quad.set_state(X_toSet=X[i,:])
        for _ in range(10):
            quad.update(u[i,:],dt=0.01)
        x_nominal[i+1,:] = quad.get_state(stack=True)
    err = (X - x_nominal)[1:]
    # print(err.shape)
    return (np.concatenate((X[1:],u[1:]),axis=1),err[:,10:])
 
def build_mlp():
    inputs = keras.Input(shape=(17,))
    # dense1 = tfa.layers.SpectralNormalization(keras.layers.Dense(32, activation="relu", name="dense_1",kernel_regularizer=keras.regularizers.L2(0e-6)))
    # x = dense1(inputs)
    # dense2 = tfa.layers.SpectralNormalization(keras.layers.Dense(32, activation="relu", name="dense_2",kernel_regularizer=keras.regularizers.L2(0e-6)))
    # x = dense2(x)
    # dense3 = tfa.layers.SpectralNormalization(keras.layers.Dense(13, name="outputs"))
    # outputs = dense3(x)
    x = keras.layers.Dense(32, activation="relu",kernel_regularizer=keras.regularizers.L2(1e-3))(inputs)
    x = keras.layers.Dense(32, activation="relu",kernel_regularizer=keras.regularizers.L2(1e-3))(x)
    outputs = keras.layers.Dense(3)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="MLP")
    model.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.MeanSquaredError())
    return model

def training(model, data,data_val):
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=1e-5,
                patience=3,verbose=1,)]
    x, y = data
    history = model.fit(x,y,batch_size=10,
    epochs=50,
    validation_data=data_val,
    # callbacks=callbacks,
    )

model = build_mlp()
data_val = generate_data(npzfile='./data_2learn/test_5.npz')
for i in range(1,5):
    file = './data_2learn/test_'+str(i)+'.npz'
    data = generate_data(file)
    training(model,data,data_val)
for i in range(2,6):
    file = './data/test_'+str(i)+'.npz'
    data = generate_data(file)
    training(model,data,data_val)

model.save("model_mlp")
x,y = data_val
y_pred = model.predict(x)
plt.plot(y[:])
plt.plot(y_pred[:],label='pred')

# x,y = data_val
# fig, axs = plt.subplots(2, 2)
# axs[0,0].plot(y[:,:3])
# axs[0,1].plot(y[:,3:6])
# axs[1,0].plot(y[:,6:10])
# axs[1,1].plot(y[:,10:])
# plt.plot(y[:,10:])
# plt.plot(y_pred[:,10:],label='pred')
plt.legend()
plt.show()
# file = './data_2learn/test_1.npz'
# data = generate_data(file)
# training(model,data)

