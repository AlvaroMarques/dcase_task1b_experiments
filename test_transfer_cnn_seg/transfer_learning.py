"""
Recebe o modelo original da baseline do dcase:

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 40, 500, 32)       1600      
_________________________________________________________________
batch_normalization_1 (Batch (None, 40, 500, 32)       128       
_________________________________________________________________
activation_1 (Activation)    (None, 40, 500, 32)       0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 100, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 8, 100, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 100, 64)        100416    
_________________________________________________________________
batch_normalization_2 (Batch (None, 8, 100, 64)        256       
_________________________________________________________________
activation_2 (Activation)    (None, 8, 100, 64)        0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 2, 1, 64)          0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 2, 1, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               12900     
_________________________________________________________________
dropout_3 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 303       
=================================================================
Total params: 115,603
Trainable params: 115,411
Non-trainable params: 192
_________________________________________________________________

Transferimos os pesos das camadas convolucionais e os das de de
batch normalization para um novo modelo que aceita entradas de 
(40, 26) em vez de (40, 500).

"""

import numpy as np 
# Todas as classes e funcoes do modulo layers sao importadas, 
# mas provavelmente seria melhor importar apenas as layers que foram usadas
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, load_model, save_model

# Os pesos do modelo pre-treinado pelo baseline sao carregados
model = load_model("model_fold_1.h5")
# Arquitetura do modelo que vai receber os pesos
model_ = Sequential([
  Conv2D(32, 7, input_shape=(40,26,1),padding="same", kernel_initializer="glorot_uniform", data_format="channels_last"),
  BatchNormalization(axis=-1),
  Activation("relu"),
  MaxPool2D(pool_size=(5,5), data_format="channels_last"),
  Dropout(0.3),
  Conv2D(64, 7, padding="same", kernel_initializer="glorot_uniform", data_format="channels_last"),
  BatchNormalization(axis=-1),
  Activation("relu"),
  MaxPool2D(pool_size=(5,5), data_format="channels_last"),
  Dropout(0.3),
  Flatten(),
  Dense(units=50, activation="relu", kernel_initializer="uniform"),
  Dropout(0.3),
  Dense(units=3,kernel_initializer="uniform", activation="softmax")
])

# Seto os pesos das camadas convolucionais iguais ao do modelo do baseline
# E desligo o treinamento dessas camadas
for i in range(10):
  model_.layers[i].set_weights(model.layers[i].get_weights())
  model_.layers[i].trainable = False

print(model_.summary())
# Modelo final eh salvo
save_model(model_, "model_transfer.h5")
