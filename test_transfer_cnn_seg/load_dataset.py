import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import glob
import tensorflow as tf
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from pickle import load, dump
import pandas as pd
import numpy as np
import os
import dcase_util
from tensorflow.keras.models import load_model, save_model
from sklearn.preprocessing import OneHotEncoder

path = "features"

FEATURES_DIR = "features"
BASE_DIR = "/pub/dcase/datasets/datasets/TAU-Segmentado/TAU-urban-acoustic-scenes-2020-3class-development"
CSV = "/evaluation_setup/fold1_train.csv"
CSV_TEST = "/evaluation_setup/fold1_evaluate.csv"

def pathname(name):
    return BASE_DIR + "/" + name

def change(name):
    return os.path.join(path, name.replace("/", "-").replace("wav","cpickle"))

#def load_pickle()

meta = pd.read_csv(pathname(CSV), sep='\t')
SIZE = meta.values.shape[0]

filenames = np.asarray([
    change(name) for name in meta['filename'].values
])

fold_stats_filename = "norm_fold_1.cpickle"
normalizer = dcase_util.data.Normalizer().load(filename=fold_stats_filename)

encoder = OneHotEncoder()

encoder.fit(meta['scene_label'].values.reshape(-1, 1))

labels_ = encoder.transform(meta['scene_label'].values.reshape(-1, 1)).toarray()

test = pd.read_csv(pathname(CSV_TEST), sep='\t')

labels_test = encoder.transform(test['scene_label'].values.reshape(-1, 1)).toarray()

with open("one_hot.cpickle", "wb") as file_ohe:
    dump(encoder, file_ohe)

p

def generator_train():
    for x, y in zip(meta['filename'].values, labels_):
        with open(change(x), 'rb') as file_open:
            array = load(file_open)
            yield normalizer.normalize(array).reshape((40,26,1)), y

def generator_test():
    for x, y in zip(test['filename'].values, labels_test):
        with open(change(x), 'rb') as file_open:
            array = load(file_open)
            yield normalizer.normalize(array).reshape((40, 26, 1)), y

dataset_test = tf.data.Dataset.from_generator(generator_test,
                                            output_types=(
                                                tf.float32,
                                                tf.float32
                                                ),
                                           output_shapes=(
                                                tf.TensorShape([40,26,1]),
                                                tf.TensorShape([3]) 
                                               )
                                           ).batch(1024)

dataset = tf.data.Dataset.from_generator(generator_train,
                                            output_types=(
                                                tf.float32,
                                                tf.float32
                                                ),
                                           output_shapes=(
                                                tf.TensorShape([40,26,1]),
                                                tf.TensorShape([3]) 
                                               )
                                           ).batch(1024)

features, labels = next(iter(dataset))
    

#iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
#el = iterator.get_next()

model = load_model("model_transfer.h5")
print(model.summary())

loss_object = tf.keras.losses.CategoricalCrossentropy(
    from_logits=False, label_smoothing=0,
    name='categorical_crossentropy'
)

#init = tf.compat.v1.initialize_all_variables() 

def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)

  return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

num_epochs = 30

train_dataset = dataset

for _ in enumerate(dataset_test):
    break

for epoch in range(num_epochs):
  print(epoch)
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
  epoch_val_loss_avg = tf.keras.metrics.Mean()
  epoch_val_accuracy = tf.keras.metrics.CategoricalAccuracy()

  # Training loop - using batches of 64
  for i, (x, y) in enumerate(dataset):
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    # training=True is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    epoch_accuracy.update_state(y, model(x, training=True))

  for i, (x, y) in enumerate(dataset_test):
    loss_value = loss(model, x, y, training=False)
    epoch_val_loss_avg.update_state(loss_value)
    epoch_val_accuracy.update_state(y, model(x, training=False))
        

  print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
  print("Epoch {:03d}: Val Loss: {:.3f}, Val Accuracy: {:.3%}".format(epoch,
                                                                epoch_val_loss_avg.result(),
                                                                epoch_val_accuracy.result()))

save_model(model, "result2.h5")
