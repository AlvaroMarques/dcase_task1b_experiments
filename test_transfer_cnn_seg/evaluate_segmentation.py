import os
# Desabilitando warnings do tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

# Mostrando estado do tensorflow
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

FEATURES_DIR = "features"
BASE_DIR = "/pub/dcase/datasets/datasets/TAU-Segmentado/TAU-urban-acoustic-scenes-2020-3class-development"

CSV_TEST = "/evaluation_setup/fold1_evaluate.csv"
fold_stats_filename = "norm_fold_1.cpickle"

MODEL_INPUT = "model_train_5.h5"

def pathname(name):
    return BASE_DIR + "/" + name

def change(name):
    return os.path.join(FEATURES_DIR, name.replace("/", "-").replace("wav","cpickle"))

with open("one_hot.cpickle", "rb") as one_hot_file:
    encoder = load(one_hot_file)

evaluate = pd.read_csv(pathname(CSV_TEST), sep="\t")

labels_test = evaluate['scene_label'].values.reshape(-1, 1)
labels_test = encoder.transform(labels_test).toarray()

normalizer = dcase_util.data.Normalizer().load(filename=fold_stats_filename)

def generator_test():
    for x, y in zip(evaluate['filename'].values, labels_test):
        with open(change(x), 'rb') as file_open:
            array = load(file_open)
            yield normalizer.normalize(array).reshape((40, 26, 1)), y

dataset_test=tf.data.Dataset.from_generator(generator_test,
    output_types=(
        tf.float32,
        tf.float32
    ),
    output_shapes=(
        tf.TensorShape([40,26,1]),
        tf.TensorShape([3])
    )
).batch(1024)

SIZE = evaluate.values.shape[0]

model = load_model(MODEL_INPUT)
model.summary()

predictions = np.array([])

for i, (x, y) in enumerate(dataset_test):
    print("{} / {} = {:.2f}% Completo".format(i*1024, SIZE, 100 * (i*1024) / SIZE))
    y_ = model(x, training=False)
    predictions = np.append(predictions, encoder.inverse_transform(y_.numpy()).reshape(-1))

evaluate['predcitions'] = predictions
evaluate.to_csv("evaluate_{}.csv".format(MODEL_INPUT.replace(".h5", "")), index=False)
