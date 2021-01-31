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

### Criacao do dataset

# Constantes relativas aos diretorios e arquivos utilizados
FEATURES_DIR = "features"
BASE_DIR = "/pub/dcase/datasets/datasets/TAU-Segmentado/TAU-urban-acoustic-scenes-2020-3class-development"
CSV = "/evaluation_setup/fold1_train.csv"
CSV_TEST = "/evaluation_setup/fold1_evaluate.csv"
fold_stats_filename = "norm_fold_1.cpickle"

MODEL_INPUT = "model_train_5.h5"
MODEL_OUTPUT = "final_output.h5"
STATE_MODEL_PATTERN = "model_train_"

def pathname(name):
		return BASE_DIR + "/" + name

def change(name):
		return os.path.join(FEATURES_DIR, name.replace("/", "-").replace("wav","cpickle"))

train = pd.read_csv(pathname(CSV), sep='\t')

SIZE = train.values.shape[0]

# Importando o normalizador usado no baseline
normalizer = dcase_util.data.Normalizer().load(filename=fold_stats_filename)

# Transforando as labels no format de one hot
encoder = OneHotEncoder()

encoder.fit(train['scene_label'].values.reshape(-1, 1))

labels_ = encoder.transform(train['scene_label'].values.reshape(-1, 1)).toarray()

test = pd.read_csv(pathname(CSV_TEST), sep='\t')

labels_test = encoder.transform(test['scene_label'].values.reshape(-1, 1)).toarray()

with open("one_hot.cpickle", "wb") as file_ohe:
		dump(encoder, file_ohe)

def generator_train():
		for x, y in zip(train['filename'].values, labels_):
				with open(change(x), 'rb') as file_open:
						array = load(file_open)
						yield normalizer.normalize(array).reshape((40,26,1)), y

def generator_test():
		for x, y in zip(test['filename'].values, labels_test):
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

model = load_model(MODEL_INPUT)
print(model.summary())

loss_object = tf.keras.losses.CategoricalCrossentropy(
		from_logits=False, label_smoothing=0,
		name='categorical_crossentropy'
)

def loss(model, x, y, training):
	# Se calcula o loss por categorical crossentropy
	y_ = model(x, training=training)
	return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
	with tf.GradientTape() as tape:
		loss_value = loss(model, inputs, targets, training=True)
	return loss_value, tape.gradient(loss_value, model.trainable_variables)

# Otimizador utilizado
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
num_epochs = 30

train_dataset = dataset

metrics = []

# Looop de treinamento principal
for epoch in range(num_epochs):
	if epoch % 5 == 0:
	 save_model(model, "{}_{}.h5".format(STATE_MODEL_PATTERN, epoch))
	epoch_loss_avg = tf.keras.metrics.Mean()
	epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
	epoch_val_loss_avg = tf.keras.metrics.Mean()
	epoch_val_accuracy = tf.keras.metrics.CategoricalAccuracy()

	# Treinamento
	for i, (x, y) in enumerate(dataset):
		# Optimize the model
		loss_value, grads = grad(model, x, y)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))

		# Adicionando os valores de loss de cada iteracao
		epoch_loss_avg.update_state(loss_value) 

		# Adicionando mais valores para o calculo da acuracia da epoca
		epoch_accuracy.update_state(y, model(x, training=True))
	# Validacao
	for i, (x, y) in enumerate(dataset_test):
		# Calcula-se o loss com treinamento desabilitado
		loss_value = loss(model, x, y, training=False)
		epoch_val_loss_avg.update_state(loss_value)
		epoch_val_accuracy.update_state(y, model(x, training=False))
				
	print("Epoch {:03d}:".format(epoch))
	print("Loss: {:.3f}, Accuracy: {:.3%}".format(epoch_loss_avg.result(), epoch_accuracy.result()))
	print("Val Loss: {:.3f}, Val Accuracy: {:.3%}".format(epoch_val_loss_avg.result(), epoch_val_accuracy.result()))

	metrics.append([epoch_loss_avg.result(), epoch_accuracy.result(), 
			epoch_val_loss_avg.result(), epoch_val_accuracy.result()])

save_model(model, MODEL_OUTPUT)
with open("metrics.csv","a") as metrics_file:
		metrics_file.write("\n".join([",".join(map(lambda x: str(float(x)), l)) for l in metrics]))
