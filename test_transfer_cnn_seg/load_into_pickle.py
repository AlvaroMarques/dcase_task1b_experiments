import os
import numpy as np
import pickle
import pandas as pd
import threading

FEATURES_DIR = "features"
BASE_DIR = "/pub/dcase/datasets/datasets/TAU-Segmentado/TAU-urban-acoustic-scenes-2020-3class-development"

def convert_filename(filename: str)->str:
    return filename.replace(".wav", ".cpickle").replace("/", "-")

def pathname(name: str):
    return f"{BASE_DIR}/{{}}".format(name)

N_THREADS = 4

counter = 0

def get_X_y(data):
    global counter
    X = np.zeros((1, 40,26))
    y = np.array([])
    for i_audio, (audio, label) in enumerate(zip(data['filename'], data['scene_label'])):
        with open('{}/{}'.format(FEATURES_DIR, convert_filename(audio)), "rb") as fileread:
            X = np.vstack([X, pickle.load(fileread).reshape(1,40,26)])
            y = np.hstack([y, label])
        counter += 1
        print("{} / {} = {:.2f}%".format(counter, size, 100*counter/size))
    return X[1:], y

def thread_X_y(csv, num):
    global size
    data = pd.read_csv(pathname(csv), sep="\t")
    size = data.shape[0]
    start = num*size//N_THREADS
    stop = (num+1)*size//N_THREADS
    data = data.iloc[start:stop,:]
    X, y = get_X_y(data)
    pickle.dump(X, open(f"X_train_{num}.cpickle", "wb"))
    pickle.dump(y, open(f"y_train_{num}.cpickle", "wb"))


CSV = "evaluation_setup/fold1_train.csv"
threads = [threading.Thread(target=thread_X_y, args=(CSV, i_thread), daemon=True) for i_thread in range(N_THREADS)]

for i_thread, _ in enumerate(threads):
    threads[i_thread].start()
for i_thread, _ in enumerate(threads):
    threads[i_thread].join()
