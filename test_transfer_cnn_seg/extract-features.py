import os
import librosa
import numpy as np
from dcase_util.processors import MelExtractorProcessor
import pandas as pd
import pickle
import logging
import threading

logging_format = "%(asctime)s: %(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO,
                        datefmt="%H:%M:%S")

BASE_DIR = "/pub/dcase/datasets/datasets/TAU-Segmentado/TAU-urban-acoustic-scenes-2020-3class-development"
FEATURES_DIR = "features"


def pathname(name: str):
    return f"{BASE_DIR}/{{}}".format(name)

if not os.path.exists(FEATURES_DIR):
    os.mkdir(FEATURES_DIR)

meta = pd.read_csv(pathname("meta.csv"), sep="\t")

options = {
    "fs": 48000,
    "spectrogram_type": "magnitude",
    "hop_length_seconds": 0.02,
    "win_length_seconds": 0.04,
    "window_type": "hamming_asymmetric",
    "n_mels": 40,
    "n_fft": 2048,
    "fmin": 0,
    "fmax": 24000,
    "htk": False,
    "normalize_mel_bands": False
}

mel_processor = MelExtractorProcessor(**options)

def create_features(filename):
    audio,  sr = librosa.load(pathname(filename), sr=None)
    extract = mel_processor.extract(audio)
    return extract

size = meta['filename'].shape[0]

count = 0

def extract_features_thread(start, stop):
    global count
    for i_audio, audio in enumerate(meta['filename'][start:stop]):
        filename = "{}/{}".format(FEATURES_DIR, audio.replace("/","-").replace(".wav", ".cpickle"))
        with open(filename, "wb") as filewrite:
            pickle.dump(create_features(audio), filewrite)
        count += 1
        logging.info("{} / {} = {:.2f}%".format(count, size, (count/size)*100 ))

a = threading.Thread(target=extract_features_thread, args=(0, size//4), daemon=True)
b = threading.Thread(target=extract_features_thread, args=(size//4, 2*(size//4)), daemon=True)
c = threading.Thread(target=extract_features_thread, args=(2*(size//4), 3*(size//4)), daemon=True)
d = threading.Thread(target=extract_features_thread, args=(3*(size//4), size), daemon=True)
a.start()
b.start()
c.start()
d.start()
a.join()
b.join()
c.join()
d.join()
'''
if __name__ == "__main__":

    logging.info("Main    : before creating thread")
    x = threading.Thread(target=thread_function, args=(1,), daemon=True)
    logging.info("Main    : before running thread")
    x.start()
    logging.info("Main    : wait for the thread to finish")
    x.join()
    logging.info("Main    : all done")
'''
