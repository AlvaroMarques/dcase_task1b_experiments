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

S = 4
threads = [threading.Thread(target=extract_features_thread, args=(i*size//(S), (i+1)*size//(S)), daemon=True) for i in range(S)]

for i_thread, _ in enumerate(threads):
    threads[i_thread].start()

for i_thread, _ in enumerate(threads):
    threads[i_thread].join()
