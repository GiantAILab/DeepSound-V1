# coding=utf-8
# Silence detection 
import logging
import librosa
import numpy as np


class Step4:
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.INFO)
    

    def run(self, 
            audio_path,
            silence_thresh=-50,
            duration_thresh=0.9):
        self.log.info("Step4: Determine whether the audio is silent.")
        y, sr = librosa.load(audio_path, sr=None)
        energy = librosa.feature.rms(y=y)[0]
        energy_db = librosa.amplitude_to_db(energy)
        silent_ratio = np.sum(energy_db < silence_thresh) / len(energy_db)
        is_silent = silent_ratio > duration_thresh
        self.log.info("Finish Step4 successfuilly")
        return is_silent
