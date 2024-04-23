import logging
import os
import random
import sys
import warnings
from functools import lru_cache

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pystoi
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

logger = logging.getLogger()


class AudioSimilarity:
    def __init__(
        self,
        original_path,
        compare_path,
        sample_rate,
        weights=None,
        verbose=True,
        sample_size=1,
    ):
        log_format = "%(message)s"
        logging.basicConfig(
            level=logging.INFO if verbose else logging.CRITICAL, format=log_format
        )

        if weights is None:
            self.weights = {
                "zcr_similarity": 0.2,
                "rhythm_similarity": 0.2,
                "spectral_flux_similarity": 0.2,
                "spectral_contrast_similarity": 0.2,
                "perceptual_similarity": 0.2,
            }
        else:
            self.weights = self.parse_weights(weights)

        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.original_path = original_path
        self.compare_path = compare_path

        self.is_directory = [
            os.path.isdir(path) for path in [original_path, compare_path]
        ]

        self.original_audios, self.compare_audios = self.load_audio_files()

        if not self.original_audios or not self.compare_audios:
            sys.exit("No valid audio files found in the provided paths.")

    def parse_weights(self, weights):
        if isinstance(weights, dict):
            return weights
        elif isinstance(weights, list):
            if len(weights) != 6:
                raise ValueError("Invalid number of weights. Expected 6 weights.")
            metric_names = [
                "zcr_similarity",
                "rhythm_similarity",
                "spectral_flux_similarity",
                "spectral_contrast_similarity",
                "perceptual_similarity",
            ]
            return dict(zip(metric_names, weights))
        else:
            raise ValueError("Invalid type for weights. Expected dict or list.")

    def load_audio_files(self):
        original_audios = []
        compare_audios = []
        valid_extensions = (".mp3", ".flac", ".wav")

        if self.is_directory[0]:
            try:
                original_files = [
                    os.path.join(self.original_path, f)
                    for f in os.listdir(self.original_path)
                    if f.endswith(valid_extensions)
                ]

            except FileNotFoundError as e:
                logging.error(f"Error loading original audio files: {e}")
                return [], []
        else:
            if not os.path.isfile(self.original_path):
                logging.error(f"Invalid original file path: {self.original_path}")
                return [], []
            original_files = [self.original_path]

        if self.is_directory[1]:
            try:
                compare_files = [
                    os.path.join(self.compare_path, f)
                    for f in os.listdir(self.compare_path)
                    if f.endswith(valid_extensions)
                ]
            except FileNotFoundError as e:
                logging.error(f"Error loading compare audio files: {e}")
                return [], []
        else:
            if not os.path.isfile(self.compare_path):
                logging.error(f"Invalid compare file path: {self.compare_path}")
                return [], []
            compare_files = [self.compare_path]

        if not original_files:
            logging.error("No original audio files found.")
        if not compare_files:
            logging.error("No compare audio files found.")

        if self.sample_size >= len(original_files):
            o_sample_size = len(original_files)
        else:
            o_sample_size = self.sample_size

        original_files = (
            random.sample(original_files, o_sample_size)
            if self.sample_size
            else original_files
        )

        if self.sample_size >= len(compare_files):
            c_sample_size = len(compare_files)
        else:
            c_sample_size = self.sample_size

        compare_files = (
            random.sample(compare_files, c_sample_size)
            if self.sample_size
            else compare_files
        )

        for original_file in tqdm(original_files, desc="Loading original files:"):
            try:
                original_audio, _ = librosa.load(original_file, sr=self.sample_rate)
                original_audios.append(original_audio)
            except FileNotFoundError as e:
                logging.error(f"Error loading file {original_file}: {e}")
                continue
            except Exception as e:
                logging.error(
                    f"Unexpected error loading file {original_file}: {type(e).__name__}, {e}"
                )
                continue

        for compare_file in tqdm(compare_files, desc="Loading comparison files:"):
            try:
                compare_audio, _ = librosa.load(compare_file, sr=self.sample_rate)
                compare_audios.append(compare_audio)
            except FileNotFoundError as e:
                logging.error(f"Error loading file {compare_file}: {e}")
                continue
            except Exception as e:
                logging.error(
                    f"Unexpected error loading file {compare_file}: {type(e).__name__}, {e}"
                )
                continue

        return original_audios, compare_audios

    @lru_cache(maxsize=None)
    def zcr_similarity(self):
        logging.info("Calculating zero crossing rate similarity...")

        total_zcr_similarity = 0
        count = 0

        for original_audio in self.original_audios:
            original_zcr = np.mean(np.abs(np.diff(np.sign(original_audio))) > 0)

            for compare_audio in self.compare_audios:
                compare_zcr = np.mean(np.abs(np.diff(np.sign(compare_audio))) > 0)
                zcr_similarity = 1 - np.abs(original_zcr - compare_zcr)
                total_zcr_similarity += zcr_similarity
                count += 1

        if count > 0:
            return total_zcr_similarity / count
        else:
            logging.info("No original audio files loaded.")
            return None

    @lru_cache(maxsize=None)
    def rhythm_similarity(self):
        logging.info("Calculating rhythm similarity...")
        total_rhythm_similarity = 0
        count = 0

        for original_audio in self.original_audios:
            min_length = len(original_audio)
            original_onset_vector = np.zeros(min_length)
            original_onsets = librosa.onset.onset_detect(
                y=original_audio, sr=self.sample_rate, units="time"
            )
            original_onsets = np.array(original_onsets) * self.sample_rate
            original_onsets = original_onsets[original_onsets < min_length]
            original_onset_vector[original_onsets.astype(int)] = 1

            for compare_audio in self.compare_audios:
                min_length = min(min_length, len(compare_audio))
                compare_onset_vector = np.zeros(min_length)
                compare_onsets = librosa.onset.onset_detect(
                    y=compare_audio[:min_length], sr=self.sample_rate, units="time"
                )
                compare_onsets = np.array(compare_onsets) * self.sample_rate
                compare_onsets = compare_onsets[compare_onsets < min_length]
                compare_onset_vector[compare_onsets.astype(int)] = 1

                rhythm_similarity = (
                    np.corrcoef(
                        original_onset_vector[:min_length],
                        compare_onset_vector[:min_length],
                    )[0, 1]
                    + 1
                ) / 2
                total_rhythm_similarity += rhythm_similarity
                count += 1

        if count > 0:
            return total_rhythm_similarity / count
        else:
            logging.info("No original audio files loaded.")
            return None

    @lru_cache(maxsize=None)
    def chroma_similarity(self):
        logging.info("Calculating chroma similarity similarity...")
        total_chroma_similarity = 0
        count = 0

        for original_audio in self.original_audios:
            for compare_audio in self.compare_audios:
                original_chroma = librosa.feature.chroma_cqt(
                    y=original_audio, sr=self.sample_rate
                )
                compare_chroma = librosa.feature.chroma_cqt(
                    y=compare_audio, sr=self.sample_rate
                )

                min_length = min(original_chroma.shape[1], compare_chroma.shape[1])
                original_chroma = original_chroma[:, :min_length]
                compare_chroma = compare_chroma[:, :min_length]

                chroma_similarity = 1 - np.mean(
                    np.abs(original_chroma - compare_chroma)
                )
                total_chroma_similarity += chroma_similarity
                count += 1

        if count > 0:
            return total_chroma_similarity / count
        else:
            logging.info("No original audio files loaded.")
            return None

    @lru_cache(maxsize=None)
    def spectral_contrast_similarity(self):
        logging.info("Calculating spectral contrast similarity...")
        if not self.original_audios or not self.compare_audios:
            logging.info("No audio files loaded.")
            return None

        total_spectral_contrast_similarity = 0
        count = 0

        for original_audio in self.original_audios:
            original_contrast = librosa.feature.spectral_contrast(
                y=original_audio, sr=self.sample_rate
            )
            min_columns = original_contrast.shape[1]

            for compare_audio in self.compare_audios:
                compare_contrast = librosa.feature.spectral_contrast(
                    y=compare_audio, sr=self.sample_rate
                )
                min_columns = min(min_columns, compare_contrast.shape[1])

            original_contrast = original_contrast[:, :min_columns]

            for compare_audio in self.compare_audios:
                compare_contrast = librosa.feature.spectral_contrast(
                    y=compare_audio, sr=self.sample_rate
                )
                compare_contrast = compare_contrast[:, :min_columns]
                contrast_similarity = np.mean(
                    np.abs(original_contrast - compare_contrast)
                )
                normalized_similarity = 1 - contrast_similarity / np.max(
                    [np.abs(original_contrast), np.abs(compare_contrast)]
                )
                total_spectral_contrast_similarity += normalized_similarity
                count += 1

        if count > 0:
            return total_spectral_contrast_similarity / count
        else:
            logging.info("No original audio files loaded.")
            return None

    @lru_cache(maxsize=None)
    def perceptual_similarity(self, sr=16000):
        logging.info("Calculating perceptual similarity...")
        if not self.original_audios or not self.compare_audios:
            logging.info("No audio files loaded.")
            return None

        total_perceptual_similarity = 0
        count = 0

        for i, original_audio in enumerate(self.original_audios):
            for j, compare_audio in enumerate(self.compare_audios):
                min_length = min(len(original_audio), len(compare_audio))
                array1_16k = librosa.resample(
                    y=original_audio[:min_length],
                    orig_sr=self.sample_rate,
                    target_sr=sr,
                )
                array2_16k = librosa.resample(
                    y=compare_audio[:min_length], orig_sr=self.sample_rate, target_sr=sr
                )
                score = pystoi.stoi(array1_16k, array2_16k, sr)
                score_normalized = (score + 1) / 2
                total_perceptual_similarity += score_normalized
                count += 1

        if count > 0:
            return total_perceptual_similarity / count
        else:
            logging.info("No original audio files loaded.")
            return None

    def stent_weighted_audio_similarity(self, metrics="swass"):
        if not self.original_audios or not self.compare_audios:
            logging.error("No audio files loaded.")
            return None

        num_original_audios = len(self.original_audios)
        num_compare_audios = len(self.compare_audios)

        zcr_similarities = np.zeros((num_original_audios, num_compare_audios))
        rhythm_similarities = np.zeros((num_original_audios, num_compare_audios))
        chroma_similarity = np.zeros((num_original_audios, num_compare_audios))
        spectral_contrast_similarities = np.zeros(
            (num_original_audios, num_compare_audios)
        )
        perceptual_similarities = np.zeros((num_original_audios, num_compare_audios))

        for i, original_audio in enumerate(self.original_audios):
            self.original_audio = original_audio
            for j, compare_audio in enumerate(self.compare_audios):
                self.compare_audio = compare_audio
                zcr_similarities[i, j] = self.zcr_similarity()
                rhythm_similarities[i, j] = self.rhythm_similarity()
                chroma_similarity[i, j] = self.chroma_similarity()
                spectral_contrast_similarities[i, j] = (
                    self.spectral_contrast_similarity()
                )
                perceptual_similarities[i, j] = self.perceptual_similarity()

        weights = np.array(list(self.weights.values()))
        similarities = (
            weights[0] * zcr_similarities
            + weights[1] * rhythm_similarities
            + weights[2] * chroma_similarity
            + weights[3] * spectral_contrast_similarities
            + weights[4] * perceptual_similarities
        )

        total_similarity = np.sum(similarities)
        count = num_original_audios * num_compare_audios

        if metrics == "all":
            return {
                "zcr_similarity": float(np.mean(zcr_similarities)),
                "rhythm_similarity": float(np.mean(rhythm_similarities)),
                "chroma_similarity": float(np.mean(chroma_similarity)),
                "spectral_contrast_similarity": float(
                    np.mean(spectral_contrast_similarities)
                ),
                "perceptual_similarity": float(np.mean(perceptual_similarities)),
                "swass": float(total_similarity / count),
            }
        elif metrics == "swass":
            if count > 0:
                return float(total_similarity / count)
            else:
                logging.error("No audio files loaded.")
                return None
        else:
            logging.error("Invalid value for 'metrics'. Choose 'swass' or 'all'.")
            return None
