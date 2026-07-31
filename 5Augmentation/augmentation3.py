#!/usr/bin/env python3

import os
import random
import wave

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops


# ----------------------------------------------------------
# GLOBAL VARIABLES
# ----------------------------------------------------------

DATASET_DIR = "./DATASET"
SAMPLE_RATE = 16000
MIN_FRECUENCY = 0
MAX_FRECUENCY = 8000
SAMPLES_FOR_FFT = 320
FRAME_STEP = 160 # next frame is 160 samples after the previous one
DURATION_SEC = 1.5
MEL_DOTS = 13

def get_mel_frecuency(vanilla_frecuency):
    return 2595 * np.log10(1 + (vanilla_frecuency) / 700)  # Convert Hz to Mel

def get_vanilla_frecuency(mel_frecuency):
    return 700 * (10 ** (mel_frecuency / 2595) - 1)  # Convert Mel to Hz

# 0
MIN_MEL_FRECUENCY = get_vanilla_frecuency(MIN_FRECUENCY)
# 2840.023047
MAX_MEL_FRECUENCY = get_mel_frecuency(MAX_FRECUENCY)
 #50
VANILLA_WIDTH_BETWEEN_BINS = SAMPLE_RATE / SAMPLES_FOR_FFT
 # 202.8588
MEL_WIDTH_BETWEEN_BINS = (MAX_MEL_FRECUENCY - MIN_MEL_FRECUENCY) / (MEL_DOTS + 1)

def get_mel_frecuency_from_filter_number(mel_filter_bank_number):
    return MIN_MEL_FRECUENCY + mel_filter_bank_number * MEL_WIDTH_BETWEEN_BINS

def get_bound_from_filter_number(mel_filter_bank_number):
    mel_frecuency =  get_mel_frecuency_from_filter_number(mel_filter_bank_number)
    vanilla_frecuency =  get_vanilla_frecuency(mel_frecuency)
    return int(vanilla_frecuency / VANILLA_WIDTH_BETWEEN_BINS)

def get_mel_filter_bank_bounds(mel_filter_bank_number):
    # TODO: verify mel filter bank indexing / MFCC convention against the ESP32 path
    # Triangular mel filter: left and right are the centers of the neighbors.
    left = get_bound_from_filter_number(mel_filter_bank_number)
    center = get_bound_from_filter_number(mel_filter_bank_number + 1 )
    right = get_bound_from_filter_number(mel_filter_bank_number + 2)
    return (left, center, right)


# ----------------------------------------------------------
# GET NORMALIZED PCM
# ----------------------------------------------------------

def get_normalized_pcm(wav_filename):

    try:
        # Load audio file
        audio_binary = tf.io.read_file(wav_filename)
        audio, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)

        # Remove channel dimension: (samples, 1) -> (samples,)
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)
        audio = audio.numpy()

        return audio

    except Exception as e:
        raise RuntimeError(f"   ✗ Error loading {file_name}: {e}")


# ----------------------------------------------------------
# get_spectrogram_original As Known As RUSSIAN NOTEBOOK FUNCTION
# This function is a adapted version from
# https://github.com/DenissStepanjuk/ESP32.INMP441.speech-recognition/blob/main/Python_INMP441/INMP441-CNN-TFL.ipynb
# This function is not the Most efficient python function to get the spectogram
# it repeats the same calculation that the ESP32 does in the hardware
# ----------------------------------------------------------

def get_spectrogram_original(audio):

    # same normalization as ESP32
    audio = audio - np.mean(audio)
    max_val = np.max(np.abs(audio))
    if max_val < 1e-6:
        max_val = 1.0

    audio = audio / max_val

    # FFT of exactly 320 points
    stft = tf.signal.stft(
        audio,
        frame_length=SAMPLES_FOR_FFT,
        frame_step=FRAME_STEP,
        fft_length=SAMPLES_FOR_FFT,
        window_fn=tf.signal.hann_window
    )

    # square of modulus
    spectrogram = tf.abs(stft) ** 2
    print("STFT shape:", spectrogram.shape)
    spectrogram = spectrogram.numpy()
    spectrogram = np.log10(spectrogram + 1e-6)
    

    #pooled = []

    #for i in range(41):

    #    start = i * 4
    #    end = min(start + 4, spectrogram.shape[1])
    #    pooled.append(
    #        np.mean(spectrogram[:, start:end], axis=1)
    #    )

    #spectrogram = np.stack(pooled, axis=1)

    # addition to epsilon to avoid log10(0)
    #spectrogram = np.log10(spectrogram + 1e-6)

    print("Pooled shape:", spectrogram.shape)

    return spectrogram

# ----------------------------------------------------------
# TF SPECTROGRAM FROM WAV
# ----------------------------------------------------------


def create_tf_spectrogram(wav_filename, png_filename, save_plot=False):

    audio = get_normalized_pcm(wav_filename)
    spectrogram = get_spectrogram_original(audio)
    plt.figure(figsize=(12, 4))

    # uncomment only for debugging
    # print(f"TF Shape: {spectrogram.shape}, Min: {np.min(spectrogram)}, Max: {np.max(spectrogram)}, Mean: {np.mean(spectrogram)} , Std: {np.std(spectrogram)}")

    if save_plot == True:
        plt.imshow(
            spectrogram.T,
            aspect="auto",
            origin="lower",
            cmap="viridis"
        )

        plt.title(
            "TensorFlow Spectrogram"
        )

        plt.xlabel("Time frames")
        plt.ylabel("Frequency bins")
        plt.colorbar(label="Log_10 Magnitude")
        plt.tight_layout()

        plt.savefig(
            png_filename,
            dpi=300
        )

        plt.close()

    return spectrogram

# --------------------------------------------------
# WAV I/O
# --------------------------------------------------


def crop_wav(audio, sample_rate, duration_sec=DURATION_SEC):
    samples = int(sample_rate * duration_sec)
    return audio[:samples]


def load_wav(filename):

    with wave.open(filename, "rb") as wav:

        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    if sample_width != 2:
        raise ValueError(f"Only supports 16-bit PCM WAV: {filename}")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

    if channels > 1:
        audio = audio.reshape(-1, channels)
        audio = np.mean(audio, axis=1)

    audio = audio / 32768.0
    audio = audio - np.mean(audio)

    return audio, sample_rate


def save_wav(filename, audio, sample_rate):

    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

    with wave.open(filename, "wb") as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


# --------------------------------------------------
# AUGMENTATIONS
# --------------------------------------------------

def time_shift_roll(audio, sample_rate, shift_number):

    # From -800 ms to +800 ms in 50 ms steps
    # shift_number=0 -> -800 ms, shift_number=32 -> +800 ms
    shift_ms = -800 + 50 * shift_number
    shift = int(sample_rate * shift_ms / 1000.0)

    result = np.empty_like(audio)

    if shift > 0:
        # Shift to the right: beginning filled with end of audio
        result[shift:] = audio[:-shift]
        result[:shift] = audio[-shift:]

    elif shift < 0:
        # Shift to the left: ending filled with beginning of audio
        shift = -shift
        result[:-shift] = audio[shift:]
        result[-shift:] = audio[:shift]

    else:
        result[:] = audio

    return result


def fit_length(audio, target_len):

    if len(audio) > target_len:
        return audio[:target_len]

    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)))

    return audio


def time_stretch(audio):

    factor = np.random.uniform(0.90, 1.10)
    x_old = np.arange(len(audio))
    x_new = np.linspace(0, len(audio) - 1, int(len(audio) / factor))
    result = np.interp(x_new, x_old, audio)

    return fit_length(result, len(audio))


def pitch_scale(audio):

    factor = np.random.uniform(0.95, 1.05)
    x_old = np.arange(len(audio))
    x_new = np.linspace(0, len(audio) - 1, int(len(audio) / factor))
    result = np.interp(x_new, x_old, audio)

    return fit_length(result, len(audio))


def add_noise_snr(audio, snr_db):

    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))

    return audio + noise


def white_noise(audio):

    snr_db = np.random.uniform(18, 35)

    return add_noise_snr(audio, snr_db)


def pink_noise(audio):

    white = np.random.randn(len(audio))
    fft = np.fft.rfft(white)
    freqs = np.arange(1, len(fft) + 1)
    fft = fft / np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=len(audio))
    pink /= np.max(np.abs(pink)) + 1e-8
    snr_db = np.random.uniform(18, 35)
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    pink *= np.sqrt(noise_power / (np.mean(pink ** 2) + 1e-8))

    return audio + pink


def brown_noise(audio):

    white = np.random.randn(len(audio))
    fft = np.fft.rfft(white)
    freqs = np.arange(1, len(fft) + 1)
    fft = fft / freqs
    brown = np.fft.irfft(fft, n=len(audio))
    brown /= np.max(np.abs(brown)) + 1e-8
    snr_db = np.random.uniform(18, 35)
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    brown *= np.sqrt(noise_power / (np.mean(brown ** 2) + 1e-8))

    return audio + brown


def background_noise(audio):

    noise_func = random.choice([
        white_noise,
        pink_noise,
        brown_noise
    ])

    return noise_func(audio)


def random_gain(audio):

    gain_db = np.random.uniform(-2.0, 0.0)
    factor = 10 ** (gain_db / 20)

    return audio * factor


def time_mask(audio):

    size = np.random.randint(0, len(audio) // 12)

    if size <= 0:
        return audio

    start = np.random.randint(0, len(audio) - size)
    result = audio.copy()
    result[start:start + size] = 0

    return result


# --------------------------------------------------
# PIPELINE
# --------------------------------------------------

def augment(audio, sample_rate, shift_number):

    result = audio.copy()

    #
    # VERY IMPORTANT:
    # teach the network that the word can appear
    # anywhere in the window.
    #
    if random.random() < 1.0:
        result = time_shift_roll(result, sample_rate, shift_number)

    if random.random() < 0.15:
        result = time_stretch(result)

    if random.random() < 0.15:
        result = pitch_scale(result)

    if random.random() < 0.60:
        result = background_noise(result)

    if random.random() < 0.30:
        result = random_gain(result)

    if random.random() < 0.10:
        result = time_mask(result)

    peak = np.max(np.abs(result))

    if peak > 0.55:
        result *= (0.55 / peak)

    return np.clip(result, -1.0, 1.0)


# --------------------------------------------------
# PROCESS FILE
# --------------------------------------------------

def process_wav_file(path):

    print(f"Processing {path}")

    audio, sample_rate = load_wav(path)
    audio = crop_wav(audio, sample_rate, duration_sec=DURATION_SEC)
    base, ext = os.path.splitext(path)

    for idx in range(0, 33):
        aug_audio = augment(audio, sample_rate, idx)
        output_wav_filename = f"{base}_aug{idx}.wav"
        save_wav(output_wav_filename, aug_audio, sample_rate)
        png_filename = f"{base}_aug{idx}_tf.png"
        spectogram = create_tf_spectrogram(output_wav_filename, png_filename, True)
        print(f"   -> {output_wav_filename}")


# --------------------------------------------------
# PROCESS FOLDER
# --------------------------------------------------

def process_folder(root_folder):

    for root, dirs, files in os.walk(root_folder):

        for file in files:

            if not file.lower().endswith(".wav"):
                continue

            if "_aug" in file:
                continue

            process_wav_file(os.path.join(root, file))


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    DATASET_DIR = "./DATASET"

    #process_folder(DATASET_DIR)
    for mel_filter_bank_number in range(0, 13):
        print(get_mel_filter_bank_bounds(mel_filter_bank_number))

    print("\nFinished.")
