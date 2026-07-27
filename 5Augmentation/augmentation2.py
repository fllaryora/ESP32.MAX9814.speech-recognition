#!/usr/bin/env python3

import os
import random
import wave

import numpy as np


# --------------------------------------------------
# WAV I/O
# --------------------------------------------------

def crop_wav(audio, sample_rate, duration_sec=1.5):
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
        raise ValueError(f"Solo soporta WAV PCM 16 bits: {filename}")

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

def time_shift(audio, sample_rate):

    # Hasta ±250 ms
    max_shift = int(sample_rate * 0.25)

    shift = np.random.randint(-max_shift, max_shift + 1)

    result = np.zeros_like(audio)

    if shift > 0:
        # Desplaza hacia la derecha
        result[shift:] = audio[:-shift]

    elif shift < 0:
        # Desplaza hacia la izquierda
        shift = -shift
        result[:-shift] = audio[shift:]

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

def augment(audio, sample_rate):

    result = audio.copy()

    #
    # MUY IMPORTANTE:
    # enseñar a la red que la palabra puede aparecer
    # en cualquier lugar de la ventana.
    #
    if random.random() < 1.0:
        result = time_shift(result, sample_rate)

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

def process_file(path):

    print(f"Procesando {path}")

    audio, sample_rate = load_wav(path)
    audio = crop_wav(audio, sample_rate, duration_sec=1.5)

    base, ext = os.path.splitext(path)

    for idx in range(1, 4):

        aug_audio = augment(audio, sample_rate)

        output = f"{base}_aug{idx}.wav"

        save_wav(output, aug_audio, sample_rate)

        print(f"   -> {output}")

    if "BASURA" in base.upper():
        save_wav(path, aug_audio, sample_rate)


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

            process_file(os.path.join(root, file))


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    DATASET_DIR = "./DATASET"

    process_folder(DATASET_DIR)

    print("\nFinalizado.")