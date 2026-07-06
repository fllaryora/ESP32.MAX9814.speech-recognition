#!/usr/bin/env python3

import os
import random
import wave

import numpy as np


# --------------------------------------------------
# WAV I/O
# --------------------------------------------------

def load_wav(filename):

    with wave.open(filename, "rb") as wav:

        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()

        raw = wav.readframes(frames)

    if sample_width != 2:
        raise ValueError(
            f"Solo soporta WAV PCM 16 bits: {filename}"
        )

    audio = np.frombuffer(
        raw,
        dtype=np.int16
    ).astype(np.float32)

    if channels > 1:
        audio = audio.reshape(-1, channels)
        audio = np.mean(audio, axis=1)

    audio = audio / 32768.0

    audio = audio - np.mean(audio)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio, sample_rate


def save_wav(filename, audio, sample_rate):

    audio = np.clip(audio, -1.0, 1.0)

    pcm = (audio * 32767).astype(np.int16)

    with wave.open(filename, "wb") as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        wav.writeframes(
            pcm.tobytes()
        )


# --------------------------------------------------
# AUGMENTATIONS
# --------------------------------------------------

def time_shift(audio, sr):

    max_shift = int(sr * 0.15)

    shift = np.random.randint(
        -max_shift,
        max_shift + 1
    )

    return np.roll(audio, shift)


def white_noise(audio):

    intensity = np.random.uniform(
        0.002,
        0.03
    )

    noise = np.random.normal(
        0,
        intensity,
        len(audio)
    )

    return audio + noise


def pink_noise(audio):

    white = np.random.randn(len(audio))

    fft = np.fft.rfft(white)

    freqs = np.arange(
        1,
        len(fft) + 1
    )

    fft = fft / np.sqrt(freqs)

    pink = np.fft.irfft(
        fft,
        n=len(audio)
    )

    pink /= (
        np.max(np.abs(pink)) + 1e-8
    )

    intensity = np.random.uniform(
        0.002,
        0.03
    )

    return audio + pink * intensity


def brown_noise(audio):

    white = np.random.randn(len(audio))

    fft = np.fft.rfft(white)

    freqs = np.arange(
        1,
        len(fft) + 1
    )

    fft = fft / freqs

    brown = np.fft.irfft(
        fft,
        n=len(audio)
    )

    brown /= (
        np.max(np.abs(brown)) + 1e-8
    )

    intensity = np.random.uniform(
        0.002,
        0.03
    )

    return audio + brown * intensity


def background_noise(audio):

    noise_func = random.choice([
        white_noise,
        pink_noise,
        brown_noise
    ])

    return noise_func(audio)


def random_gain(audio):

    gain_db = np.random.uniform(
        -5,
        5
    )

    factor = 10 ** (gain_db / 20)

    return audio * factor


def time_mask(audio):

    size = np.random.randint(
        0,
        len(audio) // 12
    )

    if size <= 0:
        return audio

    start = np.random.randint(
        0,
        len(audio) - size
    )

    result = audio.copy()

    result[start:start + size] = 0

    return result


# --------------------------------------------------
# PIPELINE
# --------------------------------------------------

def augment(audio, sr):

    result = audio.copy()

    if random.random() < 0.9:
        result = time_shift(result, sr)

    if random.random() < 0.8:
        result = background_noise(result)

    if random.random() < 0.7:
        result = random_gain(result)

    if random.random() < 0.3:
        result = time_mask(result)

    peak = np.max(np.abs(result))

    if peak > 0:
        result = result / peak

    return np.clip(
        result,
        -1.0,
        1.0
    )


# --------------------------------------------------
# PROCESS FILE
# --------------------------------------------------

def process_file(path):

    print(f"Procesando {path}")

    audio, sr = load_wav(path)

    base, ext = os.path.splitext(path)

    for idx in range(1, 4):

        aug_audio = augment(
            audio,
            sr
        )

        output = (
            f"{base}_aug{idx}.wav"
        )

        save_wav(
            output,
            aug_audio,
            sr
        )

        print(
            f"   -> {output}"
        )


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

            process_file(
                os.path.join(root, file)
            )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    DATASET_DIR = "./DATASET"

    process_folder(DATASET_DIR)

    print("\nFinalizado.")
