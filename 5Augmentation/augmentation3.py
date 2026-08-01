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

#--------
# It returns the mel frecuency from the filter number
#----
def get_mel_frecuency_from_filter_number(mel_filter_bank_number):
    return MIN_MEL_FRECUENCY + mel_filter_bank_number * MEL_WIDTH_BETWEEN_BINS

#------
# It return the bin index from the filter number
#---
def get_bound_from_filter_number(mel_filter_bank_number):
    mel_frecuency =  get_mel_frecuency_from_filter_number(mel_filter_bank_number)
    vanilla_frecuency =  get_vanilla_frecuency(mel_frecuency)
    return int(vanilla_frecuency / VANILLA_WIDTH_BETWEEN_BINS)

#------
# It returns the left, center and right bounds of the filter bank index
#---
def get_mel_filter_bank_bounds(mel_filter_bank_number):
    # TODO: verify mel filter bank indexing / MFCC convention against the ESP32 path
    # Triangular mel filter: left and right are the centers of the neighbors.
    left = get_bound_from_filter_number(mel_filter_bank_number)
    center = get_bound_from_filter_number(mel_filter_bank_number + 1 )
    right = get_bound_from_filter_number(mel_filter_bank_number + 2)
    return (left, center, right)


#-----
# It returns the triangular filter weight for the bin number in the mel spectrogram
#----
def triangular_filter_H(bin_number_in_mel_spectrogram, left, center, right):
    # H(k): triangular weight for FFT bin k given filter bank bounds (bin indices).
    if bin_number_in_mel_spectrogram < left or bin_number_in_mel_spectrogram > right:
        return 0.0
    # increasing    
    if bin_number_in_mel_spectrogram <= center:
        return (bin_number_in_mel_spectrogram - left) / (center - left)
    # decreasing
    return (right - bin_number_in_mel_spectrogram) / (right - center)


#------
# It builds the mel filter bank matrix
#---
def build_mel_filter_bank_matrix(n_freq_bins):
    # (n_freq_bins, MEL_DOTS) — used as spectrogram @ filter_bank (BLAS / C under NumPy)
    filter_bank = np.zeros((n_freq_bins, MEL_DOTS), dtype=np.float32)
    for mel_filter_bank_number in range(MEL_DOTS):
        left, center, right = get_mel_filter_bank_bounds(mel_filter_bank_number)
        for bin_number in range(left, right + 1):
            if 0 <= bin_number < n_freq_bins:
                filter_bank[bin_number, mel_filter_bank_number] = triangular_filter_H(
                    bin_number, left, center, right
                )
    return filter_bank


def discrete_cosine_transformation_type_2(mel_spectrogram):
    # DCT-II along the mel axis (axis=1), one transform per time frame.
    # Input:  (n_frames, N)  e.g. mel_spectrogram from spectrogram @ mel_filter_bank
    # Output: (n_frames, N)  MFCC-style coefficients
    #
    # Unnormalized DCT-II:
    #   X[k] = sum_{n=0}^{N-1} x[n] * cos(pi * (n + 0.5) * k / N)
    #
    n_frames, n_samples = mel_spectrogram.shape

    # --- Explicit / C++-friendly version (kept for porting) ---
    #for (int frame = 0; frame < n_frames; ++frame) {
    #   for (int sample_index_k = 0; sample_index_k < n_samples; ++sample_index_k) {
    #     float acc = 0.0f;
    #     for (int sample_index_n = 0; sample_index_n < n_samples; ++sample_index_n) {
    #       acc += mel[frame][sample_index_n] * cosf(PI * (sample_index_n + 0.5f) * sample_index_k / n_samples);
    #     }
    #     out[frame][sample_index_k] = acc;
    #   }
    # }
    # return out

    # Fast path: DCT-II basis matrix, then one matmul (NumPy -> BLAS/C)
    # dct_matrix[k, n] = cos(pi * (n + 0.5) * k / N)
    n = np.arange(n_samples, dtype=np.float32)
    k = np.arange(n_samples, dtype=np.float32)
    dct_matrix = np.cos(np.pi * np.outer(k, n + 0.5) / n_samples)
    dct_II = mel_spectrogram @ dct_matrix.T

    print("DCT-II shape:", dct_II.shape)
    #DCT-II shape: (149, 13)
    return dct_II


def inverse_discrete_fourier_transformation(mel_spectrogram):
    # Real cepstrum: IDFT along the mel axis (axis=1), one transform per time frame.
    # Input:  (n_frames, N) log-mel (or log spectrum) energies
    # Output: (n_frames, N) real cepstrum  (= real(IDFT(log-mel)))
    #
    # IDFT:
    #   x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+j * 2*pi*k*n / N)
    #
    n_frames, N = mel_spectrogram.shape

    # --- Explicit / C++-friendly real IDFT (kept for porting) ---
    # cepstrum = np.zeros((n_frames, N), dtype=np.float32)
    # for frame_index in range(n_frames):
    #     for n in range(N):
    #         acc_re = 0.0
    #         for k in range(N):
    #             angle = 2.0 * np.pi * k * n / N
    #             # real input → use cos term; take real cepstrum
    #             acc_re += mel_spectrogram[frame_index, k] * np.cos(angle)
    #         cepstrum[frame_index, n] = acc_re / N
    # return cepstrum

    # Fast path: NumPy IFFT (C under the hood), keep real part
    cepstrum = np.fft.ifft(mel_spectrogram, axis=1)
    return np.real(cepstrum).astype(np.float32)


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

    # modulus |X| = sqrt(re^2 + im^2); tf.abs on complex is that
    # (power spectrogram would be tf.abs(stft) ** 2 == re^2 + im^2)
    spectrogram = tf.abs(stft)
    #STFT shape: (149, 161)
    print("STFT shape:", spectrogram.shape)
    spectrogram = spectrogram.numpy()
    spectrogram = np.log10(spectrogram + 1e-6)
    return spectrogram

def get_mel_spectrogram_from_vanilla(spectrogram):    
    number_of_frames_inSpectrum = spectrogram.shape[0]
    n_freq_bins = spectrogram.shape[1]
    # apply mel scalling
    mel_spectrogram = np.zeros((number_of_frames_inSpectrum, MEL_DOTS))
    print("Mel spectrogram shape:", mel_spectrogram.shape)

    # --- Explicit / C++-friendly version (kept for porting) ---
    # # from 0 to 148
    # for frame_number in range(number_of_frames_inSpectrum):
    #     # from 0 to 12
    #     for mel_filter_bank_number in range(MEL_DOTS):
    #         # from (0, 2, 6) to (107, 131, 160)
    #         left, center, right = get_mel_filter_bank_bounds(mel_filter_bank_number)
    #         # from (1, to 5) to (108, 131, 159)
    #         for bin_number_in_mel_spectrogram in range(left+1, right-1):
    #             # mel filter bank
    #             filter_weight = triangular_filter_H(bin_number_in_mel_spectrogram, left, center, right)
    #             mel_spectrogram[frame_number, mel_filter_bank_number] += spectrogram[frame_number, bin_number_in_mel_spectrogram] * filter_weight
    # return mel_spectrogram

    # Fast path: prebuild H(k) matrix, then one matmul (NumPy -> BLAS/C)
    # Note: H(left)=H(right)=0, so endpoints do not change the sum.
    # Your loop used range(left+1, right-1); that skips bin right-1.
    # Comments say up to right-1 inclusive -> range(left+1, right). Fast path uses full H.
    mel_filter_bank = build_mel_filter_bank_matrix(n_freq_bins)
    mel_spectrogram = spectrogram @ mel_filter_bank

    return mel_spectrogram

def get_mfcc_from_mel_spectrogram(mel_spectrogram):
    # sometimes the mfcc coefficients are truncated (e.g. keep 12-13); here we keep all MEL_DOTS
    mfcc = discrete_cosine_transformation_type_2(mel_spectrogram)
    # shape: (n_frames, 13)

    # Simple first-order differences (easy to port to ESP32).
    # Classic HTK/librosa deltas use a wider regression window (±2); different values.
    #
    # --- Explicit / C++-friendly version (kept for porting) ---
    # n_frames, n_ceps = mfcc.shape
    # delta_mfcc = np.zeros_like(mfcc)
    # delta_delta_mfcc = np.zeros_like(mfcc)
    # for frame_index in range(1, n_frames):
    #     for coefficient_index in range(n_ceps):
    #         delta_mfcc[frame_index, coefficient_index] = (
    #             mfcc[frame_index, coefficient_index]
    #             - mfcc[frame_index - 1, coefficient_index]
    #         )
    #         delta_delta_mfcc[frame_index, coefficient_index] = (
    #             delta_mfcc[frame_index, coefficient_index]
    #             - delta_mfcc[frame_index - 1, coefficient_index]
    #         )
    #
    # Fast path (same math): frame 0 stays 0
    delta_mfcc = np.zeros_like(mfcc)
    delta_delta_mfcc = np.zeros_like(mfcc)
    delta_mfcc[1:] = mfcc[1:] - mfcc[:-1]
    delta_delta_mfcc[1:] = delta_mfcc[1:] - delta_mfcc[:-1]

    # each frame: [13 mfcc | 13 delta | 13 delta-delta] -> (n_frames, 39)
    return np.concatenate((mfcc, delta_mfcc, delta_delta_mfcc), axis=1)

# ----------------------------------------------------------
# TF SPECTROGRAM FROM WAV
# ----------------------------------------------------------


def create_tf_vanilla_spectrogram(wav_filename, png_filename, save_plot=False):

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

def create_tf_mel_spectrogram(spectrogram, png_filename, save_plot=False):

    mel_spectrogram = get_mel_spectrogram_from_vanilla(spectrogram)
    plt.figure(figsize=(12, 4))

    # uncomment only for debugging
    # print(f"TF Shape: {spectrogram.shape}, Min: {np.min(spectrogram)}, Max: {np.max(spectrogram)}, Mean: {np.mean(spectrogram)} , Std: {np.std(spectrogram)}")

    if save_plot == True:
        plt.imshow(
            mel_spectrogram.T,
            aspect="auto",
            origin="lower",
            cmap="viridis"
        )

        plt.title(
            "TensorFlow Mel Spectrogram"
        )

        plt.xlabel("Time frames")
        plt.ylabel("Mel Frequency bins")
        plt.colorbar(label="Mel Scale Magnitude")
        plt.tight_layout()

        plt.savefig(
            png_filename,
            dpi=300
        )

        plt.close()

    return mel_spectrogram

def create_tf_mfcc(mel_spectrogram, png_filename, save_plot=False):

    mfcc_coeficients = get_mfcc_from_mel_spectrogram(mel_spectrogram)
    plt.figure(figsize=(12, 4))

    # uncomment only for debugging
    # print(f"TF Shape: {spectrogram.shape}, Min: {np.min(spectrogram)}, Max: {np.max(spectrogram)}, Mean: {np.mean(spectrogram)} , Std: {np.std(spectrogram)}")

    if save_plot == True:
        plt.imshow(
            mfcc_coeficients.T,
            aspect="auto",
            origin="lower",
            cmap="viridis"
        )

        plt.title(
            "TensorFlow MFCC"
        )

        plt.xlabel("Time frames")
        plt.ylabel("MFCC Coefficients")
        plt.colorbar(label="MFCC Magnitude")
        plt.tight_layout()

        plt.savefig(
            png_filename,
            dpi=300
        )

        plt.close()

    return mfcc_coeficients

def create_tf_cepstrum(mel_spectrogram, png_filename, save_plot=False):

    cepstrum = inverse_discrete_fourier_transformation(mel_spectrogram)
    plt.figure(figsize=(12, 4))

    # uncomment only for debugging
    # print(f"TF Shape: {spectrogram.shape}, Min: {np.min(spectrogram)}, Max: {np.max(spectrogram)}, Mean: {np.mean(spectrogram)} , Std: {np.std(spectrogram)}")

    if save_plot == True:
        plt.imshow(
            cepstrum.T,
            aspect="auto",
            origin="lower",
            cmap="viridis"
        )

        plt.title(
            "TensorFlow Cepstrum"
        )

        plt.xlabel("Time frames")
        plt.ylabel("QueFrency")
        plt.colorbar(label="Cepstrum Magnitude")
        plt.tight_layout()

        plt.savefig(
            png_filename,
            dpi=300
        )

        plt.close()

    return cepstrum


def create_tf_cepstrum_frames(cepstrum, png_filename, save_plot=False):

    if save_plot == True:
        n_frames, n_ceps = cepstrum.shape
        # 10 equidistant frame indices covering the whole clip
        frame_indices = np.linspace(0, n_frames - 1, 10)
        frame_indices = np.unique(frame_indices.astype(int)).tolist()

        # Quefrency (s) for an N-point IDFT at this sample rate
        quefrency_sec = np.arange(n_ceps, dtype=np.float32) / SAMPLE_RATE

        n_plots = len(frame_indices)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.4 * n_rows), sharex=True)
        axes = np.atleast_1d(axes).ravel()

        for subplot_index, frame_index in enumerate(frame_indices):
            ax = axes[subplot_index]
            ax.plot(
                quefrency_sec,
                cepstrum[frame_index],
                color="green",
                label="Cepstrum",
            )
            ax.set_title(f"Frame {frame_index}")
            ax.set_xlabel("Quefrency (s)")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        for ax in axes[n_plots:]:
            ax.set_visible(False)

        fig.suptitle("Cepstrum")
        fig.tight_layout()
        fig.savefig(png_filename, dpi=300)
        plt.close(fig)

    return cepstrum
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
        png_filename = f"{base}_aug{idx}_tf_vanilla.png"
        mel_png_filename = f"{base}_aug{idx}_tf_mel.png"
        mfcc_png_filename = f"{base}_aug{idx}_tf_mfcc.png"
        cepstrum_png_filename = f"{base}_aug{idx}_tf_cepstrum.png"
        cepstrum_frames_png_filename = f"{base}_aug{idx}_tf_cepstrum_frames.png"
        spectogram = create_tf_vanilla_spectrogram(output_wav_filename, png_filename, False)
        mel_spectogram = create_tf_mel_spectrogram(spectogram, mel_png_filename, False)
        mfcc =  create_tf_mfcc(mel_spectogram, mfcc_png_filename, False)
        cepstrum = create_tf_cepstrum(mel_spectogram, cepstrum_png_filename, True)
        create_tf_cepstrum_frames(cepstrum, cepstrum_frames_png_filename, True)
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
    process_wav_file("../DATASET/SI/SI_10.wav")
    
    print("\nFinished.")
