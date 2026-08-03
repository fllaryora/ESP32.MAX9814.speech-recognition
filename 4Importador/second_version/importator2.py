import serial
import struct
import wave
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops

PORT = "/dev/ttyUSB0"
BAUD = 921600 #460800 #115200
SAMPLE_RATE = 16000

ser = serial.Serial(PORT, BAUD, timeout=10)

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

# ----------------------------------------------------------
# COMPARISON
# ----------------------------------------------------------

def compare_spectrograms(
        spec_tf,
        spec_esp,
        png_filename):

    mae = np.mean(
        np.abs(
            spec_tf - spec_esp
        )
    )

    rmse = np.sqrt(
        np.mean(
            (spec_tf - spec_esp) ** 2
        )
    )

    biggerThanHalf = np.sum(
        np.abs(
            spec_tf - spec_esp
        ) > 0.5
    )

    biggerThanOne = np.sum(
        np.abs(
            spec_tf - spec_esp
        ) > 1.0
    )

    #print("MAE =", mae)
    #print("RMSE =", rmse)

    diff = spec_tf - spec_esp

    #print(f"Mean diff: {np.mean(diff)}")
    #print(f"Standar diff: {np.std(diff)}")
    #print(f"Max diff: {np.max(diff)}")
    #print(f"Min diff: {np.min(diff)}")

    #print(f"Pixels with diff bigger than 0.5: {biggerThanHalf}")
    #print(f"Pixels with diff bigger than 1.0: {biggerThanOne}")

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(18, 4)
    )

    img0 =  ax[0].imshow(
        spec_tf.T,
        aspect="auto",
        origin="lower",
        cmap="viridis"
    )

    ax[0].set_title(
        "TensorFlow"
    )

    fig.colorbar(img0, ax= ax[0], label = "Log_10 Magnitude")

    img1 = ax[1].imshow(
        spec_esp.T,
        aspect="auto",
        origin="lower",
        cmap="viridis"
    )

    ax[1].set_title( "ESP32" )

    fig.colorbar(img1, ax= ax[1], label = "Log_10 Magnitude")

    img2 = ax[2].imshow(
        diff.T,
        aspect="auto",
        origin="lower",
        cmap="seismic"
    )

    ax[2].set_title(
        f"Difference\nMAE={mae:.6f}"
    )

    fig.colorbar(img2, ax= ax[2], label = "Difference")

    plt.tight_layout()

    plt.savefig(
        png_filename,
        dpi=300
    )

    plt.close()

#---------------

def wait_for_start():
    while True:
        buffer = b''
        buffer = ser.read(5)
        print("it was read ",buffer)
        if buffer == b'START':
            return

def wait_for_frame():
    while True:
        buffer = b''
        buffer = ser.read(5)
        print("it was read ",buffer)
        if buffer == b'FRAME':
            return

def record_and_save(filename):
    ser.write(b'r')

    print("Wait for START header")

    wait_for_start()

    # Read size
    size_bytes = ser.read(4)
    size = struct.unpack("<I", size_bytes)[0]

    audio = bytearray()
    while len(audio) < size:
        chunk = ser.read(size - len(audio))
        if not chunk:
            raise RuntimeError("Timeout while reading audio data")
        audio.extend(chunk)

    # Read END marker
    end = ser.read(3)
    if end != b'END':
        raise RuntimeError("Invalid transmission ending")

    print("it was read END of wav")

    # Save WAV
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio)
    print("It was saved")

# ------------------------

def spectogram_and_save(filename):

    print("Wait for FRAME header")

    wait_for_frame()

    # Read size
    size_bytes = ser.read(4)
    size = struct.unpack("<I", size_bytes)[0]
    #FRAMES = 299
    FRAMES = 149
    #BINS = 41
    BINS = 13

    expected_size = FRAMES * BINS * 4

    if size != expected_size:
        raise RuntimeError(f"Expected size: {expected_size}, actual: {size}")

    image = bytearray()
    while len(image) < size:
        chunk = ser.read(size - len(image))
        if not chunk:
            raise RuntimeError("Timeout while reading image data")
        image.extend(chunk)

    # Read END marker
    end = ser.read(3)
    if end != b'END':
        raise RuntimeError("Invalid transmission ending")

    
    spectrogram = np.frombuffer(image, dtype=np.float32)
    print(f"Float recividos: {len(spectrogram)}")

    #FRAMES = 299
    FRAMES = 149
    assert len(spectrogram) == FRAMES * BINS

    spectrogram = spectrogram.reshape(FRAMES, BINS)
    
    #print(f"ESP Shape: {spectrogram.shape}, Min: {np.min(spectrogram)}, Max: {np.max(spectrogram)}, Mean: {np.mean(spectrogram)} , Std: {np.std(spectrogram)}")

    #plt.figure (figsize=(12,4))
    #plt.imshow(
    #    spectrogram.T,
    #    aspect="auto",
    #    origin="lower",
    #    cmap="viridis"
    #)
    #plt.title("Spectrogram")
    #plt.xlabel("Time frames")
    #plt.ylabel("Frecuency Bins")
    #plt.colorbar(label = "Log_10 Magnitude")
    #plt.tight_layout()
    #plt.savefig(filename, dpi=300)
    #plt.close()

    return spectrogram

# ---------------------

labels = ["AYUDA"]

os.makedirs("DATASET", exist_ok=True)
for label in labels:
    os.makedirs(f"DATASET/{label}", exist_ok=True)
    input(f'Say "{label}" and press ENTER...')
    for i in range(99, 100):
        wav_filename = f"DATASET/{label}/{label}_{i}.wav"
        mel_png_filename = f"DATASET/{label}/{label}_{i}_esp32_mel.png"
        tf_png_filename = f"DATASET/{label}/{label}_{i}_tf.png"
        tf_mel_filename = f"DATASET/{label}/{label}_{i}_tf_mel.png"
        compare_file = f"DATASET/{label}/{label}_{i}_compare.png"
        record_and_save(wav_filename)
        mel_spectrogram_esp32 = spectogram_and_save(mel_png_filename)
        spectrogram_tensorFlow = create_tf_vanilla_spectrogram( wav_filename, tf_png_filename, False)
        mel_spectrogram_tensorFlow =create_tf_mel_spectrogram(spectrogram_tensorFlow, tf_mel_filename, True)
        compare_spectrograms( mel_spectrogram_tensorFlow, mel_spectrogram_esp32, compare_file)

print("Done")
