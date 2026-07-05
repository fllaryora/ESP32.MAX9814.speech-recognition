import serial
import struct
import wave

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops

PORT = "/dev/ttyUSB0"
BAUD = 115200
SAMPLE_RATE = 16000

ser = serial.Serial(PORT, BAUD, timeout=10)


# ----------------------------------------------------------
# RUSSIAN NOTEBOOK FUNCTION
# https://github.com/DenissStepanjuk/ESP32.INMP441.speech-recognition/blob/main/Python_INMP441/INMP441-CNN-TFL.ipynb
# ----------------------------------------------------------

def get_spectrogram_original(audio):

    # misma normalización que ESP32
    audio = audio - np.mean(audio)

    max_val = np.max(np.abs(audio))
    if max_val < 1e-6:
        max_val = 1.0

    audio = audio / max_val

    # FFT exactamente de 320 puntos
    stft = tf.signal.stft(
        audio,
        frame_length=320,
        frame_step=160,
        fft_length=320,
        window_fn=tf.signal.hann_window
    )

    # módulo²
    spectrogram = tf.abs(stft) ** 2

    print("STFT shape:", spectrogram.shape)

    spectrogram = spectrogram.numpy()

    pooled = []

    for i in range(41):

        start = i * 4
        end = min(start + 4, spectrogram.shape[1])

        pooled.append(
            np.mean(
                spectrogram[:, start:end],
                axis=1
            )
        )

    spectrogram = np.stack(
        pooled,
        axis=1
    )

    spectrogram = np.log10(
        spectrogram + 1e-6
    )

    print("Pooled shape:", spectrogram.shape)

    return spectrogram


def get_normalized_pcm( wav_filename):

    try:
        # Load audio file
        audio_binary = tf.io.read_file(wav_filename)
        audio, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
        
        # Remove channel dimension: (samples, 1) -> (samples,)
        audio = tf.squeeze(audio, axis=-1)
        

        # Normalize to [-1, 1]
        audio = tf.cast(audio, tf.float32)
        ##### comment this part ?
        #audio = audio - tf.reduce_mean(audio)
        #max_val = tf.reduce_max(tf.abs(audio))
        #audio = tf.where(max_val > 0, audio / max_val, audio)
        #####

        audio = audio.numpy()

        return audio
        
    except Exception as e:
         raise RuntimeError(f"   ✗ Error loading {file_name}: {e}")

# ----------------------------------------------------------
# TF SPECTROGRAM FROM WAV
# ----------------------------------------------------------

def create_tf_spectrogram( wav_filename, png_filename):

    audio = get_normalized_pcm(wav_filename)


    spectrogram = get_spectrogram_original( audio)


    plt.figure(figsize=(12, 4))

    print(f"TF Shape: {spectrogram.shape}, Min: {np.min(spectrogram)}, Max: {np.max(spectrogram)}, Mean: {np.mean(spectrogram)} , Std: {np.std(spectrogram)}")
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
    plt.colorbar(label = "Log_10 Magnitude")
    plt.tight_layout()

    plt.savefig(
        png_filename,
        dpi=300
    )

    plt.close()

    return spectrogram


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

    print("MAE =", mae)
    print("RMSE =", rmse)

    diff = spec_tf - spec_esp

    print(f"Mean diff: {np.mean(diff)}")
    print(f"Standar diff: {np.std(diff)}")
    print(f"Max diff: {np.max(diff)}")
    print(f"Min diff: {np.min(diff)}")

    print(f"Pixels with diff bigger than 0.5: {biggerThanHalf}")
    print(f"Pixels with diff bigger than 1.0: {biggerThanOne}")

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

    ax[1].set_title(
        "ESP32"
    )

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

    expected_size = 299 * 41 * 4

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

    assert len(spectrogram) == 299 * 41

    spectrogram = spectrogram.reshape(299,41)
    
    print(f"ESP Shape: {spectrogram.shape}, Min: {np.min(spectrogram)}, Max: {np.max(spectrogram)}, Mean: {np.mean(spectrogram)} , Std: {np.std(spectrogram)}")

    plt.figure (figsize=(12,4))
    plt.imshow(
        spectrogram.T,
        aspect="auto",
        origin="lower",
        cmap="viridis"
    )
    plt.title("Spectrogram")
    plt.xlabel("Time frames")
    plt.ylabel("Frecuency Bins")
    plt.colorbar(label = "Log_10 Magnitude")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    return spectrogram

# ---------------------

labels = ["si", "no", "listo", "ayuda", "papel"]

for label in labels:
    for i in range(10):
        input(f'Say "{label}" and press ENTER...')
        wav_filename = f"{label}_{i}.wav"
        record_and_save(wav_filename)
        print(f'LPM')
        spectrogram_esp32 = spectogram_and_save(f"{label}_{i}.png")
        spectrogram_tensorFlow = create_tf_spectrogram( wav_filename, f"{label}_{i}_tf.png")
        compare_spectrograms(
        spectrogram_tensorFlow,
        spectrogram_esp32,
        f"{label}_{i}_compare.png")

print("Done")