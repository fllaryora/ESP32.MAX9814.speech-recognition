import serial
import struct
import wave

import numpy as np
import matplotlib.pyplot as plt

PORT = "/dev/ttyUSB0"
BAUD = 115200
SAMPLE_RATE = 16000

ser = serial.Serial(PORT, BAUD, timeout=10)


def wait_for_start():
    while True:
        buffer = b''
        buffer = ser.read(5)
        print("se leyo ",buffer)
        if buffer == b'START':
            return

def wait_for_frame():
    while True:
        buffer = b''
        buffer = ser.read(5)
        print("se leyo ",buffer)
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

    print("se leyo END")

    # Save WAV
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio)
    print("Se guardo")

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
    
    print(f"Shape: {spectrogram.shape}, Min: {np.min(spectrogram)}, Max: {np.max(spectrogram)}")

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
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ---------------------

labels = ["si", "no", "listo", "ayuda", "papel"]

for label in labels:
    for i in range(10):
        input(f'Say "{label}" and press ENTER...')
        record_and_save(f"{label}_{i}.wav")
        print(f'LPM')
        spectogram_and_save(f"{label}_{i}.png")

print("Done")