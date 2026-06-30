import serial
import struct
import wave

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

def record_and_save(filename):
    ser.write(b'r')

    print("Wait for START header")
    # Wait for START header
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

    # Save WAV
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio)


# ------------------------


labels = ["si", "no", "listo", "ayuda", "papel"]

for label in labels:
    for i in range(10):
        input(f'Say "{label}" and press ENTER...')
        record_and_save(f"{label}_{i}.wav")

print("Done")