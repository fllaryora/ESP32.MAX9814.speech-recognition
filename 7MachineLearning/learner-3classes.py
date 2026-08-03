import os
import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)

# from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import itertools
import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

SAMPLE_RATE = 16000

DATASET_DIR = "./DATASET"
SECONDS_OF_WAV = 24000  # 1500 mili seconds
Epochs = 30  # 30 is better than 50.
MIN_FRECUENCY = 0
MAX_FRECUENCY = 8000
SAMPLES_FOR_FFT = 320
FRAME_STEP = 160  # next frame is 160 samples after the previous one
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
# 50
VANILLA_WIDTH_BETWEEN_BINS = SAMPLE_RATE / SAMPLES_FOR_FFT
# 202.8588
MEL_WIDTH_BETWEEN_BINS = (MAX_MEL_FRECUENCY - MIN_MEL_FRECUENCY) / (MEL_DOTS + 1)


# --------
# It returns the mel frecuency from the filter number
# ----
def get_mel_frecuency_from_filter_number(mel_filter_bank_number):
    return MIN_MEL_FRECUENCY + mel_filter_bank_number * MEL_WIDTH_BETWEEN_BINS


# ------
# It return the bin index from the filter number
# ---
def get_bound_from_filter_number(mel_filter_bank_number):
    mel_frecuency = get_mel_frecuency_from_filter_number(mel_filter_bank_number)
    vanilla_frecuency = get_vanilla_frecuency(mel_frecuency)
    return int(vanilla_frecuency / VANILLA_WIDTH_BETWEEN_BINS)


# ------
# It returns the left, center and right bounds of the filter bank index
# ---
def get_mel_filter_bank_bounds(mel_filter_bank_number):
    # TODO: verify mel filter bank indexing / MFCC convention against the ESP32 path
    # Triangular mel filter: left and right are the centers of the neighbors.
    left = get_bound_from_filter_number(mel_filter_bank_number)
    center = get_bound_from_filter_number(mel_filter_bank_number + 1)
    right = get_bound_from_filter_number(mel_filter_bank_number + 2)
    return (left, center, right)


# -----
# It returns the triangular filter weight for the bin number in the mel spectrogram
# ----
def triangular_filter_H(bin_number_in_mel_spectrogram, left, center, right):
    # H(k): triangular weight for FFT bin k given filter bank bounds (bin indices).
    if bin_number_in_mel_spectrogram < left or bin_number_in_mel_spectrogram > right:
        return 0.0
    # increasing
    if bin_number_in_mel_spectrogram <= center:
        return (bin_number_in_mel_spectrogram - left) / (center - left)
    # decreasing
    return (right - bin_number_in_mel_spectrogram) / (right - center)


# ------
# It builds the mel filter bank matrix
# ---
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


# --------------------------------------------------
# WAV --> PCM
# --------------------------------------------------


def get_normalized_pcm(wav_filename):

    try:
        # Load audio file
        audio_binary = tf.io.read_file(wav_filename)
        audio, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
        audio = audio[:SECONDS_OF_WAV]

        # Remove channel dimension: (samples, 1) -> (samples,)
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)
        audio = audio.numpy()

        return audio

    except Exception as e:
        raise RuntimeError(f"   ✗ Error loading {wav_filename}: {e}")


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
        window_fn=tf.signal.hann_window,
    )

    # modulus |X| = sqrt(re^2 + im^2); tf.abs on complex is that
    # (power spectrogram would be tf.abs(stft) ** 2 == re^2 + im^2)
    spectrogram = tf.abs(stft)
    # STFT shape: (149, 161)
    #print("STFT shape:", spectrogram.shape)
    spectrogram = spectrogram.numpy()
    spectrogram = np.log10(spectrogram + 1e-6)
    return spectrogram


def get_mel_spectrogram_from_vanilla(spectrogram):
    number_of_frames_inSpectrum = spectrogram.shape[0]
    n_freq_bins = spectrogram.shape[1]
    # apply mel scalling
    mel_spectrogram = np.zeros((number_of_frames_inSpectrum, MEL_DOTS))
    #print("Mel spectrogram shape:", mel_spectrogram.shape)

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

    if save_plot == True:
        plt.imshow(spectrogram.T, aspect="auto", origin="lower", cmap="viridis")

        plt.title("TensorFlow Spectrogram")

        plt.xlabel("Time frames")
        plt.ylabel("Frequency bins")
        plt.colorbar(label="Log_10 Magnitude")
        plt.tight_layout()

        plt.savefig(png_filename, dpi=300)

        plt.close()

    return spectrogram




# --------------------------------------------------
# BUILD DATASET
# Directory --> [spectrogram, CNN class]
# --------------------------------------------------


def load_datasetALL(root_folder):

    X_data = []
    Y_labels = []

    classes = sorted(
        [
            directory
            for directory in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, directory))
        ]
    )

    # print("\nClasses found:")
    # print(classes)

    for class_name in classes:
        class_dir = os.path.join(root_folder, class_name)

        for file in os.listdir(class_dir):
            if not file.lower().endswith(".wav"):
                continue

            wav_path = os.path.join(class_dir, file)

            try:
                # print(f"Loading: {wav_path}")
                audio = get_normalized_pcm(wav_path)
                if len(audio) < SECONDS_OF_WAV:
                    continue
                spectrogram = get_spectrogram_original(audio)
                mel_spectrogram = get_mel_spectrogram_from_vanilla(spectrogram)
                mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
                # print("expand_dims shape:", spectrogram.shape)
                X_data.append(mel_spectrogram)
                Y_labels.append(class_name)

            except Exception as e:
                print(f"ERROR {wav_path}")

                print(e)

    X_data = np.array(X_data, dtype=np.float32)
    # print("X_data shape:", X_data.shape)
    return (X_data, Y_labels, classes)


def load_dataset(
    root_folder, VALID_CLASSES, UNKNOWN_CLASS, take_others_as_unknown=True
):

    X_data = []
    Y_labels = []

    classes = sorted(
        [
            directory
            for directory in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, directory))
        ]
    )

    # print("\nOriginal classes found:")
    # print(classes)

    for class_name in classes:
        class_dir = os.path.join(root_folder, class_name)

        if take_others_as_unknown == True:
            if class_name in VALID_CLASSES:
                target_label = class_name
            else:
                target_label = UNKNOWN_CLASS
        else:
            if class_name in VALID_CLASSES:
                target_label = class_name
            else:
                continue

        for file in os.listdir(class_dir):
            if not file.lower().endswith(".wav"):
                continue

            wav_path = os.path.join(class_dir, file)

            try:
                audio = get_normalized_pcm(wav_path)
                if len(audio) < SECONDS_OF_WAV:
                    continue
                spectrogram = get_spectrogram_original(audio)
                mel_spectrogram = get_mel_spectrogram_from_vanilla(spectrogram)
                mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
                X_data.append(mel_spectrogram)
                Y_labels.append(target_label)

            except Exception as e:
                print(f"ERROR {wav_path}")
                print(e)

    X_data = np.array(X_data, dtype=np.float32)

    label_names = sorted(list(set(Y_labels)))

    return (X_data, Y_labels, label_names)


# --------------------------------------------------
# MODEL
# Thumb rule for calcularion the arena size in ESP32 C++ is:
# HEIGHT * IMAGE_WIDTH * FILTERS[0] = ARENA_SIZE_0
# HALF_HEIGHT * HALFI_MAGE_WIDTH * FILTERS[1] = ARENA_SIZE_1
# ARENA_SIZE_0 + ARENA_SIZE_1 = ARENA_SIZE
# --------------------------------------------------


def create_model(image_height, image_width, channels, num_labels):
    #filters = [12, 12, 8, 8]
    filters = [16, 16, 8, 8]
    # Initializing the model.
    model_CNN = Sequential()

    # 1st convolutional layer (input to the convolutional layer).
    model_CNN.add(
        Conv2D(
            filters[0],
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=(image_height, image_width, channels),
        )
    )
    # Max pooling layer for the first convolution layer.
    model_CNN.add(MaxPool2D((2, 2)))
    model_CNN.add(Conv2D(filters[1], (3, 3), activation="relu", padding="same"))
    model_CNN.add(MaxPool2D((2, 2)))
    model_CNN.add(Conv2D(filters[2], (3, 3), activation="relu", padding="same"))
    model_CNN.add(MaxPool2D((2, 2)))
    # A layer for converting multidimensional data into a one-dimensional vector.
    model_CNN.add(Flatten())
    # model_CNN.add( GlobalAveragePooling2D() )

    # 1st fully connected layer.
    model_CNN.add(Dense(filters[3], activation="relu"))
    # it means reduce overfitting when unknow is read
    # model_CNN.add(Dropout(0.3))
    # in smalls networks use 0 dropout.

    # 2nd fully connected layer (model output).
    model_CNN.add(Dense(num_labels, activation="softmax"))

    # model_CNN.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"] )
    # model_CNN.compile( optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05), metrics=["accuracy"] )
    model_CNN.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0),
        metrics=["accuracy"],
    )
    # Output a textual representation of the neural network model structure.
    model_CNN.summary()
    return model_CNN


def plot_confusion_matrix(cm, class_names, file_name):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    if hasattr(cm, "numpy"):
        cm = cm.numpy()
    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(f"confusion_matrix_{file_name}.png", dpi=300)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    print("\nLoading dataset...")
    VALID_CLASSES = {"SI", "NO", "BASURA"}
    UNKNOWN_CLASS = "BASURA"

    X_data, Y_labels, label_names = load_dataset(
        DATASET_DIR, VALID_CLASSES, UNKNOWN_CLASS, take_others_as_unknown=False
    )

    # print("\nDataset loaded")
    # print("X_data:",X_data.shape)
    # print("Samples:",len(Y_labels))

    encoder = LabelEncoder()

    Y_encoded = encoder.fit_transform(Y_labels)

    Y_categorical = to_categorical(Y_encoded)

    # # Let's split the dataset into two parts.
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_data, Y_categorical, train_size=0.8, random_state=42, stratify=Y_encoded
    )

    # print( "\nX_train:", X_train.shape )
    # print( "X_test:", X_test.shape )

    # Let's define an object for dynamically adjusting the learning rate during model training.
    # monitor: This parameter specifies which metric should be monitored to determine whether to reduce the learning rate.
    # patience = 2: If there is no improvement over the course of two epochs, the learning rate will be reduced.
    # verbose = 1: If set to 1, messages about a decrease in the learning rate will be displayed in the console.
    # factor=0.25: This is the factor by which the current learning rate will be scaled if no improvement is observed.
    # min_lr: This is the minimum acceptable value for the learning rate.
    # learning_rate_reduction = ReduceLROnPlateau( monitor="val_accuracy", patience=8, verbose=1, factor=0.5,min_lr=0.000001 )

    learning_rate_reduction = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )

    HEIGHT = 149
    model_CNN = create_model(
        image_height = HEIGHT, image_width = MEL_DOTS, channels=1, num_labels=len(label_names)
    )

    model_CNN.summary()
    # Number of training iterations. epochs
    # Start training the model.
    history = model_CNN.fit(
        X_train,
        Y_train,
        validation_data=(X_test, Y_test),
        epochs=Epochs,
        batch_size=32,
        callbacks=[learning_rate_reduction, early_stopping],
    )

    loss, accuracy = model_CNN.evaluate(X_test, Y_test, verbose=0)

    # print( f"\nTest Accuracy: {accuracy:.4f}")

    model_CNN.save(f"speech_commands_{HEIGHT}x{MEL_DOTS}.h5")

    # epochs = [i for i in range(Epochs)]
    epochs = range(len(history.history["accuracy"]))
    fig, ax = plt.subplots(1, 2)
    train_acc = history.history["accuracy"]
    train_loss = history.history["loss"]
    val_acc = history.history["val_accuracy"]
    val_loss = history.history["val_loss"]
    fig.set_size_inches(16, 9)

    ax[0].plot(epochs, train_acc, "go-", label="Training Accuracy")
    ax[0].plot(epochs, val_acc, "ro-", label="Testing Accuracy")
    ax[0].set_title("Training Accuracy & Testing Accuracy")
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(epochs, train_loss, "g-o", label="Training Loss")
    ax[1].plot(epochs, val_loss, "r-o", label="Testing Loss")
    ax[1].set_title("Training Loss & Testing Loss")
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    plt.savefig("accuracy_loss.png", dpi=300)
    # print( "\nDONE")

    # 1. We receive predictions
    predictions = model_CNN.predict(X_test)

    # 2. Converting one-hot encoding to labels
    y_true = tf.argmax(Y_test, axis=1)
    y_pred = tf.argmax(predictions, axis=1)

    # 3. Building an error matrix
    cm = tf.math.confusion_matrix(labels=y_true, predictions=y_pred)

    # 4. Let's calculate the error matrix
    plot_confusion_matrix(cm, label_names, "keras")

# --------------------------------------------------
# TFLITE CONVERSION
# --------------------------------------------------

def representative_dataset():
    for i in range(len(X_train)):
        # We wrap each row in a batch of size (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS).
        data = np.expand_dims(X_train[i].astype(np.float32), axis=0)

        yield [data]


    # print("\nConverting to TFLite...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model_CNN)
    converter.representative_dataset = representative_dataset
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(f"speech_commands_{HEIGHT}x{MEL_DOTS}.tflite", "wb") as f:
        f.write(tflite_model)


    # --------------------------------------------------
    # LOAD TFLITE MODEL
    # --------------------------------------------------
    interpreter = tf.lite.Interpreter(model_path=f"speech_commands_{HEIGHT}x{MEL_DOTS}.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print("\nInput details:")
    # print(input_details)
    # print("\nOutput details:")
    # print(output_details)
    input_scale, input_zero_point = input_details[0]["quantization"]
    # print("\nInput scale:",input_scale)
    # print("Input zero point:",input_zero_point)

    # --------------------------------------------------
    # EVALUATE TFLITE MODEL
    # --------------------------------------------------

    predictions_tflite = []
    predicted_labels = []
    true_labels = []

    for i in range(X_test.shape[0]):
        X_input = X_test[i].astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], np.expand_dims(X_input, axis=0))
        interpreter.invoke()
        tflite_prediction = interpreter.get_tensor(output_details[0]["index"])
        predictions_tflite.append(tflite_prediction[0])
        predicted_class = np.argmax(tflite_prediction)
        true_class = np.argmax(Y_test[i])
        predicted_labels.append(predicted_class)
        true_labels.append(true_class)

    # --------------------------------------------------
    # TFLITE ACCURACY
    # --------------------------------------------------
    tflite_accuracy = accuracy_score(true_labels, predicted_labels)
    print("\nTFLite Accuracy:", tflite_accuracy)

    # --------------------------------------------------
    # TFLITE CONFUSION MATRIX
    # --------------------------------------------------
    cm_tflite = tf.math.confusion_matrix(labels=true_labels, predictions=predicted_labels)
    plot_confusion_matrix(cm_tflite, label_names, "tflite")
    print("\nTFLite validation complete.")
