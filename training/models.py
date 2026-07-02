from __future__ import annotations

import tensorflow as tf

from training.config import IMAGE_SIZE


def build_custom_cnn() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="valid"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="valid"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="valid"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="retrained_custom_cnn",
    )
    return model


def compile_custom_cnn(model: tf.keras.Model, learning_rate: float = 0.001) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc", curve="ROC"),
        ],
    )
    return model
