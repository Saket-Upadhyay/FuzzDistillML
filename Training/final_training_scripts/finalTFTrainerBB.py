#!/usr/bin/env python
import os

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from includes.constants import BB_TF_EPOCHS, BB_TF_BATCH
from includes.helpers import get_bb_train_test_set

def train_dnn_basicblock_classification_model(save_location):

    X_train, X_test, y_train, y_test = get_bb_train_test_set()


    # Build the binary classification model
    model = Sequential(
        [
            Input(shape=(X_train.shape[1],)),
            Dense(128, activation="relu"),  # Input layer (with 64 neurons)
            Dense(64, activation="relu"),  # Input layer (with 64 neurons)
            Dense(32, activation="relu"),  # Hidden layer
            Dense(
                1, activation="sigmoid"
            ),  # Output layer (single neuron for binary classification)
        ]
    )


    # Compile the model
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train,
        y_train,
        epochs=BB_TF_EPOCHS,
        batch_size=BB_TF_BATCH,
        validation_data=(X_test, y_test),
    )
    # Save the entire model (architecture, weights, optimizer, and training configuration)
    loss, accuracy = model.evaluate(X_test, y_test)
    model.save(
        os.path.join(os.path.relpath(save_location),f"BBpredict_TF_{str(BB_TF_EPOCHS)}_{str(BB_TF_BATCH)}_{round(accuracy*100)}.keras")
    )

    return float(accuracy * 100)
