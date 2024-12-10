#!/usr/bin/env python
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from includes.constants import FN_TF_BATCH, FN_TF_EPOCHS
from includes.helpers import get_function_train_test_set


def train_dnn_function_classification_model(save_location):
    X_train, X_test, y_train, y_test = get_function_train_test_set()
    act = 'relu'
    model = Sequential([
        Input(shape=(X_train.shape[1],)),               # Dropout with 50% probability
        Dense(128, activation=act),  # First dense layer
        Dropout(0.3),                  # Dropout with 50% probability
        Dense(64, activation=act),  # Second dense layer
        Dropout(0.2),                  # Dropout with 30% probability
        Dense(32, activation=act),  # Third dense layer
        Dropout(0.1),                  # Dropout with 20% probability
        Dense(1, activation='sigmoid') # Output layer
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=FN_TF_EPOCHS, batch_size=FN_TF_BATCH, validation_data=(X_test, y_test),callbacks=[early_stopping])

    loss, accuracy = model.evaluate(X_test, y_test)
    model.save(os.path.join(os.path.relpath(save_location),f"FNpredict_TF_{str(FN_TF_EPOCHS)}_{str(FN_TF_BATCH)}_{round(accuracy*100)}.keras"))
    return float(accuracy * 100)
