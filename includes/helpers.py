import pandas as pd
from sklearn.model_selection import train_test_split

from includes.constants import (
    BB_DATASET_FILE,
    FN_DATASET_FILE,
    GLOBAL_RANDOM_STATE
)
from includes.constants import (
    BB_TARGET_FEATURE,
    BB_CPP_MEMORY_FEATURES,
    BB_EXPLICIT_EXCLUDE_FEATURES,
    BB_UNIMPORTANT_FEATURES,
)
from includes.constants import (
    FN_UNIMPORTANT_FEATURES,
    FN_TARGET_FEATURE,
    FN_CPP_MEMORY_FEATURES,
    FN_EXPLICIT_EXCLUDE_FEATURES,
)


def get_bb_train_test_set():
    data = pd.read_csv(BB_DATASET_FILE, sep=";")
    X = data.drop(
        BB_UNIMPORTANT_FEATURES
        + BB_TARGET_FEATURE
        + BB_CPP_MEMORY_FEATURES
        + BB_EXPLICIT_EXCLUDE_FEATURES,
        axis=1,
    )  # Exclude both ID, Name, and target column
    y = data[BB_TARGET_FEATURE[0]]
    return train_test_split(X, y, test_size=0.2, random_state=GLOBAL_RANDOM_STATE)


def get_function_train_test_set():
    data = pd.read_csv(FN_DATASET_FILE, sep=";")
    X = data.drop(
        FN_UNIMPORTANT_FEATURES
        + FN_TARGET_FEATURE
        + FN_CPP_MEMORY_FEATURES
        + FN_EXPLICIT_EXCLUDE_FEATURES,
        axis=1,
    )  # Exclude both ID, Name, and target column
    y = data[FN_TARGET_FEATURE[0]]

    return train_test_split(X, y, test_size=0.2, random_state=GLOBAL_RANDOM_STATE, stratify=y)
