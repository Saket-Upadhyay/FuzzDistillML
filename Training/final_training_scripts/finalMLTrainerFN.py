# Test Multiple Classifiers for Function features
import os.path

import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from includes.helpers import get_function_train_test_set
from includes.constants import GLOBAL_RANDOM_STATE
import pickle


def save_object(obj, loc):
    try:
        with open(os.path.join(loc,"XGBFunctionModel.pickle"), "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def train_xboost_function_classification_model(save_location):

    X_train, X_test, y_train, y_test = get_function_train_test_set()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=GLOBAL_RANDOM_STATE,
        colsample_bytree=0.8,
        learning_rate=0.05,
        max_depth=10,
        n_estimators=400,
        subsample=0.8
    )

    model.fit(
        X_train_scaled, y_train
    )

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    save_object(model,save_location)

    return float(accuracy * 100)

