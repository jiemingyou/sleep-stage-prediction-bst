import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from constants import WINDOW_SIZE, EPOCH_SIZE


def load_data(path: str):
    """
    Load the data from the given path. The output is a dataframe with the
    following columns:
        - id: user id
        - time: time in hours since the beginning of the recording
        - count: activity count
        - hr: heart rate
        - cosine: circadian phase
        - psg: sleep stage

    Args:
        path (str): Path to the folder containing the data.

    Returns:
        pd.DataFrame: Dataframe containing the data from the given path.
    """
    files = os.listdir(path)
    users = [f.split("_")[0] for f in files]
    data = pd.DataFrame()
    for u in tqdm(users):
        # Files corresponding to the user
        user_files = [
            f
            for f in os.listdir(path)
            if f.startswith(u) and not f.startswith(".") and not "README" in f
        ]
        user_data = pd.DataFrame()

        # Get the data for each file
        for f in user_files:
            # Find the word between _ and _ using regex
            measurement = re.search("(?<=_)(.*)(?=_)", f).group(0)

            # Read the file
            df = pd.read_csv(path + f, names=[measurement])

            # Add it to the user data
            user_data = pd.concat([user_data, df], axis=1)

        user_data.insert(0, "id", u)

        # Add user data to the full data
        data = pd.concat([data, user_data], axis=0)

    return data


def split_users(users, test_size=0.2):
    """
    Split the users into train and test sets.
    """
    # Split the users unto train and test
    train_users, test_users = train_test_split(users, test_size=test_size)

    return train_users, test_users


def get_sleep_stage_label(x):
    return {
        0: 0,  # Wake
        1: 1,  # N1
        2: 1,  # N2
        3: 1,  # N3
        5: 2,  # REM
    }.get(x, 0)


def make_features(df, users):
    """
    Construct features from the given dataframe.
    """

    # Modify the sleep stage labels to be 0, 1, 2, 3
    df = df[df.psg != 4].reset_index(drop=True)
    df["psg"] = df["psg"].apply(get_sleep_stage_label)

    # Features
    X_raw = []
    y_raw = []

    # Construct features by taking 5 minutes of data around each time point
    for user in tqdm(users):
        df_user = df[df["id"] == user]
        df_user = df_user.sort_values("time")
        df_user_features = df_user[["count", "hr", "cosine", "time"]]

        df_user_labels = df_user["psg"]

        n_rows = df_user_features.shape[0]
        buffer = WINDOW_SIZE // 2
        for epoch in range(buffer, n_rows - buffer - 1):
            # Add the label to the data
            y_raw.append(df_user_labels.iloc[epoch])

            # Get the data for the window
            df_window = df_user_features.iloc[epoch - buffer : epoch + buffer + 1, :]

            # Aggregated features
            hr_mean = df_window["hr"].mean()
            total_count = df_window["count"].sum()
            time = df_window["time"].iloc[buffer]

            # Make it a numpy array
            df_window = df_window[["count", "hr", "cosine"]].to_numpy().T.reshape(-1)

            # Add the mean HR and total count to the data
            df_window = np.append(df_window, [hr_mean, total_count, time])

            # Add the features to the data
            X_raw.append(df_window)

    return np.array(X_raw), np.array(y_raw)


def svd(X_train, X_test, n_components=10):
    """
    Perform SVD on the given data.
    """
    svd = TruncatedSVD(n_components=n_components)
    X_train_svd = svd.fit_transform(X_train)
    X_test_svd = svd.transform(X_test)

    return X_train_svd, X_test_svd


def cross_validate(
    df,
    catbst,
    svd,
    n=5,
):
    """
    Perform cross validation on the given data.
    """
    # K-fold
    kfold = KFold(n_splits=n, shuffle=True, random_state=42)

    # Accuracies
    labels = []
    predictions = []
    feature_importances = []

    users = df["id"].unique()

    # K-fold loop
    for i, (train_idx, test_idx) in enumerate(kfold.split(users)):
        print(f"Fold {i+1}/{n}")

        # Split the users into train and test
        train_users = users[train_idx]
        test_users = users[test_idx]

        # Make features
        X_train, y_train = make_features(df, train_users)
        X_test, y_test = make_features(df, test_users)
        labels.append(y_test)

        # SVD
        X_train = svd.fit_transform(X_train)
        X_test = svd.transform(X_test)

        # Fit the model
        catbst.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
        )

        # Predict the classes
        y_pred_cat = catbst.predict(X_test)
        predictions.append(y_pred_cat)

        # Reverse engineer the feature importances
        importances = catbst.get_feature_importance()
        top_components = np.argsort(-importances)[:3]
        original_feature_loadings = svd.components_[top_components, :]
        original_feature_importance = np.sum(np.abs(original_feature_loadings), axis=0)
        feature_importances.append(original_feature_importance)

    return labels, predictions, feature_importances


if __name__ == "__main__":
    data = load_data("data/")
    print(data.head(5))
