import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostClassifier

from preprocessing.feature_engineering import (
    load_data,
    split_users,
    make_features,
    svd,
    cross_validate,
)
from evaluation.evaluation_metrics import (
    get_class_accuracy_cv,
    confusion_matrix_cv,
    print_classification_report,
    plot_feature_importances,
)
from figures.construct_plots import plot_data

from constants import WINDOW_SIZE, EPOCH_SIZE


def main():
    # Load the data
    data = load_data("data/")
    print("Data loaded.", end="\n")

    # Initiate the model and the SVD transformer
    catbst = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=100,
        depth=6,
        learning_rate=0.05,
        verbose=False,
        auto_class_weights="SqrtBalanced",
    )
    svd = TruncatedSVD(n_components=10)

    # Train the model using cross-validation
    print("Training model...")
    y_tests, y_preds, feature_importances = cross_validate(data, catbst, svd, n=5)

    # Evaluate the accuracy of the model
    accs = get_class_accuracy_cv(y_tests, y_preds)
    print("\nWake\tNREM\tREM", end="\n")
    print("\t".join(accs.round(2).astype(str)))

    # Confusion matrix
    confusion_matrix_cv(
        np.concatenate(y_tests),
        np.concatenate(y_preds).reshape(-1),
        save=True,
    )

    # Classification report
    print_classification_report(y_tests, y_preds)

    # Feature importances
    plot_feature_importances(np.array(feature_importances).mean(axis=0), save=True)


if __name__ == "__main__":
    main()
