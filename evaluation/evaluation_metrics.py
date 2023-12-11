import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report


def heatmap(data, ax):
    """
    A simple heatmap function.
    """
    sns.heatmap(
        data,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        cbar=False,
        ax=ax,
        annot_kws={"fontsize": 10},
        linewidths=0.1,
        linecolor="black",
    )


def get_class_accuracy(y_true, y_pred):
    """
    Get the accuracy for each class.
    """
    # Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize the confusion matrix along the row
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # The diagonal entries are the accuracies of each class
    class_accuracy = np.diag(cm_norm)

    return class_accuracy


def get_class_accuracy_cv(y_trues, y_preds):
    """
    Get the accuracy for each class from cross validation.
    """
    class_accuracies = []
    for y_true, y_pred in zip(y_trues, y_preds):
        class_accuracies.append(get_class_accuracy(y_true, y_pred))

    # Calcualte mean
    class_accuracies = np.array(class_accuracies)
    class_accuracies = class_accuracies.mean(axis=0)

    return class_accuracies


def confusion_matrix_cv(y_test, y_pred, save=False):
    # Plot confusion matrix for CatBoost
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    cm = confusion_matrix(y_test, y_pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels=["Wake", "NREM", "REM"])
    cmd.plot(cmap="Blues", ax=ax[0])

    # Remove colorbar
    ax[0].get_images()[0].colorbar.remove()
    ax[0].set_ylabel("True label", fontsize=12, labelpad=10)
    ax[0].set_xlabel("Predicted label", fontsize=12, labelpad=10)
    ax[0].set_title("Classification confusion matrix", fontsize=14, pad=10)

    # Show percentages and remove the underlying numbers
    for i in range(3):
        for j in range(3):
            ax[0].text(
                j,
                i + 0.2,
                f"({cm[i, j] / cm.sum(axis=1)[i]*100:.0f}%)",
                ha="center",
                va="center",
                color="white" if i == 1 and j == 1 else "black",
                fontsize=11,
                fontweight="bold",
            )
    """
    # TODO: FIX THE ROC CURVE CODE FOR CROSS VALIDATION

    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)


    RocCurveDisplay.from_predictions(
        y_onehot_test.ravel(),
        y_pred_proba_cat.ravel(),
        name="Fitted classifier",
        color="darkblue",
        ax=ax[1]
    )
    plot_roc_auc(y_test, np.random.rand(len(y_test)), ax[1], color="black", label="Random classifier", ls="--")


    ax[1].set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
    ax[1].set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
    ax[1].set_title("Micro-averaged One-vs-Rest\nROC-curve", fontsize=14)
    """
    # Remove the second subplot
    ax[1].axis("off")

    if save:
        plt.savefig("figures/confusion_matrix.png", dpi=300, bbox_inches="tight")

    plt.show()


def print_classification_report(y_tests, y_preds):
    """
    A hacky way to get the mean classification report for cross validation.
    """
    result_dict = {
        "Wake": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0},
        "NREM": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0},
        "REM": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0},
        "accuracy": 0.0,
        "macro avg": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0},
        "weighted avg": {
            "precision": 0.0,
            "recall": 0.0,
            "f1-score": 0.0,
            "support": 0,
        },
    }
    for y_test, y_pred in zip(y_tests, y_preds):
        clf_dict = classification_report(
            y_test,
            y_pred,
            target_names=["Wake", "NREM", "REM"],
            output_dict=True,
        )
        for stage in ["Wake", "NREM", "REM"]:
            for metric in ["precision", "recall", "f1-score", "support"]:
                result_dict[stage][metric] += clf_dict[stage][metric]
        result_dict["accuracy"] += clf_dict["accuracy"]
        for metric in ["precision", "recall", "f1-score", "support"]:
            result_dict["macro avg"][metric] += clf_dict["macro avg"][metric]
            result_dict["weighted avg"][metric] += clf_dict["weighted avg"][metric]

    for stage in ["Wake", "NREM", "REM"]:
        for metric in ["precision", "recall", "f1-score", "support"]:
            result_dict[stage][metric] /= len(y_tests)
    result_dict["accuracy"] /= len(y_tests)
    for metric in ["precision", "recall", "f1-score", "support"]:
        result_dict["macro avg"][metric] /= len(y_tests)
        result_dict["weighted avg"][metric] /= len(y_tests)

    return result_dict


def plot_feature_importances(original_feature_importance, save=False):
    # Plot the feature importances as a heatmap
    fig, ax = plt.subplots(figsize=(10, 4))

    heatmap(
        np.concatenate(
            (
                original_feature_importance[:33].reshape(3, 11),
                np.array(
                    [
                        original_feature_importance[34],
                        original_feature_importance[33],
                        original_feature_importance[35],
                    ]
                ).reshape(3, 1),
            ),
            axis=1,
        ),
        ax,
    )

    ax.set_aspect("equal")
    ax.set_yticklabels(
        ["Activity", "HR", "Circadian"], rotation=0, fontsize=12, fontweight="bold"
    )
    ax.set_xticklabels(
        [
            "-2.5",
            "-2",
            "-1.5",
            "-1",
            "-0.5",
            "0",
            "+0.5",
            "+1",
            "+1.5",
            "+2",
            "+2.5",
            "Agg",
        ]
    )
    ax.set_xlabel("Minutes before/after the label", fontsize=12, labelpad=10)
    ax.set_title("Feature importances", fontsize=14, fontweight="bold", pad=15)
    fig.align_ylabels(ax)

    if save:
        plt.savefig("figures/feature_importances.png", dpi=300, bbox_inches="tight")

    plt.show()

    def plot_prediction(y_valid, y_pred_valid):
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.plot(y_valid, color=cmap[3], lw=5, label="Label")
        ax.plot(y_pred_valid, color=cmap[1], lw=1.5, alpha=0.8, label="Prediction")

        # Bold titles
        ax.set_title("Sleep stage prediction", fontweight="bold", fontsize=14)
        ax.set_xlabel("Hours since sleep", fontsize=14)

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["wake", "NREM", "REM"])

        ax.legend(loc="upper left", frameon=False, fontsize=14)

        plt.show()
