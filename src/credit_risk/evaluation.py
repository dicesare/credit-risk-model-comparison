from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    roc_auc: float
    average_precision: float
    recall: float


def evaluate_binary_predictions(y_true, probabilities, threshold: float = 0.5) -> BinaryMetrics:
    from sklearn.metrics import average_precision_score, recall_score, roc_auc_score

    predictions = [int(value >= threshold) for value in probabilities]
    return BinaryMetrics(
        round(float(roc_auc_score(y_true, probabilities)), 4),
        round(float(average_precision_score(y_true, probabilities)), 4),
        round(float(recall_score(y_true, predictions)), 4),
    )
