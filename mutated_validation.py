"""implementation of Mutation  Validation Score"""

from enum import Enum
from dataclasses import dataclass
import random
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    roc_auc_score,
    recall_score
)
from imblearn.metrics import geometric_mean_score
import numpy as np

from mutate_labels import MutatedValidation

random.seed(42)
np.random.seed(42)


class EvaluationMetricsType(str, Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    ROC_AUC = "roc_auc"
    G_MEAN = "g_mean"


class TrainRun(int, Enum):
    ORIGINAL = 0
    MUTATED = 1


class ResultState(float, Enum):
    MAX = 1.0
    MIN = 0.0


@dataclass(slots=True)
class EvaluationMetric:
    predicted_labels: np.ndarray
    original_labels: np.ndarray
    metric_score: float | int


def get_metric_score(
        predicted_labels: np.ndarray, original_labels: np.ndarray, metric: str
) -> EvaluationMetric:
    score = ""
    match metric:
        case EvaluationMetricsType.ACCURACY:
            score = accuracy_score(original_labels, predicted_labels)
        case EvaluationMetricsType.RECALL:
            score = recall_score(original_labels, predicted_labels)
        case EvaluationMetricsType.PRECISION:
            score = precision_score(original_labels, predicted_labels)
        case EvaluationMetricsType.ROC_AUC:
            score = roc_auc_score(original_labels, predicted_labels)
        case EvaluationMetricsType.G_MEAN:
            score = geometric_mean_score(original_labels, predicted_labels)
    return EvaluationMetric(
        predicted_labels=predicted_labels,
        original_labels=original_labels,
        metric_score=score,  # type: ignore
    )


@dataclass(slots=True)
class MutatedValidationScore:
    mutated_labels: MutatedValidation
    mutate: np.ndarray
    original_training_predicted_labels: np.ndarray
    mutated_training_predicted_labels: np.ndarray
    evaluation_metric: str

    @property
    def mutated_training_metric_score_based_original_labels(self):
        return get_metric_score(
            predicted_labels=self.mutated_training_predicted_labels,
            original_labels=self.mutated_labels.labels,
            metric=self.evaluation_metric,
        )

    @property
    def original_training_metric_score_based_on_original_labels(self):
        return get_metric_score(
            predicted_labels=self.original_training_predicted_labels,
            original_labels=self.mutated_labels.labels,
            metric=self.evaluation_metric,
        )

    @property
    def mutated_training_metric_score_based_mutated_labels(self):
        return get_metric_score(
            predicted_labels=self.mutated_training_predicted_labels,
            original_labels=self.mutate,
            metric=self.evaluation_metric,
        )

    @property
    # def compute_mv_score(self):
    def get_mv_score(self):
        return (
                (
                        (1 - 2 * self.mutated_labels.perturbation_ratio)
                        * self.mutated_training_metric_score_based_original_labels.metric_score
                )
                + (
                        self.original_training_metric_score_based_on_original_labels.metric_score
                        - self.mutated_training_metric_score_based_mutated_labels.metric_score
                )
                + self.mutated_labels.perturbation_ratio
        )

    # @property
    # def get_mv_score(self):
    #     mv_score = self.compute_mv_score
    #     cond1 = mv_score <= ResultState.MAX.value
    #     match cond1:
    #         case True:
    #             return mv_score
    #         case False:
    #             return np.floor(mv_score)
    #     match mv_score <= ResultState.MAX:
    #         case True:
    #             return mv_score
    #         case False:
    #             return np.floor(mv_score)
    #
    #     # return mv_score if mv_score <= 1 else np.floor(mv_score)
