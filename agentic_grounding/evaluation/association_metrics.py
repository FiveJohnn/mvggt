from __future__ import annotations

from collections.abc import Sequence


def pairwise_association_metrics(
    predicted_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
) -> dict[str, float]:
    if len(predicted_ids) != len(ground_truth_ids):
        raise ValueError("predicted_ids and ground_truth_ids must have equal length")
    tp = fp = fn = 0
    for i in range(len(predicted_ids)):
        for j in range(i + 1, len(predicted_ids)):
            predicted_same = predicted_ids[i] == predicted_ids[j]
            actual_same = ground_truth_ids[i] == ground_truth_ids[j]
            tp += int(predicted_same and actual_same)
            fp += int(predicted_same and not actual_same)
            fn += int(not predicted_same and actual_same)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}

