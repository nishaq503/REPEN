import numpy as np

from . import utils


def _cutoff(scores: np.array, cutoff_threshold: float, unsorted: bool = True):
    if unsorted:
        indices = np.arange(start=0, stop=scores.shape[0])
        scores = scores
    else:
        indices = np.argsort(scores, axis=0)
        scores = scores[indices]

    scores_mean = np.mean(scores)
    scores_std = np.std(scores)
    threshold = scores_mean + scores_std * cutoff_threshold
    if threshold > np.max(scores):  # use 90th percentile outlier score
        threshold = np.percentile(scores, 90)

    candidate_outlier_indices = np.where(scores > threshold)[0]
    candidate_inlier_indices = np.where(scores <= threshold)[0]

    if not unsorted:
        candidate_outlier_indices = indices[candidate_outlier_indices]
        candidate_inlier_indices = indices[candidate_inlier_indices]

    return candidate_inlier_indices, candidate_outlier_indices


def _generate_one_batch(
        data: np.array,
        inlier_indices: np.array,
        positive_weights: np.array,
        outlier_indices: np.array,
        negative_weights: np.array,
        batch_size: int,
):
    anchors = np.zeros([batch_size], dtype=int)
    positives = np.zeros([batch_size], dtype=int)
    negatives = np.zeros([batch_size], dtype=int)

    for i in range(batch_size):
        anchor_sample = np.random.choice(inlier_indices.shape[0], p=positive_weights)
        anchors[i] = inlier_indices[anchor_sample]

        positive_sample = np.random.choice(inlier_indices.shape[0])
        while anchor_sample == positive_sample:
            positive_sample = np.random.choice(inlier_indices.shape[0], 1)
        positives[i] = inlier_indices[positive_sample]

        negative_sample = np.random.choice(outlier_indices.shape[0], p=negative_weights)
        negatives[i] = outlier_indices[negative_sample]

    anchors = data[anchors]
    positives = data[positives]
    negatives = data[negatives]

    return anchors, positives, negatives


def batch_generator(
        data: np.array,
        candidate_scores: np.array,
        batch_size: int,
        cutoff_threshold: float = None,
        unsorted: bool = True,
):
    cutoff_threshold: float = np.sqrt(3) if cutoff_threshold is None else cutoff_threshold
    inlier_indices, outlier_indices = _cutoff(candidate_scores, cutoff_threshold, unsorted)

    transforms = np.sum(candidate_scores[inlier_indices]) - candidate_scores[inlier_indices]

    total_weights_positive = np.sum(transforms)
    positive_weights = (transforms / total_weights_positive).flatten()

    total_weights_negative = np.sum(candidate_scores[outlier_indices])
    negative_weights = (candidate_scores[outlier_indices] / total_weights_negative).flatten()

    while True:
        anchors, positives, negatives = _generate_one_batch(
            data=data,
            inlier_indices=inlier_indices,
            positive_weights=positive_weights,
            outlier_indices=outlier_indices,
            negative_weights=negative_weights,
            batch_size=batch_size,
        )
        yield [anchors, positives, negatives], None
