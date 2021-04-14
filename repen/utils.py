import os
from typing import List
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import normalize

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = '/data/abd/chaoda_data'

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
NORMALIZED_DATA_DIR = os.path.join(DATA_DIR, 'normalized')

CHAODA_DATASETS = [
    'annthyroid',
    'arrhythmia',
    'breastw',
    'cardio',
    'cover',
    'glass',
    'http',
    'ionosphere',
    'lympho',
    'mammography',
    'mnist',
    'musk',
    'optdigits',
    'pendigits',
    'pima',
    'satellite',
    'satimage-2',
    'shuttle',
    'smtp',
    'thyroid',
    'vertebral',
    'vowels',
    'wbc',
    'wine',
]

REPEN_DATASETS = [
    'backdoor',
    'campaign',
    'celeba',
    'census',
    'donors',
    'fraud',
    'thyroid-21',
]

MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

DIRS = [
    MODELS_DIR,
    LOGS_DIR,
    RESULTS_DIR,
]


def create_dirs():
    for _dir in DIRS:
        os.makedirs(_dir, exist_ok=True)
    return


def read_dataset(dataset: str, *, normalized: bool, adjust: bool = True) -> Tuple[np.array, np.array]:
    """ Read a dataset by name and return the 2-d data array and the 1-d labels array.

    :param dataset: name of dataset to read.
    :param normalized: whether to read the normalized version.
    :param adjust: whether to adjust the labels to be floats in the [0.01, 0.99] range to avoid div-by-zero errors.
    :return: tuple of 2-d data array and 1-d labels array
    """
    if normalized:
        data_path = os.path.join(NORMALIZED_DATA_DIR, f'{dataset}.npy')
    else:
        data_path = os.path.join(RAW_DATA_DIR, f'{dataset}.npy')
    labels_path = os.path.join(RAW_DATA_DIR, f'{dataset}_labels.npy')

    data = np.asarray(np.load(data_path), dtype=np.float32)
    labels = np.asarray(np.load(labels_path), dtype=np.uint8)

    if adjust:
        labels = np.asarray(labels, dtype=np.float32) * 0.98 + 0.01

    return data, np.squeeze(labels)


def contaminate_labels(labels: np.array, *, fraction: float) -> np.array:
    """ Randomly flips labels (0 -> 1, 1 -> 0) with probability = fraction

    :param labels: 1-d array of [0, 1] integer labels
    :param fraction: chance to flip each label
    :return: contaminated labels
    """
    if not (0. <= fraction <= 1.):
        raise ValueError(f'fraction is a probability so it should be a float in the [0, 1] range.')

    for i in range(labels.shape[0]):
        if np.random.uniform() <= fraction:
            labels[i] = 1 - labels[i]
    return labels


def get_model_path(model_name: str, model_number: int) -> str:
    """ create and return the path where this model will be saved.
    """
    return os.path.join(MODELS_DIR, f'{model_name}-{model_number}')


def increment_model_number(model_name: str, model_number: int) -> int:
    """ Increments the model number until reaching a new number for saving experimental models.
    """
    filepath = get_model_path(model_name, model_number)
    while os.path.exists(filepath):
        model_number += 1
        filepath = get_model_path(model_name, model_number)
    return model_number


def calculate_metrics(
        y_true: np.array,
        y_pred: np.array,
) -> Tuple[float, float, float]:
    """ Calculate the precision, accuracy, and recall from the predictions.'
    """
    precision: float = precision_score(y_true, y_pred, average='macro')
    accuracy: float = accuracy_score(y_true, y_pred)
    recall: float = recall_score(y_true, y_pred, average='macro')
    return precision, accuracy, recall


def normalize_data():
    def _normalize_data(datasets: List[str]):
        for dataset in datasets:
            raw_path = os.path.join(RAW_DATA_DIR, f'{dataset}.npy')
            normalized_path = os.path.join(NORMALIZED_DATA_DIR, f'{dataset}.npy')

            data = np.asarray(np.load(raw_path), dtype=np.float32)
            data = normalize(data, axis=0)
            np.save(
                file=normalized_path,
                arr=np.asarray(data, dtype=np.float32),
                allow_pickle=False,
                fix_imports=False,
            )
        return

    _normalize_data(CHAODA_DATASETS)
    _normalize_data(REPEN_DATASETS)
    return


def lesinn(data: np.array) -> np.array:
    # TODO
    raise NotImplemented(f'LeSiNN has not yet been implemented')


if __name__ == '__main__':
    create_dirs()
