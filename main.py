from typing import Dict

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

from repen import utils
from repen.datagen import batch_generator
from repen.net import REPEN


def _check_datagen(dataset: str = 'wine'):
    data, labels = utils.read_dataset(dataset, normalized=True)
    print(f'data-shape: {data.shape}, labels-shape: {labels.shape}')

    (a, p, n), _ = batch_generator(
        data=data,
        candidate_scores=labels,
        batch_size=32,
    ).__next__()
    print(f'anchors-shape: {a.shape}')
    print(f'positives-shape: {p.shape}')
    print(f'negatives-shape: {n.shape}')
    return


def train_validate_test_split(data: np.array, labels: np.array, **params):
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.4)
    validate_x, test_x, validate_y, test_y = train_test_split(test_x, test_y, test_size=0.5)

    train_gen = batch_generator(train_x, train_y, **params)
    validation_gen = batch_generator(validate_x, validate_y, **params)
    test_gen = batch_generator(test_x, test_y, **params)

    return train_gen, validation_gen, test_gen


if __name__ == '__main__':
    _dataset = 'musk'
    _model_name = f'test_model_{_dataset}'
    _delete_old = False

    if _delete_old:
        import shutil
        shutil.rmtree(utils.MODELS_DIR)
        shutil.rmtree(utils.LOGS_DIR)

        _model_number = 0
    else:
        _model_number = utils.increment_model_number(_model_name, 0)

    utils.create_dirs()

    _datagen_params: Dict = {
        'batch_size': 256,
        'cutoff_threshold': np.sqrt(3),
        'unsorted': True,
    }

    _data, _labels = utils.read_dataset(_dataset, normalized=True, adjust=True)
    _initial_scores = utils.contaminate_labels(np.copy(_labels), fraction=0.1)

    # TODO: Use LeSiNN or other unsupervised detector to generate initial scores
    # _base_detector = utils.lesinn
    # _initial_scores = _base_detector(_data)

    _train_gen, _val_gen, _test_gen = train_validate_test_split(_data, _initial_scores, **_datagen_params)

    _train_params: Dict = {
        'steps_per_epoch': max(_data.shape[0] // _datagen_params['batch_size'], 256),
        'num_epochs': 64,
        'verbose': 1,
        'es_schedule': [1e-5, 10],
        'lr_schedule': [0.1, 8, 4],
    }
    _train_params['validation_steps'] = _train_params['steps_per_epoch'] // 4

    _model = REPEN(
        model_name=f'{_model_name}_{_model_number}',
        input_dim=_data.shape[1],
        step_size=20,
        embedding_size=20,
        triplet_alpha=1000.,
    )
    _model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    _model.summary()

    # exit(1)

    _model.train(_train_gen, _val_gen, **_train_params)
    _model.save()

    _evaluation = _model.evaluate(_test_gen, _train_params['steps_per_epoch'] * 4)
    if isinstance(_evaluation, list):
        _line = ', '.join([f'{_p:.4f}' for _p in _evaluation])
    else:
        _line = f'loss: {_evaluation:.4f}'

    print(f'model {_model.model_name}, performance: {_line}')

    # _metrics = _model.predict(_data, _labels, _base_detector)
    # print(_metrics)
