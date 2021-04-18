import os
from functools import reduce
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from tensorflow import keras

from . import utils


class Triplet(keras.layers.Layer):
    def __init__(self, alpha: float, **kwargs):
        self.is_placeholder = True
        self.alpha: float = alpha
        super(Triplet, self).__init__(**kwargs)

    # noinspection PyMethodMayBeStatic
    def _euclidean_sq(self, x, y):
        return keras.backend.sum(keras.backend.square(x - y), axis=-1)

    def loss(self, anchor, positive, negative):
        positive_distances = self._euclidean_sq(anchor, positive)
        negative_distances = self._euclidean_sq(anchor, negative)
        loss = negative_distances - positive_distances
        return keras.backend.mean(keras.backend.maximum(0., self.alpha - loss))

    def call(self, inputs, **kwargs):
        loss = self.loss(*inputs)
        self.add_loss(loss, inputs=inputs)
        return inputs[0]  # this output is only used for saving the model


class REPEN:
    def __init__(
            self,
            model_name: str,
            input_dim: int,
            step_size: int,
            embedding_size: int,
            triplet_alpha: float
    ):
        if not input_dim >= max(step_size, embedding_size):
            raise ValueError(f'input-dim {input_dim} is too small compared to step-size {step_size} and embedding-size {embedding_size}')

        if not (embedding_size <= step_size):
            raise ValueError(f'embedding-size {embedding_size} should be smaller than step-size {step_size}.')

        self.model_name: str = model_name

        input_anchor = keras.layers.Input(shape=(input_dim,), name='input_anchor')
        input_positive = keras.layers.Input(shape=(input_dim,), name='input_positive')
        input_negative = keras.layers.Input(shape=(input_dim,), name='input_negative')

        # layers = [
        #     keras.layers.Dense(units=units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))
        #     for units in reversed(range(step_size, input_dim, step_size))
        # ]
        layers = list()
        layers.append(keras.layers.Dense(
            units=embedding_size,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(1e-3),
            name='hidden_output'
        ))

        layer_apply = lambda x: reduce(lambda res, layer: layer(res), layers, x)

        hidden_anchor = layer_apply(input_anchor)
        hidden_positive = layer_apply(input_positive)
        hidden_negative = layer_apply(input_negative)

        output_layer = Triplet(alpha=triplet_alpha)([hidden_anchor, hidden_positive, hidden_negative])

        self.model = keras.models.Model(
            inputs=[input_anchor, input_positive, input_negative],
            outputs=output_layer,
        )

        self.representation = keras.models.Model(
            inputs=input_anchor,
            outputs=hidden_anchor,
        )

    def summary(self):
        return self.representation.summary()

    def compile(self, optimizer):
        self.model.compile(optimizer=optimizer, loss=None)
        return

    def train(
            self,
            train_generator,
            validation_generator,
            steps_per_epoch: int = 1024,
            num_epochs: int = 128,
            validation_steps: int = 128,
            verbose: int = 1,
            es_schedule: Tuple[float, int] = None,
            lr_schedule: Tuple[float, int, int] = None,
    ):
        callbacks = [keras.callbacks.TensorBoard(
            log_dir=os.path.join(utils.LOGS_DIR, self.model_name),
        )]

        if es_schedule is not None:
            delta, patience = es_schedule
            if patience > num_epochs:
                raise ValueError(f'es_schedule[0], i.e. the number of epochs to wait, should not be greater than the total number of epochs')
            if not isinstance(delta, float) or not (0. < delta):
                raise ValueError(f'es_schedule[1], i.e. the minimum delta by which LR must fall, should be a small, positive float e.g. 1e-4.')
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=delta,
                patience=patience,
                verbose=verbose,
                restore_best_weights=True,
            ))

        if lr_schedule is not None:
            factor, patience, cooldown = lr_schedule
            if not isinstance(factor, float) or not (0. < factor < 1.):
                raise ValueError(f'lr_schedule[0], i.e. the factor by which to reduce the learning rate, should be a float in the (0, 1) range.')
            if not isinstance(patience, int) or patience > num_epochs:
                raise ValueError(f'lr_schedule[1], i.e. the number of epochs to wait for val_loss to fall, should be an int no larger than num_epochs.')
            if not isinstance(cooldown, int) or cooldown > patience:
                raise ValueError(f'lr_schedule[2], i.e. the number of epochs to wait between reducing the LR, '
                                 f'should be an int no larger than lr_patience, i.e. lr_schedule[1].')

        self.model.fit(
            x=train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=validation_generator,
            validation_steps=validation_steps,
        )
        return

    def evaluate(
            self,
            evaluate_generator: keras.utils.Sequence,
            steps: int,
            verbose: int = 1,
    ):
        return self.model.evaluate(evaluate_generator, verbose=verbose, steps=steps)

    def predict(
            self,
            x_test: np.array,
            y_true: np.array,
            base_detector: Callable[[np.array], np.array],
            verbose: int = 1,
            return_metrics: bool = True,
    ):
        representation = self.representation.predict(x_test, verbose=verbose)
        y_pred = base_detector(representation)
        return utils.calculate_metrics(y_true, y_pred) if return_metrics else y_pred

    def save(self):
        path = os.path.join(utils.MODELS_DIR, self.model_name)
        self.model.save(filepath=path)
        return

    @staticmethod
    def load(model_params: Dict, model_name: str) -> 'REPEN':
        model = REPEN(**model_params)
        path = os.path.join(utils.MODELS_DIR, model_name)
        model.model = keras.models.load_model(path)
        return model
