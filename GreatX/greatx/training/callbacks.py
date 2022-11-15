# The following codes are mainly borrowed and adapted from tensorflow.
# You may refer to tensorflow for more details:
#
#     https://github.com/tensorflow/tensorflow
#
# Copyright The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import os
import time

import numpy as np
import torch

from greatx.utils import BunchDict, Progbar


class ModeKeys(object):
    """Standard names for model modes.

    The following standard keys are defined:

    * `TRAIN`: training/fitting mode.
    * `TEST`: testing/evaluation mode.
    * `PREDICT`: prediction/inference mode.
    """

    TRAIN = 'train'
    TEST = 'test'
    PREDICT = 'predict'


class CallbackList:
    """Container abstracting a list of callbacks."""
    def __init__(self, callbacks=None, add_history=False, add_progbar=False,
                 model=None, **params):
        """Container for `Callback` instances.

        This object wraps a list of `Callback` instances, making it possible
        to call them all at once via a single endpoint
        (e.g. `callback_list.on_epoch_end(...)`).

        Args:
          callbacks: List of `Callback` instances.
          add_history: Whether a `History` callback should be added,
            if one does not already exist in the `callbacks` list.
          add_progbar: Whether a `ProgbarLogger` callback should be added,
            if one does not already exist in the `callbacks` list.
          model: The `Model` these callbacks are used with.
          **params: If provided, parameters will be passed to each
            `Callback` via `Callback.set_params`.
        """
        self.callbacks = callbacks if callbacks else []
        self._add_default_callbacks(add_history, add_progbar)

        if model:
            self.set_model(model)
        if params:
            self.set_params(params)

        self._check_timing = any(cbk.__class__.__name__ not in globals()
                                 for cbk in self.callbacks)
        self._num_batches_for_timing_check = 5
        self._hook_times = {}
        self._batch_start_time = None
        self._batch_times = []

    def _add_default_callbacks(self, add_history, add_progbar):
        """Adds `Callback`s that are always present."""
        self._progbar = None
        self._history = None

        for cb in self.callbacks:
            if isinstance(cb, ProgbarLogger):
                self._progbar = cb
            elif isinstance(cb, History):
                self._history = cb

        if self._progbar is None and add_progbar:
            self._progbar = ProgbarLogger()
            self.callbacks.insert(0, self._progbar)

        if self._history is None and add_history:
            self._history = History()
            self.callbacks.append(self._history)

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        self.model = model
        if self._history:
            model.history = self._history
        for callback in self.callbacks:
            callback.set_model(model)

    def _call_batch_hook(self, mode, hook, batch, logs=None):
        """Helper function for all batch_{begin | end} methods."""
        if not self.callbacks:
            return

        if hook == 'begin':
            self._call_batch_begin_hook(mode, batch, logs)
        elif hook == 'end':
            self._call_batch_end_hook(mode, batch, logs)
        else:
            raise ValueError('Unrecognized hook: {}'.format(hook))

    def _call_batch_begin_hook(self, mode, batch, logs):
        """Helper function for `on_*_batch_begin` methods."""
        hook_name = 'on_{mode}_batch_begin'.format(mode=mode)
        self._call_batch_hook_helper(hook_name, batch, logs)

        if self._check_timing:
            self._batch_start_time = time.time()

    def _call_batch_end_hook(self, mode, batch, logs):
        """Helper function for `on_*_batch_end` methods."""
        hook_name = 'on_{mode}_batch_end'.format(mode=mode)

        if self._check_timing and batch >= 1:
            batch_time = time.time() - self._batch_start_time
            self._batch_times.append(batch_time)

        self._call_batch_hook_helper(hook_name, batch, logs)

        if len(self._batch_times) >= self._num_batches_for_timing_check:
            end_hook_name = hook_name
            begin_hook_name = 'on_{mode}_batch_begin'.format(mode=mode)
            avg_batch_time = sum(self._batch_times) / len(self._batch_times)
            avg_end_hook_time = sum(self._hook_times[end_hook_name]) / len(
                self._hook_times[end_hook_name])
            avg_begin_hook_time = sum(self._hook_times[begin_hook_name]) / len(
                self._hook_times[begin_hook_name])

            threshold_time = 1.0 * avg_batch_time
            warning_msg = (
                'Callback method `{hook}` is slow compared to '
                'the batch time (batch time: {batch_time:.4f}s vs '
                '`{hook}` time: {hook_time:.4f}s). Check your callbacks.')
            if avg_begin_hook_time > threshold_time:
                logging.warning(
                    warning_msg.format(hook=begin_hook_name,
                                       batch_time=avg_batch_time,
                                       hook_time=avg_begin_hook_time))
            if avg_end_hook_time > threshold_time:
                logging.warning(
                    warning_msg.format(hook=end_hook_name,
                                       batch_time=avg_batch_time,
                                       hook_time=avg_end_hook_time))
            self._check_timing = False
            self._batch_start_time = None
            self._batch_times = []
            self._hook_times = {}

    def _call_batch_hook_helper(self, hook_name, batch, logs):
        """Helper function for `on_*_batch_*` methods."""
        logs = logs or {}
        if self._check_timing:
            start_time = time.time()

        for callback in self.callbacks:
            hook = getattr(callback, hook_name)
            hook(batch, logs)

        if self._check_timing:
            if hook_name not in self._hook_times:
                self._hook_times[hook_name] = []
            self._hook_times[hook_name].append(time.time() - start_time)

    def _call_begin_hook(self, mode):
        """Helper function for on_{train|test|predict}_begin methods."""
        if mode == ModeKeys.TRAIN:
            self.on_train_begin()
        elif mode == ModeKeys.TEST:
            self.on_test_begin()
        else:
            self.on_predict_begin()

    def _call_end_hook(self, mode):
        """Helper function for on_{train|test|predict}_end methods."""
        if mode == ModeKeys.TRAIN:
            self.on_train_end()
        elif mode == ModeKeys.TEST:
            self.on_test_end()
        else:
            self.on_predict_end()

    def on_batch_begin(self, batch, logs=None):
        self._call_batch_hook(ModeKeys.TRAIN, 'begin', batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self._call_batch_hook(ModeKeys.TRAIN, 'end', batch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        """Calls the `on_epoch_begin` methods of its callbacks.

        This function should only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument
                for this method but that may change in the future.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Calls the `on_epoch_end` methods of its callbacks.

        This function should only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed.
              Validation result keys are prefixed with `val_`.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        """Calls the `on_train_batch_begin` methods of its callbacks.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict, contains the return value of `model.train_step`.
                Typically, the values of the `Model`'s metrics are returned.
                Example: `{'loss': 0.2, 'acc': 0.7}`.
        """
        self._call_batch_hook(ModeKeys.TRAIN, 'begin', batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        """Calls the `on_train_batch_end` methods of its callbacks.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        self._call_batch_hook(ModeKeys.TRAIN, 'end', batch, logs=logs)

    def on_test_batch_begin(self, batch, logs=None):
        """Calls the `on_test_batch_begin` methods of its callbacks.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict, contains the return value of `model.test_step`.
                Typically, the values of the `Model`'s metrics are returned.
                Example: `{'loss': 0.2, 'acc': 0.7}`.
        """
        self._call_batch_hook(ModeKeys.TEST, 'begin', batch, logs=logs)

    def on_test_batch_end(self, batch, logs=None):
        """Calls the `on_test_batch_end` methods of its callbacks.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        self._call_batch_hook(ModeKeys.TEST, 'end', batch, logs=logs)

    def on_predict_batch_begin(self, batch, logs=None):
        """Calls the `on_predict_batch_begin` methods of its callbacks.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict, contains the return value of `model.predict_step`,
              it typically returns a dict with a key 'outputs' containing
              the model's outputs.
        """
        self._call_batch_hook(ModeKeys.PREDICT, 'begin', batch, logs=logs)

    def on_predict_batch_end(self, batch, logs=None):
        """Calls the `on_predict_batch_end` methods of its callbacks.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        self._call_batch_hook(ModeKeys.PREDICT, 'end', batch, logs=logs)

    def on_train_begin(self, logs=None):
        """Calls the `on_train_begin` methods of its callbacks.

        Args:
            logs: Dict. Currently no data is passed to this argument
                for this method but that may change in the future.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Calls the `on_train_end` methods of its callbacks.

        Args:
            logs: Dict. Currently no data is passed to this argument
                for this method but that may change in the future.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_test_begin(self, logs=None):
        """Calls the `on_test_begin` methods of its callbacks.

        Args:
            logs: Dict. Currently no data is passed to this argument
                for this method but that may change in the future.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        """Calls the `on_test_end` methods of its callbacks.

        Args:
            logs: Dict. Currently no data is passed to this argument
                for this method but that may change in the future.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_predict_begin(self, logs=None):
        """Calls the 'on_predict_begin` methods of its callbacks.

        Args:
            logs: Dict. Currently no data is passed to this argument
                for this method but that may change in the future.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_predict_begin(logs)

    def on_predict_end(self, logs=None):
        """Calls the `on_predict_end` methods of its callbacks.

        Args:
            logs: Dict. Currently no data is passed to this argument
                for this method but that may change in the future.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_predict_end(logs)

    def __iter__(self):
        return iter(self.callbacks)

    def __getitem__(self, index):
        return self.callbacks[index]

    def __str__(self) -> str:

        format_string = ""
        for ix, cb in enumerate(self.callbacks):
            format_string += f'\n  ({ix}){cb},'
        if format_string:
            # replace last ``,`` as ``\n``
            format_string = format_string[:-1] + '\n'
        return f"{self.__class__.__name__}({format_string})"

    __repr__ = __str__


class Callback:
    """Abstract base class used to build new callbacks.

    Callbacks can be passed to keras methods such as `fit`,
    `evaluate`, and `predict` in order to hook into the various
    stages of the model training and inference lifecycle.

    See https://www.tensorflow.org/guide/keras/custom_callback
    for more information.

    Attributes:
        params: Dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: Instance of `keras.models.Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch (see method-specific docstrings).
    """
    def __init__(self):
        self.validation_data = None
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model
        self.model.stop_training = False

    def on_batch_begin(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_begin`."""

    def on_batch_end(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_end`."""

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.

        Subclasses should override for any actions to run.
        This function should only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument
                for this method but that may change in the future.
        """

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.

        Subclasses should override for any actions to run.
        This function should only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch,
                and for the validation epoch if validation is performed.
                Validation result keys are prefixed with `val_`.
                For training epoch, the values of the
                `Model`'s metrics are returned.
                Example : `{'loss': 0.2, 'acc': 0.7}`.
        """

    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be
        called every `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict, contains the return value of `model.train_step`.
                Typically, the values of the `Model`'s metrics are returned.
                Example: `{'loss': 0.2, 'acc': 0.7}`.
        """
        # For backwards compatibility.
        self.on_batch_begin(batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be
        called every `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        # For backwards compatibility.
        self.on_batch_end(batch, logs=logs)

    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `evaluate` methods.

        Also called at the beginning of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be
        called every `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict, contains the return value of `model.test_step`.
                Typically, the values of the `Model`'s metrics are returned.
                Example: `{'loss': 0.2, 'acc': 0.7}`.
        """

    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `evaluate` methods.

        Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be
        called every `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """

    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be
        called every `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict, contains the return value of `model.predict_step`,
              it typically returns a dict with a key 'outputs' containing
              the model's outputs.
        """

    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be
        called every `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument
                for this method but that may change in the future.
        """

    def on_train_end(self, logs=None):
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
                `on_epoch_end()` is passed to this argument for this method
                but that may change in the future.
        """

    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument
                for this method but that may change in the future.
        """

    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_test_batch_end()` is passed to this argument for this method
              but that may change in the future.
        """

    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument
                for this method but that may change in the future.
        """

    def on_predict_end(self, logs=None):
        """Called at the end of prediction.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument
                for this method but that may change in the future.
        """

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"

    __repr__ = __str__


class History(Callback):
    """Callback that records events into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """
    def __init__(self):
        super().__init__()
        self.history = BunchDict()

    def on_train_begin(self, logs=None):
        self.epoch = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # Set the history attribute on the model after
        # the epoch ends.
        # This will make sure that the state which
        # is set is the latest one.
        self.model.history = self


class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN
    loss is encountered. """
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            loss = float(loss)
            if np.isnan(loss) or np.isinf(loss):
                print('Batch %d: Invalid loss, terminating training' % (batch))
                self.model.stop_training = True


class ModelCheckpoint(Callback):
    """Callback to save the Keras model or model weights
    at some frequency.

    """
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=True, save_weights_only=True, autoload=True,
                 mode='auto', **kwargs):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath if filepath.endswith(
            '.pth') else filepath + '.pth'
        self.save_best_only = save_best_only
        self.epochs_since_last_save = 0
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        self._filepaths = []

        if autoload and not save_weights_only:
            logging.warning(
                '`autoload` is only work for `save_weights_only=True`, '
                'fallback to `save_weights_only.')
            save_weights_only = True

        self.save_weights_only = save_weights_only
        self.autoload = autoload

        if mode not in ['auto', 'min', 'max']:
            logging.warning(
                'ModelCheckpoint mode %s is unknown, '
                'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'loss' in self.monitor:
                self.monitor_op = np.less
                self.best = np.Inf
            else:
                self.monitor_op = np.greater
                self.best = -np.Inf

    def on_train_begin(self, logs=None):
        folder = os.path.split(self.filepath)[0]
        if folder:
            folder = folder
            if self.verbose > 0:
                print(f"mkdir {folder}.")
            os.mkdir(folder)

    def on_train_end(self, logs=None):
        if self.autoload and self._filepaths:
            self.model.load_state_dict(torch.load(self._filepaths[-1]))
            for filepath in self._filepaths:
                if os.path.exists(filepath):
                    os.remove(filepath)

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        self._save_model(epoch=epoch, logs=logs)

    def _save_model(self, epoch, logs):
        """Saves the model.

        Args:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end`
                or `on_epoch_end`.
        """
        logs = logs or {}

        filepath = self._get_file_path(epoch, logs)

        try:
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning(
                        'Can save best model only with %s available, '
                        'skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f '
                                  'to %0.5f, saving model to %s' %
                                  (epoch + 1, self.monitor, self.best, current,
                                   filepath))
                        self.best = current
                        if self.save_weights_only:
                            torch.save(self.model.state_dict(), filepath)
                        else:
                            torch.save(self.model, filepath)
                        self._filepaths.append(filepath)
                    else:
                        if self.verbose > 0:
                            print(
                                '\nEpoch %05d: %s did not improve from %0.5f' %
                                (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' %
                          (epoch + 1, filepath))
                if self.save_weights_only:
                    torch.save(self.model.state_dict(), filepath)
                else:
                    torch.save(self.model, filepath)
                self._filepaths.append(filepath)

        except IOError as e:
            if 'is a directory' in str(e.args[0]).lower():
                raise IOError('Please specify a non-directory filepath for '
                              'ModelCheckpoint. Filepath used is an existing '
                              'directory: {}'.format(filepath))
            # Re-throw the error for any other causes.
            raise e

    def _get_file_path(self, epoch, logs):
        """Returns the file path for checkpoint."""
        try:
            # `filepath` may contain placeholders such as `{epoch:02d}` and
            # `{mape:.2f}`. A mismatch between logged metrics and the path's
            # placeholders can cause formatting to fail.
            file_path = self.filepath.format(epoch=epoch + 1, **logs)
        except KeyError as e:
            raise KeyError('Failed to format this callback filepath: "{}". '
                           'Reason: {}'.format(self.filepath, e))
        return file_path

    def __str__(self) -> str:
        format_string = ""
        attrs = {
            'monitor', 'verbose', 'filepath', 'save_best_only',
            'save_weights_only'
        }
        for attr in attrs:
            format_string += f'\n  {attr}={getattr(self,attr)},'
        if format_string:
            # replace last ``,`` as ``\n``
            format_string = format_string[:-1] + '\n'
        return f"{self.__class__.__name__}({format_string})"

    __repr__ = __str__


class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving.

    Assuming the goal of a training is to minimize the loss. With this, the
    metric to be monitored would be `'loss'`, and mode would be `'min'`. A
    `model.fit()` training loop will check at end of every epoch whether
    the loss is no longer decreasing, considering the `min_delta` and
    `patience` if applicable. Once it's found no longer decreasing,
    `model.stop_training` is marked True and the training terminates.

    The quantity to be monitored needs to be available in `logs` dict.
    To make it so, pass the loss or metrics at `model.compile()`.

    Args:
      monitor: Quantity to be monitored.
      min_delta: Minimum change in the monitored quantity
          to qualify as an improvement, i.e. an absolute
          change of less than min_delta, will count as no
          improvement.
      patience: Number of epochs with no improvement
          after which training will be stopped.
      verbose: verbosity mode.
      mode: One of `{"auto", "min", "max"}`. In `min` mode,
          training will stop when the quantity
          monitored has stopped decreasing; in `"max"`
          mode it will stop when the quantity
          monitored has stopped increasing; in `"auto"`
          mode, the direction is automatically inferred
          from the name of the monitored quantity.
      baseline: Baseline value for the monitored quantity.
          Training will stop if the model doesn't show improvement over the
          baseline.

    """
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0,
                 mode='auto', baseline=None):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.mode = mode

        if mode not in ['auto', 'min', 'max']:
            logging.warning(
                'EarlyStopping mode %s is unknown, '
                'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'loss' in self.monitor:
                self.monitor_op = np.less
            else:
                self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            if self.baseline is None or self._is_improvement(
                    current, self.baseline):
                self.wait = 0

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s',
                self.monitor, ','.join(list(logs.keys())))
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)

    def __str__(self) -> str:
        format_string = ""
        attrs = {'monitor', 'verbose', 'patience', 'min_delta', 'mode'}
        for attr in attrs:
            format_string += f'\n  {attr}={getattr(self,attr)},'
        if format_string:
            # replace last ``,`` as ``\n``
            format_string = format_string[:-1] + '\n'
        return f"{self.__class__.__name__}({format_string})"

    __repr__ = __str__


class ProgbarLogger(Callback):
    """Callback that prints metrics to stdout.
    TODO: on_[test/predict]_[begin/end] haven't been tested.
    """
    def __init__(self):
        super().__init__()
        # Defaults to all Model's metrics except for loss.
        self.seen = 0
        self.progbar = None
        self.target = None
        self.verbose = None
        self.epochs = None

    def set_params(self, params):
        self.verbose = params['verbose']
        self.epochs = params['epochs']
        self.target = self.epochs

    def on_train_begin(self, logs=None):
        self._reset_progbar()

    def on_test_begin(self, logs=None):
        self._reset_progbar()
        self._maybe_init_progbar()

    def on_predict_begin(self, logs=None):
        self._reset_progbar()
        self._maybe_init_progbar()

    def on_epoch_begin(self, epoch, logs=None):
        self._maybe_init_progbar()
        if self.verbose > 2 and self.epochs > 1:
            print('Epoch %d/%d' % (epoch + 1, self.epochs))

    def on_train_batch_end(self, batch, logs=None):
        self._batch_update_progbar(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        self._batch_update_progbar(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        # Don't pass prediction results.
        self._batch_update_progbar(batch, None)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.verbose > 2:
            self._finalize_progbar(logs)
        elif self.verbose > 0:
            self.progbar.update(epoch + 1, logs)

    def on_test_end(self, logs=None):
        self._finalize_progbar(logs)

    def on_predict_end(self, logs=None):
        self._finalize_progbar(logs)

    def _reset_progbar(self):
        if self.verbose > 2:
            self.seen = 0
            self.progbar = None

    def _maybe_init_progbar(self):
        if self.progbar is None:
            self.progbar = Progbar(
                target=self.target,
                verbose=self.verbose - 2 if self.verbose > 2 else self.verbose)

    def _batch_update_progbar(self, batch, logs=None):
        """Updates the progbar."""
        logs = logs or {}
        self._maybe_init_progbar()
        self.seen = batch

        if self.verbose > 2:
            # Only block async when verbose = 1.
            self.progbar.update(self.seen, logs, finalize=False)

    def _finalize_progbar(self, logs):
        logs = logs or {}
        if self.target is None:
            self.progbar.target = self.target = self.seen
        self.progbar.update(self.target, logs, finalize=True)
        self._reset_progbar()

    def __str__(self) -> str:
        format_string = ""
        attrs = {'epochs', 'verbose'}
        for attr in attrs:
            format_string += f'\n  {attr}={getattr(self,attr)},'
        if format_string:
            # replace last ``,`` as ``\n``
            format_string = format_string[:-1] + '\n'
        return f"{self.__class__.__name__}({format_string})"

    __repr__ = __str__


class Scheduler(Callback):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def on_train_batch_end(self, batch, logs=None):
        self.scheduler.step()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(scheduler={self.scheduler})"

    __repr__ = __str__


class Optimizer(Callback):
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer

    def on_train_batch_begin(self, batch, logs=None):
        self.optimizer.zero_grad()

    def on_train_batch_end(self, batch, logs=None):
        self.optimizer.step()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(optimizer=" +\
            f"{self.optimizer.__class__.__name__})"

    __repr__ = __str__
