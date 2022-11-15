import sys
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from greatx.training.callbacks import (Callback, CallbackList, Optimizer,
                                       Scheduler)
from greatx.utils import BunchDict, Progbar


class Trainer:
    """A simple trainer to train graph neural network models conveniently.

    Parameters
    ----------
    model : nn.Module
        the model used for training
    device : Union[str, torch.device], optional
        the device used for training, by default 'cpu'
    cfg : other keyword arguments, such as `lr` and `weight_decay`.

    Example
    -------
    >>> from greatx.training.trainer import Trainer
    >>> model = ... # your model
    >>> trainer = Trainer(model, device='cuda')

    >>> data # PyG-like data, e.g., Cora
    Data(x=[2485, 1433], edge_index=[2, 10138], y=[2485])

    >>> # simple training
    >>> trainer.fit(data, data.train_mask)

    >>> # train with model picking
    >>> from greatx.training import ModelCheckpoint
    >>> cb = ModelCheckpoint('my_ckpt', monitor='val_acc')
    >>> trainer.fit(data, mask=(data.train_mask,
    ...                         data.val_mask), callbacks=[cb])

    >>> # get training logs
    >>> history = trainer.model.history

    >>> trainer.evaluate(data, your_test_mask) # evaluation

    >>> predict = trainer.predict(data, your_mask) # prediction
    """
    def __init__(self, model: nn.Module,
                 device: Union[str, torch.device] = 'cpu', **cfg):
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.cfg = BunchDict(cfg)

        # if cfg:
        #     print("Received extra configuration:\n" + str(self.cfg))

        self.cfg.setdefault("lr", 1e-2)
        self.cfg.setdefault("weight_decay", 5e-4)
        self.cfg.setdefault("empty_cache", False)

        self.optimizer = self.config_optimizer()
        self.scheduler = self.config_scheduler(self.optimizer)

    def fit(self, data: Union[Data, Tuple[Data, Data]],
            mask: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
            callbacks: Optional[Callback] = None, verbose: Optional[int] = 1,
            epochs: int = 100) -> "Trainer":
        """Simple training method designed for `:attr:model`

        Parameters
        ----------
        data : Union[Data, Tuple[Data, Data]]
            An instance or a tuple of
            :class:`torch_geometric.data.Data` denoting the graph.
            They are used for `train_step` and `val_step`, respectively.
        mask : Optional[Union[Tensor, Tuple[Tensor, Tensor]]]
            node masks used for training and validation.
        callbacks : Optional[Callback], optional
            callbacks used for training,
            see `greatx.training.callbacks`, by default None
        verbose : Optional[int], optional
            verbosity during training, can be:
            :obj:`None, 1, 2, 3, 4`, by default 1
        epochs : int, optional
            training epochs, by default 100

        Example
        -------
        >>> # simple training
        >>> trainer.fit(data, data.train_mask)

        >>> # train with model picking
        >>> from greatx.training import ModelCheckpoint
        >>> cb = ModelCheckpoint('my_ckpt', monitor='val_acc')
        >>> trainer.fit(data, mask=(data.train_mask,
        ...                         data.val_mask), callbacks=[cb])
        """

        empty_cache = self.cfg['empty_cache']
        model = self.model.to(self.device)
        model.stop_training = False

        validation = isinstance(data, tuple) or isinstance(mask, tuple)
        if isinstance(data, tuple):
            train_data, val_data = data
        else:
            train_data = val_data = data

        if isinstance(mask, tuple):
            train_mask, val_mask = mask
        else:
            train_mask = val_mask = mask

        # Setup callbacks
        self.callbacks = callbacks = self.config_callbacks(
            verbose, epochs, callbacks=callbacks)

        logs = BunchDict()

        if verbose:
            print("Training...")

        callbacks.on_train_begin()
        try:
            for epoch in range(epochs):
                if empty_cache and self.device.type.startswith('cuda'):
                    torch.cuda.empty_cache()
                callbacks.on_epoch_begin(epoch)
                train_logs = self.train_step(train_data, train_mask)
                logs.update(train_logs)

                if validation:
                    val_logs = self.test_step(val_data, val_mask)
                    val_logs = {f'val_{k}': v for k, v in val_logs.items()}
                    logs.update(val_logs)

                callbacks.on_epoch_end(epoch, logs)

                if model.stop_training:
                    print(f"Early Stopping at Epoch {epoch}", file=sys.stderr)
                    break

        finally:
            callbacks.on_train_end()

        return self

    def train_step(self, data: Data, mask: Optional[Tensor] = None) -> dict:
        """One-step training on the inputs.

        Parameters
        ----------
        data : Data
            the training data.
        mask : Optional[Tensor]
            the mask of training nodes.

        Returns
        -------
        dict
            the output logs, including `loss` and `acc`, etc.
        """
        model = self.model
        self.callbacks.on_train_batch_begin(0)

        model.train()
        data = data.to(self.device)
        adj_t = getattr(data, 'adj_t', None)
        y = data.y.squeeze()

        if adj_t is None:
            out = model(data.x, data.edge_index, data.edge_weight)
        else:
            out = model(data.x, adj_t)

        if mask is not None:
            out = out[mask]
            y = y[mask]

        loss = F.cross_entropy(out, y)
        loss.backward()
        self.callbacks.on_train_batch_end(0)

        return dict(loss=loss.item(),
                    acc=out.argmax(-1).eq(y).float().mean().item())

    def evaluate(self, data: Data, mask: Optional[Tensor] = None,
                 verbose: Optional[int] = 1) -> BunchDict:
        """Simple evaluation step for `:attr:model`

        Parameters
        ----------
        data : Data
            the testing data used for :meth:`test_step`.
        mask : Optional[Tensor]
            the mask of testing nodes used for :meth:`test_step`.
        verbose : Optional[int], optional
            verbosity during evaluation, by default 1

        Returns
        -------
        BunchDict
            the dict-like output logs

        Example
        -------
        >>> trainer.evaluate(data, data.test_mask) # evaluation
        """
        if verbose:
            print("Evaluating...")

        self.model = self.model.to(self.device)

        progbar = Progbar(target=1, verbose=verbose)
        logs = BunchDict(**self.test_step(data, mask))
        progbar.update(1, logs)
        return logs

    @torch.no_grad()
    def test_step(self, data: Data, mask: Optional[Tensor] = None) -> dict:
        """One-step evaluation on the inputs.

        Parameters
        ----------
        data : Data
            the testing data.
        mask : Optional[Tensor]
            the mask of testing nodes.

        Returns
        -------
        dict
            the output logs, including `loss` and `acc`, etc.
        """
        model = self.model
        model.eval()
        data = data.to(self.device)
        adj_t = getattr(data, 'adj_t', None)
        y = data.y.squeeze()

        if adj_t is None:
            out = model(data.x, data.edge_index, data.edge_weight)
        else:
            out = model(data.x, adj_t)

        if mask is not None:
            out = out[mask]
            y = y[mask]

        loss = F.cross_entropy(out, y)

        return dict(loss=loss.item(),
                    acc=out.argmax(-1).eq(y).float().mean().item())

    def predict_step(self, data: Data,
                     mask: Optional[Tensor] = None) -> Tensor:
        """One-step prediction on the inputs.

        Parameters
        ----------
        data : Data
            the prediction data.
        mask : Optional[Tensor]
            the mask of prediction nodes.

        Returns
        -------
        Tensor
            the output prediction.
        """
        model = self.model
        model.eval()
        data = data.to(self.device)
        adj_t = getattr(data, 'adj_t', None)

        if adj_t is None:
            out = model(data.x, data.edge_index, data.edge_weight)
        else:
            out = model(data.x, adj_t)

        if mask is not None:
            out = out[mask]
        return out

    @torch.no_grad()
    def predict(
        self, data: Data, mask: Optional[Tensor] = None,
        transform: Callable = torch.nn.Softmax(dim=-1)
    ) -> Tensor:
        """
        Parameters
        ----------
        data : Data
            the prediction data used for :meth:`predict_step`.
        mask : Optional[Tensor]
            the mask of prediction nodes used for :meth:`predict_step`.
        transform : Callable
            Callable function applied on output predictions.

        Example
        -------
        >>> predict = trainer.predict(data, mask) # prediction
        """

        self.model.to(self.device)
        out = self.predict_step(data, mask).squeeze()
        if transform is not None:
            out = transform(out)
        return out

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.01)
        weight_decay = self.cfg.get('weight_decay', 5e-4)
        return torch.optim.Adam(self.model.parameters(), lr=lr,
                                weight_decay=weight_decay)

    def config_scheduler(self, optimizer: torch.optim.Optimizer):
        return None

    def config_callbacks(self, verbose, epochs,
                         callbacks=None) -> CallbackList:
        callbacks = CallbackList(callbacks=callbacks, add_history=True,
                                 add_progbar=True if verbose else False)
        if self.optimizer is not None:
            callbacks.append(Optimizer(self.optimizer))
        if self.scheduler is not None:
            callbacks.append(Scheduler(self.scheduler))
        callbacks.set_model(self.model)
        callbacks.set_params(dict(verbose=verbose, epochs=epochs))
        return callbacks

    @property
    def model(self) -> Optional[torch.nn.Module]:
        return self._model

    @model.setter
    def model(self, m: Optional[torch.nn.Module]):
        assert m is None or isinstance(m, torch.nn.Module)
        self._model = m

    def cache_clear(self) -> "Trainer":
        """Clear cached inputs or intermediate results
        of the model."""
        if hasattr(self.model, 'cache_clear'):
            self.model.cache_clear()
        return self

    def __repr__(self) -> str:
        name = self.model.__class__.__name__
        return f"{self.__class__.__name__}(model={name}{self.extra_repr()})"

    __str__ = __repr__

    def extra_repr(self) -> str:
        string = ""
        blank = ' ' * (len(self.__class__.__name__) + 1)
        for k, v in self.cfg.items():
            if v is None:
                continue
            string += f",\n{blank}{k}={v}"
        return string
