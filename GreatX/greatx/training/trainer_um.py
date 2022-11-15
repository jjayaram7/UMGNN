import sys
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch.distributions import Categorical


from greatx.training.callbacks import (Callback, CallbackList, Optimizer,
                                       Scheduler)
from greatx.utils import BunchDict, Progbar


class TrainerUM:
    
    def __init__(self, Gmodel: nn.Module, Hmodel: nn.Module,
                 device: Union[str, torch.device] = 'cpu', **cfg):
        self.device = torch.device(device)
        self.Gmodel = Gmodel.to(self.device)
        self.Hmodel = Hmodel.to(self.device)

        self.cfg = BunchDict(cfg)

        # if cfg:
        #     print("Received extra configuration:\n" + str(self.cfg))

        self.cfg.setdefault("lr", 1e-2)
        self.cfg.setdefault("weight_decay", 5e-4)
        self.cfg.setdefault("empty_cache", False)

        self.optimizer = self.config_optimizer()
        self.scheduler = self.config_scheduler(self.optimizer)

    def fit(self, data: Union[Data, Tuple[Data, Data, Data]],
            mask: Optional[Union[Tensor, Tuple[Tensor, Tensor, Tensor]]] = None,
            callbacks: Optional[Callback] = None, verbose: Optional[int] = 1,
            epochs: int = 100) -> "Trainer":

        empty_cache = self.cfg['empty_cache']
        Gmodel = self.Gmodel.to(self.device)
        Gmodel.stop_training = False
        Hmodel = self.Hmodel.to(self.device)
        Hmodel.stop_training = False

        validation = isinstance(data, tuple) or isinstance(mask, tuple)
        if isinstance(data, tuple):
            train_data, val_data, test_data = data
        else:
            train_data = val_data = test_data = data

        if isinstance(mask, tuple):
            train_mask, val_mask, test_mask = mask
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
                train_logs = self.train_step(train_data, train_mask, torch.cat([train_mask, val_mask]))
                logs.update(train_logs)

                if validation:
                    Hval_logs = self.test_step(test_data, test_mask)
                    Hval_logs = {f'Hval_{k}': v for k, v in Hval_logs.items()}
                    logs.update(Hval_logs)

                    Gval_logs = self.test_step(test_data, test_mask, "G")
                    Gval_logs = {f'Gval_{k}': v for k, v in Gval_logs.items()}
                    logs.update(Gval_logs)

                callbacks.on_epoch_end(epoch, logs)

                if Gmodel.stop_training or Hmodel.stop_training:
                    print(f"Early Stopping at Epoch {epoch}", file=sys.stderr)
                    break

        finally:
            callbacks.on_train_end()

        return self

    def train_step(self, data: Data, mask: Optional[Tensor] = None, mask_: Optional[Tensor] = None) -> dict:
        Gmodel = self.Gmodel
        Hmodel = self.Hmodel
        self.callbacks.on_train_batch_begin(0)

        Gmodel.train()
        Hmodel.train()

        data = data.to(self.device)
        adj_t = getattr(data, 'adj_t', None)
        y = data.y.squeeze()

        nr = 10
        if adj_t is None:
            glist = []
            for t in range(nr):
                outputs = Gmodel(data.x, data.edge_index, data.edge_weight)
                glist.append(outputs)
            out1 = torch.mean(torch.stack(glist),dim=0)
            out2 = Hmodel(data.x)
        else:
            glist = []
            for t in range(nr):
                outputs = Gmodel(data.x, adj_t)
                glist.append(outputs)
            out1 = torch.mean(torch.stack(glist),dim=0)
            out2 = Hmodel(data.x)

        # if mask is not None:
        #     out1 = out1[mask]
        #     out2 = out2[mask]
        #     y = y[mask]

        p_out1 = F.softmax(out1, 1)
        p_out2 = F.softmax(out2, 1)
        ent = Categorical(p_out1).entropy()
        inv_ent = torch.log(1/(1+ent))

        logp_out1 = F.log_softmax(out1, 1) # Predictions from neighborhood
        loss1 = F.nll_loss(logp_out1[mask], y[mask]) # G predictions are matched against GT labels
        
        term1 = torch.sum(F.kl_div(p_out2[mask_].log(), p_out1[mask_],reduction='none'),dim=1)
        target_uni = torch.ones_like(p_out1[mask_])/out1[mask_].shape[1]
        term2 =1e-4*torch.sum(F.kl_div(p_out2[mask_].log(), target_uni,reduction='none'),dim=1)
        loss2 = torch.sum(F.softmax(inv_ent[mask_],dim=0)*term1)+torch.mean(term2)

        loss = 3.0*loss1 + 5.0*loss2
        loss.backward()
        self.callbacks.on_train_batch_end(0)

        return dict(loss=loss.item(),
                    acc=out1.argmax(-1).eq(y).float().mean().item())

    def evaluate(self, data: Data, mask: Optional[Tensor] = None, mode = 'H',
                 verbose: Optional[int] = 1) -> BunchDict:
        if verbose:
            print("Evaluating...")
        if mode == 'H':
            self.Hmodel = self.Hmodel.to(self.device)
        else:
            self.Gmodel = self.Gmodel.to(self.device)

        progbar = Progbar(target=1, verbose=verbose)
        logs = BunchDict(**self.test_step(data, mask, mode))
        progbar.update(1, logs)
        return logs

    @torch.no_grad()
    def test_step(self, data: Data, mask: Optional[Tensor] = None, mode = 'H') -> dict:
        data = data.to(self.device)
        adj_t = getattr(data, 'adj_t', None)
        y = data.y.squeeze()

        if mode == 'H':
            model = self.Hmodel
            model.eval()
            out = model(data.x)
        else:
            model = self.Gmodel
            model.eval()
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
                     mask: Optional[Tensor] = None, mode = 'H') -> Tensor:
        data = data.to(self.device)
        adj_t = getattr(data, 'adj_t', None)
        
        if mode == 'H':
            model = self.Hmodel
            model.eval()
            out = model(data.x)
        else:
            model = self.Gmodel
            model.eval()
            if adj_t is None:
                out = model(data.x, data.edge_index, data.edge_weight)
            else:
                out = model(data.x, adj_t)

        if mask is not None:
            out = out[mask]
        return out

    @torch.no_grad()
    def predict(
        self, data: Data, mask: Optional[Tensor] = None, mode = 'H',
        transform: Callable = torch.nn.Softmax(dim=-1)
    ) -> Tensor:
        if mode == 'H':
            self.Hmodel.to(self.device)
        else:
            self.Gmodel.to(self.device)
        out = self.predict_step(data, mask, mode).squeeze()
        if transform is not None:
            out = transform(out)
        return out

    def config_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.get('lr', 0.01)
        weight_decay = self.cfg.get('weight_decay', 5e-4)
        return torch.optim.Adam(list(self.Gmodel.parameters())+list(self.Hmodel.parameters()), lr=lr,
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
        callbacks.set_model(self.Hmodel)
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
        name = self.Hmodel.__class__.__name__
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
