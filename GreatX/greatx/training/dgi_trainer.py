from typing import Optional

import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from greatx.training.trainer import Trainer

bce = F.binary_cross_entropy_with_logits


class DGITrainer(Trainer):
    """Custom trainer for :class:`~greatx.nn.models.unsupervised.DGI`

    Parameters
    ----------
    model : nn.Module
        the model used for training
    device : Union[str, torch.device], optional
        the device used for training, by default 'cpu'
    cfg : other keyword arguments, such as `lr` and `weight_decay`.
    """
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

        if adj_t is None:
            postive, negative = model(data.x, data.edge_index,
                                      data.edge_weight)
        else:
            postive, negative = model(data.x, adj_t)

        loss = bce(postive, postive.new_ones(postive.size(0))) + \
            bce(negative, negative.new_zeros(negative.size(0)))

        loss.backward()
        self.callbacks.on_train_batch_end(0)
        return dict(loss=loss.item())

    def test(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError
