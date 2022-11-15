import os.path as osp

import torch
import torch_geometric.transforms as T

from greatx.datasets import GraphDataset
from greatx.defense import FeaturePropagation
from greatx.nn.models import GCN
from greatx.training.callbacks import ModelCheckpoint
from greatx.training.trainer import Trainer
from greatx.utils import MissingFeature, split_nodes

dataset = 'Cora'
root = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data')
dataset = GraphDataset(
    root=root,
    name=dataset,
    transform=T.Compose([
        T.LargestConnectedComponents(),
        # here we generate 50% missing features
        MissingFeature(missing_rate=0.5),
    ]))

data = dataset[0]
data = FeaturePropagation(missing_mask=data.missing_mask)(data)
splits = split_nodes(data.y, random_state=15)

num_features = data.x.size(-1)
num_classes = data.y.max().item() + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_features, num_classes)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit(data, mask=(splits.train_nodes, splits.val_nodes), callbacks=[ckp])
trainer.evaluate(data, splits.test_nodes)
