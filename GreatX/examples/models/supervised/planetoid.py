import os.path as osp

import torch
from torch_geometric.datasets import Planetoid

from greatx.nn.models import GCN
from greatx.training import Trainer
from greatx.training.callbacks import ModelCheckpoint

dataset = 'Cora'
root = osp.join(osp.dirname(osp.realpath(__file__)), '../../..', 'data')
dataset = Planetoid(root=root, name=dataset)

data = dataset[0]

num_features = data.x.size(-1)
num_classes = data.y.max().item() + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_features, num_classes)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit(data, mask=(data.train_mask, data.val_mask), callbacks=[ckp])
trainer.evaluate(data, data.test_mask)
