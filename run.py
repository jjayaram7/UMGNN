import torch
import torch_geometric.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from greatx.datasets import GraphDataset
from greatx.utils import split_nodes
from argparse import ArgumentParser
from greatx.attack.untargeted import RandomAttack, DICEAttack, MinmaxAttack, PGDAttack, Metattack, FGAttack
from greatx.nn.models import GCN, GAT
from greatx.training.trainer import Trainer
from modeldefs import FCN
from greatx.training.trainer_um import TrainerUM 

device = 'cuda'

parser = ArgumentParser()
parser.add_argument("--dname", default='citeseer', help='dataset name')
parser.add_argument("--model", default='GCN', help='GCN, GAT')
parser.add_argument("--attack", default='meta', help='type of attack (dice, minmax, pgd, meta, fga')
parser.add_argument("--noise", default=0.5, type = float, help='level of noise')
parser.add_argument("--lr", default=0.005, type = float, help='learning rate')
parser.add_argument("--epochs", default=100, type = int, help='epcohs')
parser.add_argument("--seed", default=14, type = int, help='seed')

args = parser.parse_args()

root = 'data'
dataset = GraphDataset(root=root, name=args.dname,
                       transform=T.LargestConnectedComponents())
data = dataset[0]
splits = split_nodes(data.y, random_state=args.seed)
num_features = data.x.size(-1)
num_classes = data.y.max().item() + 1
print(data)

## Train vanilla model
surr = Trainer(GCN(num_features, num_classes, hids = [32],dropout=0.5),lr = args.lr, device=device)

surr.fit(data = data, mask=(splits.train_nodes, splits.val_nodes), epochs = 50)
logs = surr.evaluate(data, splits.test_nodes,verbose=0)
print(f"Done training a GCN on Clean Graph! (Acc = {logs.acc})")

if args.attack == 'random':
    attacker = RandomAttack(data, device=device)
elif args.attack == 'dice':
    attacker = DICEAttack(data, device=device)
elif args.attack == 'minmax':
    attacker = MinmaxAttack(data, device=device)
    attacker.setup_surrogate(surr.model,
                         labeled_nodes=splits.train_nodes)
elif args.attack == 'pgd':
    attacker = PGDAttack(data, device=device)
    attacker.setup_surrogate(surr.model,
                         labeled_nodes=splits.train_nodes)
elif args.attack == 'meta':
    attacker = Metattack(data, device=device)
    attacker.setup_surrogate(surr.model,
                         labeled_nodes=splits.train_nodes,
                         unlabeled_nodes=splits.test_nodes, lambda_=0.)
elif args.attack == 'fga':
    attacker = FGAttack(data, device=device)
    attacker.setup_surrogate(surr.model, splits.train_nodes)

attacker.reset()
attacker.attack(args.noise)

## Train GNN on the poisoned graph
if args.model == 'GCN':
    trainer_basic = Trainer(GCN(num_features, num_classes, hids = [32],dropout=0.5),lr = args.lr, device=device)
elif args.model == 'GAT':
    trainer_basic = Trainer(GAT(num_features, num_classes, hids = [32],dropout=0.5),lr = args.lr, device=device)
else:
    print('Not implemented!!')
trainer_basic.fit(data = attacker.data(), mask=(splits.train_nodes, splits.val_nodes), epochs = 50,verbose=1)
logs = trainer_basic.evaluate(attacker.data(), splits.test_nodes,verbose=0)
print(f'Done training Vanilla GNN on Poisoned Graph (Acc = {logs.acc})')
acc_basic = logs.acc

## Train UMGNN on the poisoned graph
if args.model == 'GCN':
    Gmodel = GCN(num_features, num_classes, hids = [32,32],dropout=0.7)
elif args.model == 'GAT':
    Gmodel = GAT(num_features, num_classes, hids = [32],dropout=0.8)
else:
    print('Not implemented!!')
Hmodel = FCN(in_feats=num_features, hidden_neurons=[32,32], out_feats=num_classes)
trainer_umgnn = TrainerUM(Gmodel, Hmodel, lr = args.lr)
trainer_umgnn.fit(data = attacker.data(), mask=(splits.train_nodes, splits.val_nodes, splits.test_nodes), epochs = args.epochs)
logs = trainer_umgnn.evaluate(attacker.data(), splits.test_nodes, mode = 'G')
acc_g = logs.acc
logs = trainer_umgnn.evaluate(attacker.data(), splits.test_nodes, mode = 'H')
acc_h = logs.acc
logs.pop('loss'); logs.pop('acc')
logs.update({'acc (GNN)': acc_basic, 'acc (UMGNN-G)': acc_g, 'acc (UMGNN-H)': acc_h})
print(logs)

import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'serif',
        'size'   : 18}
matplotlib.rc('font', **font)
Hval_acc = trainer_umgnn.callbacks._history.history.Hval_acc
Gval_acc = trainer_umgnn.callbacks._history.history.Gval_acc
plt.figure(figsize=(6,5))
plt.axhline(acc_basic , linestyle ='--', color='k', linewidth='1.5',label='Vanilla GNN')
plt.plot(Gval_acc[1:] , linestyle ='-', color='darkgreen', linewidth='1.5',label='Teacher (GNN)')
plt.plot(Hval_acc[1:] , linestyle ='-', color='m', linewidth='1.5',label='Student (FCN)')
plt.xlabel('Iteration'); plt.ylabel('Test Accuracy')
plt.title(f'Noise Level = {args.noise}')
plt.legend()
plt.title(f'{args.dname}: model={args.model}, attack={args.attack}, level={args.noise}')
plt.savefig(f'./results/{args.dname}_{args.model}_{args.attack}_{args.noise}.png',bbox_inches='tight')