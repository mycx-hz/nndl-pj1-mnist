"""Train the simple CNN on MNIST using the same data split / preprocessing as test_train.py
so the MLP-vs-CNN comparison is fair."""
import os
import argparse
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--optimizer', choices=['sgd', 'momentum'], default='momentum')
parser.add_argument('--mu', type=float, default=0.9)
parser.add_argument('--scheduler', choices=['none', 'multistep'], default='multistep')
parser.add_argument('--save-tag', type=str, default='cnn')
parser.add_argument('--limit-train', type=int, default=0,
                    help='Use only this many training samples (0 = all). Useful for smoke tests.')
args = parser.parse_args()

np.random.seed(309)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'outputs')
os.makedirs(os.path.join(OUT, 'history'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'figs'), exist_ok=True)
train_images_path = os.path.join(HERE, 'dataset', 'MNIST', 'train-images-idx3-ubyte.gz')
train_labels_path = os.path.join(HERE, 'dataset', 'MNIST', 'train-labels-idx1-ubyte.gz')

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

idx_path = os.path.join(OUT, 'idx.pickle')
if os.path.exists(idx_path):
    with open(idx_path, 'rb') as f:
        idx = pickle.load(f)
else:
    idx = np.random.permutation(np.arange(num))
    with open(idx_path, 'wb') as f:
        pickle.dump(idx, f)

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

if args.limit_train > 0:
    train_imgs = train_imgs[:args.limit_train]
    train_labs = train_labs[:args.limit_train]

train_imgs = (train_imgs.astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)
valid_imgs = (valid_imgs.astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)

model = nn.models.Model_CNN(in_channels=1, num_classes=10,
                            conv_channels=(8, 16), hidden=64,
                            lambda_list=[1e-4, 1e-4, 1e-4, 1e-4])

if args.optimizer == 'sgd':
    optimizer = nn.optimizer.SGD(init_lr=args.lr, model=model)
else:
    optimizer = nn.optimizer.MomentGD(init_lr=args.lr, model=model, mu=args.mu)

if args.scheduler == 'multistep':
    # roughly: drop lr at the middle and 3/4 of training
    steps_per_epoch = train_imgs.shape[0] // args.batch_size
    total = steps_per_epoch * args.epochs
    milestones = [total // 2, total * 3 // 4]
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.5)
else:
    scheduler = None

loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=int(train_labs.max() + 1))

runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn,
                           batch_size=args.batch_size, scheduler=scheduler)

save_dir = os.path.join(OUT, 'best_models', args.save_tag)
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs],
             num_epochs=args.epochs, log_iters=50, eval_iters=100, save_dir=save_dir)

hist_path = os.path.join(OUT, 'history', f'history_{args.save_tag}.pickle')
with open(hist_path, 'wb') as f:
    pickle.dump({
        'train_loss': runner.train_loss,
        'train_scores': runner.train_scores,
        'dev_loss': runner.dev_loss,
        'dev_scores': runner.dev_scores,
        'best_score': runner.best_score,
        'args': vars(args),
    }, f)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.set_tight_layout(1)
plot(runner, axes)
fig.savefig(os.path.join(OUT, 'figs', f'curve_{args.save_tag}.png'), dpi=120)
print(f'[{args.save_tag}] best dev acc: {runner.best_score:.4f}')
