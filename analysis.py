"""Part C direction 5: error analysis and visualization.

Generates under outputs/figs/:
  - cm_<tag>.png       confusion matrix on the test set
  - wrong_<tag>.png    grid of misclassified test samples
  - weights_<tag>.png  (MLP only) first-layer weight templates
  - kernels_<tag>.png  (CNN only) first conv layer kernels

Usage:
  python analysis.py --tag mlp_fair       # any save_tag under outputs/best_models/
  python analysis.py --tag cnn

The architecture is inferred from the tag prefix: mlp* -> Model_MLP, cnn* -> Model_CNN.
"""
import os
import argparse
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt

import mynn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default=None,
                    help='Arbitrary save_tag under outputs/best_models/<tag>/best_model.pickle.')
parser.add_argument('--model', choices=['mlp', 'cnn'], default=None,
                    help='Legacy alias for --tag (kept for backward compat).')
parser.add_argument('--ckpt', type=str, default=None)
args = parser.parse_args()

tag = args.tag or args.model
if tag is None:
    raise SystemExit('Provide --tag (preferred) or --model (legacy).')

# Infer architecture from tag prefix.
arch = 'mlp' if tag.lower().startswith('mlp') else 'cnn'

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'outputs')
FIG_DIR = os.path.join(OUT, 'figs')
os.makedirs(FIG_DIR, exist_ok=True)
ckpt = args.ckpt or os.path.join(OUT, 'best_models', tag, 'best_model.pickle')

if arch == 'mlp':
    model = nn.models.Model_MLP()
else:
    model = nn.models.Model_CNN(build=False)
model.load_model(ckpt)

test_images_path = os.path.join(HERE, 'dataset', 'MNIST', 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(HERE, 'dataset', 'MNIST', 't10k-labels-idx1-ubyte.gz')

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs_raw = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28, 28)
with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs_raw.astype(np.float32) / 255.0
if arch == 'mlp':
    inputs = test_imgs.reshape(num, -1)
else:
    inputs = test_imgs.reshape(num, 1, 28, 28)

# batched inference
bs = 256
preds = np.zeros(num, dtype=np.int64)
for s in range(0, num, bs):
    logits = model(inputs[s:s+bs])
    preds[s:s+bs] = np.argmax(logits, axis=-1)

acc = (preds == test_labs).mean()
print(f'[{tag}] test acc = {acc:.4f}')

# --------- confusion matrix ----------
K = 10
cm = np.zeros((K, K), dtype=np.int64)
for t, p in zip(test_labs, preds):
    cm[t, p] += 1

fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(K)); ax.set_yticks(range(K))
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Confusion matrix ({tag}) acc={acc:.4f}')
for i in range(K):
    for j in range(K):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=7)
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f'cm_{tag}.png'), dpi=130)
plt.close(fig)

# --------- misclassified examples ----------
wrong_idx = np.where(preds != test_labs)[0]
print(f'  total wrong: {len(wrong_idx)}')
np.random.seed(0)
pick = np.random.choice(wrong_idx, size=min(25, len(wrong_idx)), replace=False)
fig, axes = plt.subplots(5, 5, figsize=(8, 8))
for ax, i in zip(axes.flatten(), pick):
    ax.imshow(test_imgs_raw[i], cmap='gray')
    ax.set_title(f'T:{test_labs[i]}  P:{preds[i]}', fontsize=8)
    ax.axis('off')
fig.suptitle(f'Misclassified examples ({tag})')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f'wrong_{tag}.png'), dpi=130)
plt.close(fig)

# --------- weight / kernel visualization ----------
if arch == 'mlp':
    # first-layer weights: [784, hidden] -> each column is a 28x28 "template"
    W = model.layers[0].params['W']
    n_show = min(64, W.shape[1])
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for k, ax in enumerate(axes.flatten()):
        if k < n_show:
            ax.imshow(W[:, k].reshape(28, 28), cmap='seismic')
        ax.axis('off')
    fig.suptitle(f'{tag} first-layer weight templates (first 64 hidden units)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'weights_{tag}.png'), dpi=130)
    plt.close(fig)
else:
    # first conv kernels: [out_C, in_C=1, k, k]
    W = model.layers[0].params['W'][:, 0]
    out_C = W.shape[0]
    cols = 4
    rows = (out_C + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    for k, ax in enumerate(axes.flatten()):
        if k < out_C:
            ax.imshow(W[k], cmap='seismic')
            ax.set_title(f'k{k}', fontsize=8)
        ax.axis('off')
    fig.suptitle(f'{tag} first conv-layer kernels')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'kernels_{tag}.png'), dpi=130)
    plt.close(fig)

print(f'Saved analysis artefacts under {FIG_DIR}/.')
