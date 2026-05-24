"""Evaluate a saved model on the MNIST test set.

  python test_model.py --model mlp                # built-in tag
  python test_model.py --model cnn                # built-in tag
  python test_model.py --tag mlp_fair             # any save_tag under outputs/best_models/
  python test_model.py --tag cnn_sgd_ms

When --tag is given, the architecture is inferred from the tag prefix
(`mlp...` -> Model_MLP, `cnn...` -> Model_CNN).
"""
import os
import argparse
import mynn as nn
import numpy as np
from struct import unpack
import gzip

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['mlp', 'cnn'], default=None,
                    help='Built-in convenience tag.')
parser.add_argument('--tag', type=str, default=None,
                    help='Arbitrary save_tag under outputs/best_models/<tag>/best_model.pickle.')
parser.add_argument('--ckpt', type=str, default=None,
                    help='Explicit checkpoint path; overrides --model/--tag.')
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))

if args.ckpt is not None:
    ckpt = args.ckpt
    tag = os.path.basename(os.path.dirname(ckpt))
elif args.tag is not None:
    tag = args.tag
    ckpt = os.path.join(HERE, 'outputs', 'best_models', tag, 'best_model.pickle')
elif args.model is not None:
    tag = args.model
    ckpt = os.path.join(HERE, 'outputs', 'best_models', tag, 'best_model.pickle')
else:
    raise SystemExit('Provide one of --model / --tag / --ckpt')

# Infer architecture from tag prefix.
arch = 'mlp' if tag.lower().startswith('mlp') else 'cnn'

if arch == 'mlp':
    model = nn.models.Model_MLP()
else:
    model = nn.models.Model_CNN(build=False)
model.load_model(ckpt)

test_images_path = os.path.join(HERE, 'dataset', 'MNIST', 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(HERE, 'dataset', 'MNIST', 't10k-labels-idx1-ubyte.gz')

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs.astype(np.float32) / 255.0
if arch == 'cnn':
    test_imgs = test_imgs.reshape(-1, 1, 28, 28)

# batched forward to avoid memory spike for CNN
bs = 256
correct = 0
for s in range(0, test_imgs.shape[0], bs):
    logits = model(test_imgs[s:s+bs])
    pred = np.argmax(logits, axis=-1)
    correct += int((pred == test_labs[s:s+bs]).sum())
acc = correct / test_imgs.shape[0]
print(f'[{tag}] test accuracy: {acc:.4f}  ({correct}/{test_imgs.shape[0]})')
