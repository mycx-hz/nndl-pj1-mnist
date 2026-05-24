# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import os
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'outputs')
os.makedirs(os.path.join(OUT, 'history'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'figs'), exist_ok=True)
train_images_path = os.path.join(HERE, 'dataset', 'MNIST', 'train-images-idx3-ubyte.gz')
train_labels_path = os.path.join(HERE, 'dataset', 'MNIST', 'train-labels-idx1-ubyte.gz')

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open(os.path.join(OUT, 'idx.pickle'), 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs.astype(np.float32) / 255.0
valid_imgs = valid_imgs.astype(np.float32) / 255.0

linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, batch_size=32, scheduler=scheduler)

save_dir = os.path.join(OUT, 'best_models', 'mlp')
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs],
             num_epochs=5, log_iters=100, eval_iters=100, save_dir=save_dir)

# Save runner history for later plotting / report.
with open(os.path.join(OUT, 'history', 'history_mlp.pickle'), 'wb') as f:
        pickle.dump({
                'train_loss': runner.train_loss,
                'train_scores': runner.train_scores,
                'dev_loss': runner.dev_loss,
                'dev_scores': runner.dev_scores,
                'best_score': runner.best_score,
        }, f)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.set_tight_layout(1)
plot(runner, axes)
fig.savefig(os.path.join(OUT, 'figs', 'curve_mlp.png'), dpi=120)
print('MLP best dev acc:', runner.best_score)
