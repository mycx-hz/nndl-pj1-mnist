import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        eval_iters = kwargs.get("eval_iters", log_iters)
        save_dir = kwargs.get("save_dir", "best_model")

        os.makedirs(save_dir, exist_ok=True)

        best_score = 0
        last_dev_score = 0.0
        last_dev_loss = 0.0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            n_iter = X.shape[0] // self.batch_size
            for iteration in range(n_iter):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)

                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                if iteration % eval_iters == 0:
                    dev_score, dev_loss = self.evaluate(dev_set)
                    last_dev_score = dev_score
                    last_dev_loss = dev_loss
                    if dev_score > best_score:
                        save_path = os.path.join(save_dir, 'best_model.pickle')
                        self.save_model(save_path)
                        print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                        best_score = dev_score
                self.dev_scores.append(last_dev_score)
                self.dev_loss.append(last_dev_loss)

                if iteration % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {last_dev_loss}, score: {last_dev_score}")

        self.best_score = best_score

    def evaluate(self, data_set, batch_size=None):
        X, y = data_set
        # batched eval to avoid huge memory spikes (CNN)
        bs = batch_size if batch_size is not None else max(self.batch_size, 256)
        n = X.shape[0]
        total_correct = 0
        total_loss = 0.0
        total_samples = 0
        for start in range(0, n, bs):
            xb = X[start:start+bs]
            yb = y[start:start+bs]
            logits = self.model(xb)
            loss = self.loss_fn(logits, yb)
            score = self.metric(logits, yb)
            total_loss += float(loss) * xb.shape[0]
            total_correct += float(score) * xb.shape[0]
            total_samples += xb.shape[0]
        return total_correct / total_samples, total_loss / total_samples

    def save_model(self, save_path):
        self.model.save_model(save_path)
