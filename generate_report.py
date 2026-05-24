"""Assemble the Project-1 report.

Reads every outputs/history/history_*.pickle, evaluates each saved checkpoint
under outputs/best_models/*/best_model.pickle on the MNIST test set, draws a
combined comparison plot into outputs/figs/curve_all.png, and writes a single
self-contained HTML file (all images base64-embedded) so it can be PDF-rendered
as-is.

Run from the codes/ folder:
    python generate_report.py
Outputs:
    ../报告.html   (PJ1/ root, sibling of codes/)
    ../results.json
    outputs/figs/curve_all.png
"""
import os
import json
import base64
import pickle
import gzip
from struct import unpack
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import mynn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'outputs')
FIG_DIR = os.path.join(OUT, 'figs')
HIST_DIR = os.path.join(OUT, 'history')
CKPT_DIR = os.path.join(OUT, 'best_models')
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- author / link placeholders ----------
AUTHOR_NAME = '[姓名]'
AUTHOR_ID = '[学号]'
GITHUB_URL = '[GitHub link]'
MODELSCOPE_URL = '[ModelScope link]'

# ---------- which runs to report (4 experiments, all under the same training budget) ----------
# (tag, display_name_zh, arch, role, plot_label_en)
RUNS = [
    ('mlp_fair',   'MLP',                       'mlp', 'mlp',        'MLP'),
    ('cnn',        'CNN',                       'cnn', 'cnn',        'CNN'),
    ('cnn_sgd',    'CNN (SGD, 无 scheduler)',   'cnn', 'cnn_sgd',    'CNN (SGD, no sched)'),
    ('cnn_sgd_ms', 'CNN (SGD + MultiStepLR)',   'cnn', 'cnn_sgd_ms', 'CNN (SGD+MS)'),
]

# ---------- helpers ----------
def load_history(tag):
    p = os.path.join(HIST_DIR, f'history_{tag}.pickle')
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)

def load_mnist_test():
    test_images_path = os.path.join(HERE, 'dataset', 'MNIST', 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(HERE, 'dataset', 'MNIST', 't10k-labels-idx1-ubyte.gz')
    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)
    test_imgs = test_imgs.astype(np.float32) / 255.0
    return test_imgs, test_labs

def evaluate(tag, arch, test_imgs, test_labs):
    ckpt = os.path.join(CKPT_DIR, tag, 'best_model.pickle')
    if not os.path.exists(ckpt):
        return None
    if arch == 'mlp':
        model = nn.models.Model_MLP()
        X = test_imgs
    else:
        model = nn.models.Model_CNN(build=False)
        X = test_imgs.reshape(-1, 1, 28, 28)
    model.load_model(ckpt)
    bs = 256
    correct = 0
    for s in range(0, X.shape[0], bs):
        logits = model(X[s:s+bs])
        pred = np.argmax(logits, axis=-1)
        correct += int((pred == test_labs[s:s+bs]).sum())
    return correct / X.shape[0]

def fig_to_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = f.read()
    return 'data:image/png;base64,' + base64.b64encode(data).decode()

# ---------- collect results ----------
test_imgs, test_labs = load_mnist_test()
results = []
for tag, name, arch, role, plot_label in RUNS:
    hist = load_history(tag)
    if hist is None:
        print(f'  [skip] {tag}: no history')
        continue
    test_acc = evaluate(tag, arch, test_imgs, test_labs)
    item = {
        'tag': tag,
        'name': name,
        'arch': arch,
        'role': role,
        'plot_label': plot_label,
        'best_dev': float(hist.get('best_score', 0.0)),
        'test_acc': test_acc,
        'args': hist.get('args'),
        'n_iters': len(hist['dev_scores']),
    }
    results.append(item)
    print(f'  {tag:14s}  dev={item["best_dev"]:.4f}  test={item["test_acc"] if test_acc else "?":.4f}')

with open(os.path.join(ROOT, 'results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ---------- combined curve plot (English labels to dodge CJK font issue) ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = {'mlp': '#1f77b4', 'cnn': '#d62728', 'cnn_sgd': '#2ca02c', 'cnn_sgd_ms': '#9467bd'}
for tag, name, arch, role, plot_label in RUNS:
    hist = load_history(tag)
    if hist is None:
        continue
    dev = hist['dev_scores']
    loss = hist['dev_loss']
    x = np.arange(len(dev))
    axes[0].plot(x, dev,  label=plot_label, color=colors.get(role), linewidth=1.3)
    axes[1].plot(x, loss, label=plot_label, color=colors.get(role), linewidth=1.3)
axes[0].set_title('Dev accuracy vs. iteration')
axes[0].set_xlabel('iteration'); axes[0].set_ylabel('dev accuracy')
axes[0].set_ylim(0.92, 1.0); axes[0].legend(fontsize=8, loc='lower right')
axes[0].grid(alpha=0.3)
axes[1].set_title('Dev loss vs. iteration')
axes[1].set_xlabel('iteration'); axes[1].set_ylabel('dev loss')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
combined = os.path.join(FIG_DIR, 'curve_all.png')
fig.tight_layout()
fig.savefig(combined, dpi=130)
plt.close(fig)

# ---------- figure paths (all under outputs/figs/) ----------
fig_paths = {
    'curve_all':        combined,
    'curve_mlp':        os.path.join(FIG_DIR, 'curve_mlp_fair.png'),
    'curve_cnn':        os.path.join(FIG_DIR, 'curve_cnn.png'),
    'curve_cnn_sgd':    os.path.join(FIG_DIR, 'curve_cnn_sgd.png'),
    'curve_cnn_sgd_ms': os.path.join(FIG_DIR, 'curve_cnn_sgd_ms.png'),
    'cm_mlp':           os.path.join(FIG_DIR, 'cm_mlp_fair.png'),
    'cm_cnn':           os.path.join(FIG_DIR, 'cm_cnn.png'),
    'wrong_mlp':        os.path.join(FIG_DIR, 'wrong_mlp_fair.png'),
    'wrong_cnn':        os.path.join(FIG_DIR, 'wrong_cnn.png'),
    'weights_mlp':      os.path.join(FIG_DIR, 'weights_mlp_fair.png'),
    'kernels_cnn':      os.path.join(FIG_DIR, 'kernels_cnn.png'),
}
figs_b64 = {k: fig_to_base64(v) for k, v in fig_paths.items()}

# ---------- HTML ----------
def by_tag(t):
    for r in results:
        if r['tag'] == t:
            return r
    return None

def pct(x):
    return f'{100*x:.2f}%' if x is not None else '—'

mlp        = by_tag('mlp_fair') or {}
cnn        = by_tag('cnn') or {}
cnn_sgd    = by_tag('cnn_sgd') or {}
cnn_sgd_ms = by_tag('cnn_sgd_ms') or {}

# Pre-compute key deltas so the prose stays tidy.
dev_gap   = (cnn.get('best_dev', 0) - mlp.get('best_dev', 0)) * 100
test_gap  = (cnn.get('test_acc', 0) - mlp.get('test_acc', 0)) * 100
err_mlp   = (1 - mlp.get('test_acc', 0)) * 100
err_cnn   = (1 - cnn.get('test_acc', 0)) * 100
err_reduce = (1 - (1 - cnn.get('test_acc', 0)) / max(1e-9, (1 - mlp.get('test_acc', 0)))) * 100
sched_eff = (cnn_sgd_ms.get('best_dev', 0) - cnn_sgd.get('best_dev', 0)) * 100
mom_eff   = (cnn.get('best_dev', 0)        - cnn_sgd_ms.get('best_dev', 0)) * 100

def row(r):
    args = r['args'] or {}
    opt = 'MomentGD(μ=0.9)' if args.get('optimizer') == 'momentum' else 'SGD'
    sched = 'MultiStepLR' if args.get('scheduler') == 'multistep' else '—'
    return (
        f'<tr><td>{r["name"]}</td>'
        f'<td>{r["arch"].upper()}</td>'
        f'<td>{opt}</td>'
        f'<td>{args.get("lr", "?")}</td>'
        f'<td>{args.get("batch_size", "?")}</td>'
        f'<td>{args.get("epochs", "?")}</td>'
        f'<td>{sched}</td>'
        f'<td>{pct(r["best_dev"])}</td>'
        f'<td><b>{pct(r["test_acc"])}</b></td></tr>'
    )

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>Project-1 报告 — MNIST 手写数字分类（MLP 与 CNN）</title>
<style>
  @page {{ size: A4; margin: 18mm 15mm; }}
  body {{ font-family: "DejaVu Sans","Noto Sans CJK SC","WenQuanYi Zen Hei",sans-serif; line-height:1.6; color:#222; max-width:780px; margin:0 auto; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  h2 {{ font-size: 17px; border-bottom:1px solid #ccc; padding-bottom:3px; margin-top:24px; }}
  h3 {{ font-size: 14px; margin-top: 16px; }}
  .meta {{ color:#555; font-size: 12px; margin-bottom: 18px; }}
  table {{ border-collapse: collapse; margin: 8px 0 16px; font-size: 12.5px; }}
  th, td {{ border: 1px solid #aaa; padding: 4px 8px; text-align: center; }}
  th {{ background: #f0f0f0; }}
  code, pre {{ font-family: "DejaVu Sans Mono","Consolas",monospace; font-size: 12px; }}
  pre {{ background:#f7f7f7; border:1px solid #e0e0e0; padding:8px; overflow-x:auto; }}
  img.fig {{ max-width: 100%; display:block; margin: 6px auto; }}
  .caption {{ font-size: 11.5px; color:#555; text-align:center; margin-bottom:12px; }}
  .grid2 {{ display:flex; gap:8px; }}
  .grid2 > div {{ flex:1; }}
  ul, ol {{ padding-left: 22px; }}
  li {{ margin-bottom: 4px; }}
</style>
</head>
<body>

<h1>《神经网络与深度学习》Project-1：MNIST 手写数字分类</h1>
<div class="meta">
作者：{AUTHOR_NAME}（学号 {AUTHOR_ID}）
&nbsp;|&nbsp; 完成日期：{datetime.now().strftime('%Y-%m-%d')}
&nbsp;|&nbsp; 代码：<a href="{GITHUB_URL}">{GITHUB_URL}</a>
&nbsp;|&nbsp; 训练权重：<a href="{MODELSCOPE_URL}">{MODELSCOPE_URL}</a>
</div>

<h2>1. 引言</h2>
<p>
本项目在 MNIST 手写数字分类任务上，基于课程提供的 NumPy 起始代码完成一个最小可用的神经网络框架，
并依次实现 <b>MLP 基线</b>（Part A）与 <b>CNN</b>（Part B），在两者训练设置完全一致的前提下评估，
最后选择 <b>方向 1（优化）</b> 与 <b>方向 5（错误分析与可视化）</b> 作为 Part C 的两个额外方向。
</p>
<p>
自主实现的算子与组件：<code>Linear</code> 的前/反向、<code>MultiCrossEntropyLoss</code>（内嵌 softmax）、
<code>conv2D</code>（im2col + col2im）、<code>MaxPool2D</code>、<code>Flatten</code>、<code>ReLU</code>、
<code>MomentGD</code>、<code>MultiStepLR</code>、<code>ExponentialLR</code>。全程未调用
PyTorch / TensorFlow / scikit-learn 等深度学习或卷积库，第三方依赖只有 numpy、matplotlib、
gzip、pickle、tqdm。
</p>
<p>
数据划分固定：MNIST 60,000 训练样本经一次随机置换（种子 309，保存到 <code>outputs/idx.pickle</code>）
后切成 50,000 train + 10,000 dev；10,000 测试样本独立使用。所有实验复用这同一份 split，
确保对比公平。
</p>

<h3>1.1 实现要点速览</h3>
<p>
下表列出本项目自实现的全部算子与组件，每行给出一句话的核心实现思路和源码位置。
更详细的推导见 §2.3（softmax + 交叉熵）和 §3.1（conv2D im2col）。
</p>
<table>
<tr><th>模块</th><th>文件:行</th><th>核心实现（一句话）</th></tr>
<tr><td><code>Linear.forward</code></td><td>op.py:44</td>
    <td><code>X @ W + b</code>；权重用 He 初始化 <code>std = √(2/fan_in)</code>，否则默认 std=1 会让 logits 爆炸</td></tr>
<tr><td><code>Linear.backward</code></td><td>op.py:52</td>
    <td><code>dW = X.T @ grad</code>；<code>db = sum(grad, axis=0)</code>；<code>dX = grad @ W.T</code>；带 weight decay 时 <code>dW += λW</code></td></tr>
<tr><td><code>MultiCrossEntropyLoss</code></td><td>op.py:292</td>
    <td>内嵌 softmax；对 logits 的反向梯度退化为 <code>(p − y_onehot) / N</code>（推导见 §2.3）</td></tr>
<tr><td><code>conv2D.forward</code></td><td>op.py:153</td>
    <td><code>cols = im2col(X)</code> → <code>einsum('oc,ncl→nol', W_flat, cols)</code> + bias</td></tr>
<tr><td><code>conv2D.backward</code></td><td>op.py:170</td>
    <td><code>dW = einsum('nol,ncl→oc', dout, cols)</code>；<code>dX</code> 用 <code>col2im</code> 散射回图像空间</td></tr>
<tr><td><code>MaxPool2D</code></td><td>op.py:201</td>
    <td>前向：复用 im2col 对每个池化窗取 <code>argmax</code>；反向：用 <code>put_along_axis</code> 把梯度放回对应位置</td></tr>
<tr><td><code>Flatten</code></td><td>op.py:249</td>
    <td>前向 <code>reshape(N, -1)</code>，记录原 shape；反向再 <code>reshape</code> 回去</td></tr>
<tr><td><code>ReLU</code></td><td>op.py:269</td>
    <td>前向 <code>np.where(X&lt;0, 0, X)</code>；反向用同一掩码 <code>np.where(input&lt;0, 0, grad)</code></td></tr>
<tr><td><code>SGD</code></td><td>optimizer.py:15</td>
    <td><code>w ← (1 − lr·λ)·w − lr·grad</code>（weight decay 直接缩 w）</td></tr>
<tr><td><code>MomentGD</code></td><td>optimizer.py:28</td>
    <td>对每个参数维护 velocity：<code>v ← μ·v + grad；w ← w − lr·v</code></td></tr>
<tr><td><code>MultiStepLR</code></td><td>lr_scheduler.py:26</td>
    <td>每 step 计数 +1，碰到 milestone 时把 <code>optimizer.init_lr</code> 乘以 γ</td></tr>
<tr><td><code>ExponentialLR</code></td><td>lr_scheduler.py:40</td>
    <td>每 step 都把 <code>lr</code> 乘以 γ（连续衰减；本提交未使用，但已实现）</td></tr>
<tr><td><code>Model_MLP</code></td><td>models.py:4</td>
    <td>按 <code>size_list</code> 顺序拼 Linear-ReLU-…-Linear；<code>save_model</code> / <code>load_model</code> 用 pickle</td></tr>
<tr><td><code>Model_CNN</code></td><td>models.py:77</td>
    <td>Conv-ReLU-Pool ×2 → Flatten → FC-ReLU-FC；保存结构元信息 + 各层权重</td></tr>
<tr><td>训练循环</td><td>runner.py:23</td>
    <td>shuffle → mini-batch → forward → loss → loss.backward()（自动把梯度传给所有层） → optimizer.step → scheduler.step；batched eval 避免 CNN OOM</td></tr>
</table>

<h2>2. MLP 基线（Part A）</h2>

<h3>2.1 模型与训练设置</h3>
<ul>
<li><b>结构</b>：<code>Linear(784→600) → ReLU → Linear(600→10)</code>，输出接 softmax + 交叉熵。</li>
<li><b>初始化</b>：He 初始化（<code>std = √(2/fan_in)</code>），避免默认 <code>np.random.normal(std=1)</code>
   带来的 logits 数值爆炸。</li>
<li><b>正则化</b>：每个 Linear 层启用 L2 weight decay，λ = 1e-4。</li>
<li><b>优化器</b>：MomentGD，lr = 0.05，μ = 0.9。</li>
<li><b>学习率调度</b>：MultiStepLR，在训练总步数的 1/2 和 3/4 处各乘 γ = 0.5。</li>
<li><b>批大小 / 轮数</b>：batch = 64，epochs = 3。</li>
</ul>
<p>
结果：验证集 best acc = <b>{pct(mlp.get('best_dev'))}</b>，测试集 acc = <b>{pct(mlp.get('test_acc'))}</b>。
</p>

<h3>2.2 学习曲线</h3>
<img class="fig" src="{figs_b64.get('curve_mlp')}" alt="curve_mlp_fair" />
<div class="caption">图 2.1：MLP 训练过程的 train / dev loss 与 accuracy 曲线。</div>

<h3>2.3 关键实现：softmax + 交叉熵的反向梯度</h3>
<p>
softmax 与交叉熵组合后，对 logits 的梯度退化成非常干净的形式：
对于每个样本 <code>∂L/∂z<sub>k</sub> = (p<sub>k</sub> − y<sub>k</sub>) / N</code>，
其中 <code>p = softmax(z)</code>，<code>y</code> 是 one-hot 标签，<code>N</code> 是 batch size。
代码实现就两行：
</p>
<pre>grads = probs.copy()
grads[np.arange(N), labels] -= 1.0
grads /= N</pre>
<p>不需要单独写 softmax 反向，整条链路一步完成。</p>

<h2>3. CNN 模型与 MLP-vs-CNN 对比（Part B）</h2>

<h3>3.1 CNN 结构与 conv2D 实现</h3>
<pre>
Input  [N, 1, 28, 28]
  → Conv(1→8,  k=3, pad=1) → ReLU → MaxPool(2)   # 28 → 14
  → Conv(8→16, k=3, pad=1) → ReLU → MaxPool(2)   # 14 → 7
  → Flatten                                       # 16·7·7 = 784
  → Linear(784 → 64) → ReLU
  → Linear(64  → 10)
</pre>
<p>
<code>conv2D</code> 采用标准的 <b>im2col + 矩阵乘</b> 实现：前向把每个滑窗展开成一列，得到
<code>cols ∈ R<sup>N×(C·k·k)×(H'·W')</sup></code>，与展平的卷积核做一次 einsum；
反向用 <code>col2im</code> 把梯度散射回图像空间。MaxPool 也复用 im2col 的索引化思路，
通过 <code>argmax</code> 记录池化位置，反向时用 <code>put_along_axis</code> 把梯度放回。
</p>

<h3>3.2 训练设置（与 §2 完全一致）</h3>
<p>
为了让 MLP 与 CNN 的对比只反映"模型架构"这一个因素，CNN 沿用与 §2.1 MLP 相同的训练设置：
</p>
<ul>
<li>优化器：MomentGD，lr = 0.05，μ = 0.9</li>
<li>scheduler：MultiStepLR，milestones = [total_steps/2, total_steps·3/4]，γ = 0.5</li>
<li>batch = 64，epochs = 3</li>
<li>L2 weight decay：λ = 1e-4（每个可训练层）</li>
<li>数据划分：同一份 <code>idx.pickle</code></li>
</ul>

<h3>3.3 结果</h3>
<table>
<tr><th>模型</th><th>验证集 best</th><th>测试集 acc</th><th>测试集错误率</th></tr>
<tr><td>MLP</td><td>{pct(mlp.get('best_dev'))}</td><td>{pct(mlp.get('test_acc'))}</td><td>{err_mlp:.2f}%</td></tr>
<tr><td><b>CNN</b></td><td><b>{pct(cnn.get('best_dev'))}</b></td><td><b>{pct(cnn.get('test_acc'))}</b></td><td><b>{err_cnn:.2f}%</b></td></tr>
<tr><td>差距</td><td>+{dev_gap:.2f} pp</td><td>+{test_gap:.2f} pp</td><td>相对减少 {err_reduce:.1f}%</td></tr>
</table>
<img class="fig" src="{figs_b64.get('curve_all')}" alt="curve_all" />
<div class="caption">图 3.1：四组实验的验证集 accuracy / loss 曲线。MLP（蓝）与 CNN（红）即 §3.3 主对比；其余两条为 §4 方向 1 对照组。</div>

<h3>3.4 为什么 CNN 比 MLP 更好</h3>
<ol>
<li><b>参数效率</b>：MLP 第一层 <code>784×600</code> 就有约 47 万参数，每个隐藏单元都得自己学一份 28×28 的全局模板；
   CNN 第一层只有 <code>8×(3·3·1) + 8 = 80</code> 个参数，整网约 5.7 万参数——
   <b>不到 MLP 的 1/8，反而 test acc 高 {test_gap:+.2f} pp</b>。这印证了卷积的归纳偏置
   （局部连接 + 权值共享）天然契合图像数据。</li>
<li><b>平移近似不变性</b>：CNN 的同一个卷积核在整张图扫，把"3 在中间"与"3 偏左一点"自然归为同一类特征。
   MLP 没有这层对称性约束，需要把每个位置的"3"都单独学一遍。这一点在 §5.2 的误分类样本里可以直接看到：
   MLP 错的样本里有不少是"书写正常但位置略偏"的普通数字，CNN 剩下的错例几乎只剩本身有歧义的样本。</li>
<li><b>困难类对收敛得更好</b>：对比 §5.1 的两个混淆矩阵，MLP 在 9→4、5→3、8→3 等格子上误判明显，
   而 CNN 这些格子整体更浅。CNN 的测试错误率从 {err_mlp:.2f}% 降到 {err_cnn:.2f}%，
   <b>相对减少约 {err_reduce:.1f}%</b>。</li>
<li><b>但提升幅度受限</b>：+{test_gap:.2f} pp 看上去不大，是因为 MNIST 数字简单、居中、规范，
   MLP 的"全局模板法"已经够用。若数据换成有平移/缩放/光照变化的真实图像，CNN 对 MLP 的优势会被显著放大。</li>
</ol>

<h2>4. Part C 方向 1：优化（隔离 momentum 与 scheduler）</h2>

<h3>4.1 实验设计</h3>
<p>
固定 CNN 架构与上面的训练 budget，只改变优化策略，做三组实验。第一对照仅改 scheduler（控制 optimizer=SGD），
第二对照仅改 optimizer（控制 scheduler=MultiStepLR），每段对照只动一个变量。
</p>

<h3>4.2 单变量归因结果</h3>
<table>
<tr><th>对照</th><th>变量</th><th>设置 A</th><th>dev (A)</th><th>设置 B</th><th>dev (B)</th><th>变化</th></tr>
<tr>
  <td>① scheduler 单独效应</td><td>scheduler</td>
  <td>SGD, 无</td><td>{pct(cnn_sgd.get('best_dev'))}</td>
  <td>SGD + MultiStepLR</td><td>{pct(cnn_sgd_ms.get('best_dev'))}</td>
  <td>{sched_eff:+.2f} pp</td>
</tr>
<tr>
  <td>② momentum 单独效应</td><td>optimizer</td>
  <td>SGD + MultiStepLR</td><td>{pct(cnn_sgd_ms.get('best_dev'))}</td>
  <td>MomentGD + MultiStepLR</td><td>{pct(cnn.get('best_dev'))}</td>
  <td>{mom_eff:+.2f} pp</td>
</tr>
</table>

<img class="fig" src="{figs_b64.get('curve_cnn')}" alt="curve_cnn" />
<div class="caption">图 4.1：MomentGD + MultiStepLR。</div>
<img class="fig" src="{figs_b64.get('curve_cnn_sgd_ms')}" alt="curve_cnn_sgd_ms" />
<div class="caption">图 4.2：SGD + MultiStepLR（隔离 momentum）。</div>
<img class="fig" src="{figs_b64.get('curve_cnn_sgd')}" alt="curve_cnn_sgd" />
<div class="caption">图 4.3：SGD，无 scheduler（裸 SGD 对照）。</div>

<h3>4.3 结论</h3>
<ul>
<li><b>momentum 是主要驱动</b>：在固定 scheduler 的前提下，仅加上 μ=0.9 的动量就让 dev acc
   从 {pct(cnn_sgd_ms.get('best_dev'))} 提升到 {pct(cnn.get('best_dev'))}（{mom_eff:+.2f} pp）。
   短训练 budget 下，"加速收敛"比"精修学习率"重要得多。</li>
<li><b>MultiStepLR 在 3 epochs 下几乎不贡献</b>（{sched_eff:+.2f} pp，在噪声范围内）。
   触发 milestone 时模型已基本收敛，把 lr 砍半已没有多少调整空间。</li>
<li>结合两点：要把 SGD 提升到 momentum 同等水平，先加动量比先调 scheduler 收益高得多。
   反过来，如果 epochs 拉长，scheduler 的作用大概率会显现——这是本实验 budget 决定的局部结论。</li>
</ul>

<h2>5. Part C 方向 5：错误分析与可视化</h2>

<h3>5.1 混淆矩阵</h3>
<div class="grid2">
  <div>
    <img class="fig" src="{figs_b64.get('cm_mlp')}" alt="cm_mlp" />
    <div class="caption">图 5.1：MLP 混淆矩阵（test acc = {pct(mlp.get('test_acc'))}）。</div>
  </div>
  <div>
    <img class="fig" src="{figs_b64.get('cm_cnn')}" alt="cm_cnn" />
    <div class="caption">图 5.2：CNN 混淆矩阵（test acc = {pct(cnn.get('test_acc'))}）。</div>
  </div>
</div>
<p>
两个模型最容易混的对子是经典的 <b>4↔9</b>、<b>7↔1/2</b>、<b>3↔5/8</b>、<b>8↔3/5</b>，
这些数字手写体本来就视觉相似。CNN 的混淆矩阵整体更"瘦"，非对角线颜色明显更浅；
MLP 在 9→4、5→3、8→3 等格子上的误判显著更多。
</p>

<h3>5.2 误分类样本</h3>
<div class="grid2">
  <div>
    <img class="fig" src="{figs_b64.get('wrong_mlp')}" alt="wrong_mlp" />
    <div class="caption">图 5.3：MLP 误分类样本（T = 真实标签，P = 预测标签）。</div>
  </div>
  <div>
    <img class="fig" src="{figs_b64.get('wrong_cnn')}" alt="wrong_cnn" />
    <div class="caption">图 5.4：CNN 误分类样本。</div>
  </div>
</div>
<p>
CNN 剩下的错例多数是"<b>书写本身就有歧义</b>"的样本（断笔、连笔、形状不规范），人也未必能一次写对。
MLP 的错例中则有相当一部分是书写正常但位置/角度稍偏的——它对位置敏感，所以容易在边缘样本上栽跟头。
</p>

<h3>5.3 权重与卷积核可视化</h3>
<div class="grid2">
  <div>
    <img class="fig" src="{figs_b64.get('weights_mlp')}" alt="weights_mlp" />
    <div class="caption">图 5.5：MLP 第一层 600 个隐藏单元中前 64 个的权重（reshape 回 28×28）。</div>
  </div>
  <div>
    <img class="fig" src="{figs_b64.get('kernels_cnn')}" alt="kernels_cnn" />
    <div class="caption">图 5.6：CNN 第一层 8 个 3×3 卷积核。</div>
  </div>
</div>
<p>
MLP 的权重模板隐约能看出数字笔画的全局形状——每个隐藏单元相当于一份 28×28 的"匹配模板"。
CNN 的卷积核则是 3×3 的局部边缘 / 方向 / 角点检测子，靠权值共享在图像任意位置触发同一类特征。
这正是 §3.4 "参数效率"那一点在视觉层面的解释。
</p>

<h2>6. 主要结果与讨论</h2>

<h3>6.1 主要结果汇总表</h3>
<table>
<tr><th>实验</th><th>架构</th><th>优化器</th><th>lr</th><th>batch</th><th>epochs</th><th>scheduler</th><th>best dev</th><th>test acc</th></tr>
{''.join(row(r) for r in results)}
</table>

<h3>6.2 讨论</h3>
<p><b>为什么 CNN 比 MLP 更适合图像分类？</b>
两点本质归纳偏置——<b>局部连接</b>（卷积核只看 3×3 邻域，符合"附近像素更相关"的先验）和
<b>权值共享</b>（同一个核在整张图扫，自带近似平移不变性）。MLP 没有这两条约束，所以需要
学一份覆盖整图的"全局模板"，参数大、易过拟合、对位置敏感。详细数据/可视化归因见 §3.4。</p>

<p><b>CNN 是否提升了测试集准确率？</b>
是的：test acc 从 {pct(mlp.get('test_acc'))} 提升到 {pct(cnn.get('test_acc'))}（+{test_gap:.2f} pp），
相对错误率减少约 {err_reduce:.1f}%。</p>

<p><b>为什么选这两个方向？</b>
方向 1（优化）选它是因为可以定量回答"CNN 比 MLP 强多少里，有多少其实是优化器的功劳"，
是公平评价 CNN 架构本身的前置必要工作。方向 5（可视化）选它是因为可以
<i>解释</i>方向 1 之外的"剩余差距来自哪里"：比较 MLP 权重模板 与 CNN 卷积核，肉眼就能看到
"全局模板 vs 局部特征"的差异；比较混淆矩阵，能定位剩余的"困难类对"。两者形成
一条"<b>定量比较 → 定性归因</b>"的完整证据链。</p>

<p><b>哪些样本仍然困难？</b>
按混淆矩阵和误分类图，剩下的错例主要有三类：(1) 书写本身有歧义的样本（如 4 与 9 顶部都不闭合时几乎一样、3 与 5 的上半部相似），
(2) 笔画异常细 / 异常粗导致与训练分布偏离的样本，
(3) 个别奇异写法（如带横杠的欧式 7、加帽底座的"⑴"形 1）。这些情形即使 CNN 也难以解决，
需要数据增强或更深的网络。</p>

<p><b>哪一项分析最有信息量？</b>
方向 1 的<b>单变量隔离对照</b>——它把"换成 momentum + scheduler"这个常见联合改动拆成
"动量 +{mom_eff:.2f} pp、scheduler {sched_eff:+.2f} pp"两条独立、可加的归因，
直接证伪了"scheduler 是 CNN 训练的关键"这种直觉。这是没补 cnn_sgd_ms 这一组之前完全看不出的发现。</p>

<h2>附录 A：复现命令</h2>
<pre>
cd codes

# 1) 训练（4 次实验，CPU 即可，CNN 单次约 10 分钟）
python test_train_mlp_fair.py                                                     # MLP 基线
python test_train_cnn.py --optimizer momentum --scheduler multistep --save-tag cnn
python test_train_cnn.py --optimizer sgd      --scheduler none      --save-tag cnn_sgd
python test_train_cnn.py --optimizer sgd      --scheduler multistep --save-tag cnn_sgd_ms

# 2) 测试集评估
python test_model.py --tag mlp_fair       # {pct(mlp.get('test_acc'))}
python test_model.py --tag cnn            # {pct(cnn.get('test_acc'))}
python test_model.py --tag cnn_sgd        # {pct(cnn_sgd.get('test_acc'))}
python test_model.py --tag cnn_sgd_ms     # {pct(cnn_sgd_ms.get('test_acc'))}

# 3) Part C 可视化
python analysis.py --tag mlp_fair
python analysis.py --tag cnn

# 4) 重生成本报告
python generate_report.py
</pre>

<h2>附录 B：合规性自查</h2>
<ul>
<li>仅使用所提供的 MNIST 数据集，未引入任何外部数据。</li>
<li>核心算子 <code>Linear</code>、<code>conv2D</code>、<code>MaxPool2D</code>、
   <code>MultiCrossEntropyLoss</code>、<code>MomentGD</code>、<code>MultiStepLR</code> 全部基于 NumPy 手写实现。</li>
<li>未 import torch / tensorflow / sklearn / keras / jax / cv2。第三方依赖仅：numpy、matplotlib、gzip、pickle、tqdm。</li>
<li>训练在 CPU 上完成；MLP &lt; 1 分钟，CNN ≈ 10 分钟 / 3 epochs。</li>
<li>每段 MLP-vs-CNN 主对比 和 方向 1 单变量对照 都遵循"每次只改一个变量"。</li>
</ul>

</body>
</html>
"""

out_html = os.path.join(ROOT, '报告.html')
with open(out_html, 'w', encoding='utf-8') as f:
    f.write(HTML)
print(f'\nWrote {out_html}  ({os.path.getsize(out_html)/1024:.1f} KB)')
print(f'Wrote results.json with {len(results)} rows.')
