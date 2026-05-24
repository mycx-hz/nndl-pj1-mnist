"""Upload the 4 reported checkpoints + report to a ModelScope model repo.

Prereqs:
    1) pip install modelscope
    2) Get your SDK access token from https://www.modelscope.cn/my/myaccesstoken
    3) Create an empty model repo on the website first:
       https://www.modelscope.cn/models/create  (e.g. <username>/nndl-pj1-mnist)
    4) export MODELSCOPE_API_TOKEN=<your token>
    5) python upload_to_modelscope.py --repo <username>/nndl-pj1-mnist

What gets uploaded (under the repo root):
    mlp.pickle            <- outputs/best_models/mlp_fair/best_model.pickle
    cnn.pickle            <- outputs/best_models/cnn/best_model.pickle
    cnn_sgd.pickle        <- outputs/best_models/cnn_sgd/best_model.pickle
    cnn_sgd_ms.pickle     <- outputs/best_models/cnn_sgd_ms/best_model.pickle
    报告.pdf, 报告.html   <- from the PJ1 root
    README.md             <- short auto-generated description

The legacy "mlp" (old baseline) run is NOT uploaded — it was retired from the
report because its training setup did not match the CNN's. See the report § 3.2
for the comparison setup.
"""
import os
import argparse
import tempfile
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--repo', required=True,
                    help='ModelScope model id, e.g. yourname/nndl-pj1-mnist')
parser.add_argument('--commit-msg', default='upload PJ1 checkpoints + report')
parser.add_argument('--token', default=os.environ.get('MODELSCOPE_API_TOKEN'),
                    help='SDK token (or set $MODELSCOPE_API_TOKEN).')
args = parser.parse_args()

if not args.token:
    raise SystemExit('Provide --token or set $MODELSCOPE_API_TOKEN. '
                     'Get one at https://www.modelscope.cn/my/myaccesstoken')

from modelscope.hub.api import HubApi   # type: ignore

api = HubApi()
api.login(args.token)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (source tag under outputs/best_models/, upload filename)
CKPTS = [
    ('mlp_fair',   'mlp.pickle'),
    ('cnn',        'cnn.pickle'),
    ('cnn_sgd',    'cnn_sgd.pickle'),
    ('cnn_sgd_ms', 'cnn_sgd_ms.pickle'),
]

# Stage all files into a temp dir with a flat layout.
stage = tempfile.mkdtemp(prefix='ms_upload_')
print(f'staging at {stage}')

readme_text = """# NN&DL Project-1 — MNIST checkpoints

Four MNIST classifiers trained from scratch with a NumPy-only mini-framework.
See https://github.com/<your-handle>/<repo> for full code and the report.

All four runs share the same training budget so they can be compared cleanly
(MomentGD vs SGD, with / without MultiStepLR — see report §3 / §4).

| File | Architecture | Optimizer | Scheduler | Test acc |
|---|---|---|---|---|
| mlp.pickle        | MLP 784→600→10                | MomentGD μ=0.9 lr=0.05 | MultiStepLR | 97.77% |
| cnn.pickle        | CNN Conv(1,8)→Conv(8,16)+FC(64) | MomentGD μ=0.9 lr=0.05 | MultiStepLR | 98.89% |
| cnn_sgd.pickle    | same CNN                      | SGD lr=0.05            | —           | 98.12% |
| cnn_sgd_ms.pickle | same CNN                      | SGD lr=0.05            | MultiStepLR | 97.99% |

Common training settings: batch=64, epochs=3, weight_decay λ=1e-4, He init.

Load with the project's mini-framework:
```python
import mynn as nn
m = nn.models.Model_CNN(build=False)      # or Model_MLP() for mlp.pickle
m.load_model('cnn.pickle')
```
"""
with open(os.path.join(stage, 'README.md'), 'w', encoding='utf-8') as f:
    f.write(readme_text)

for tag, upload_name in CKPTS:
    src = os.path.join(HERE, 'outputs', 'best_models', tag, 'best_model.pickle')
    if not os.path.exists(src):
        print(f'  [skip] {src} missing')
        continue
    dst = os.path.join(stage, upload_name)
    shutil.copy2(src, dst)
    print(f'  staged {upload_name} ({os.path.getsize(dst)/1024:.1f} KB)')

for name in ('报告.pdf', '报告.html'):
    src = os.path.join(ROOT, name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(stage, name))
        print(f'  staged {name}')

api.push_model(
    model_id=args.repo,
    model_dir=stage,
    commit_message=args.commit_msg,
)
print(f'\nDone. Browse it at https://www.modelscope.cn/models/{args.repo}/files')
