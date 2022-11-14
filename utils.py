# some convenient functions used in this project
# reference: https://github.com/d2l-ai/d2l-en
import os
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
from sklearn.metrics import f1_score, accuracy_score
from IPython import display
import time
import sklearn.metrics
import json
import collections
import numpy as np

def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_utils`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        
    def save(self, output_dir, fname):
        plt.savefig(os.path.join(output_dir, fname), dpi=600)

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


def accuracy(y_pred, y):
    """Compute the number of correct predictions.
    Defined in :numref:`sec_utils`"""
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = torch.argmax(y_pred, axis=1)
    cmp = (y_pred.type(y.dtype)) == y
    return float(torch.sum(cmp.type(y.dtype)))

def evaluate_loss_and_acc_gpu(net, data_iter, loss=None, device=None):
    """Compute the accuracy for a model on a dataset using a GPU."""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    if loss:
        metric = Accumulator(3)
    else:
        metric = Accumulator(2)
    
    with torch.no_grad():
        for batch in data_iter:
            if isinstance(batch, dict):
                for k, v in batch.items():
                    batch[k] = batch[k].to(device)
            else:
                batch = batch.to(device)
            y = batch['label_id']
            y_pred = net(**batch)
            if loss:
                metric.add(loss(y_pred, y).sum(), accuracy(y_pred, y), torch.numel(y))
            else:
                metric.add(accuracy(y_pred, y), torch.numel(y))
    return (metric[0] / metric[2], metric[1] / metric[2]) if loss else metric[0] / metric[1]

def test(net, data_iter, output_dir=None, device=None):
    net.eval()
    full_logits=[]
    full_label_ids=[]
    full_aspect_ids = []
    for i, batch in enumerate(data_iter):
        with torch.no_grad():
            if isinstance(batch, dict):
                for k, v in batch.items():
                    batch[k] = batch[k].to(device)
            else:
                batch = batch.to(device)
            
            logits = net(**batch)
            logits = logits.detach().cpu().numpy()
            y = batch['label_id'].cpu().numpy()
            aspect_id = batch['aspect_id'].cpu().numpy()

            full_logits.extend(logits.tolist())
            full_label_ids.extend(y.tolist())
            full_aspect_ids.extend(aspect_id.tolist())
    
    
    y_pred = [np.argmax(logit) for logit in full_logits]
    y_true = full_label_ids
    f1 =sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    acc =sklearn.metrics.accuracy_score(y_true, y_pred)
    print(f"test on final epoch -- acc: {acc*100 :.2f}, f1-macro: {f1*100 :.2f}")
    if output_dir:
        output_eval_json = os.path.join(output_dir, str(net.__class__.__name__) + "_predictions.json") 
        with open(output_eval_json, "w") as fw:
            json.dump({"logits": full_logits, "label_ids": full_label_ids, "aspect_ids": full_aspect_ids}, fw)
            
def test_on_checkpoint(net, file_path, data_iter, output_dir=None, device=None):
    state_dict = torch.load(file_path, map_location=device)
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.to(device)
    net.eval()
    full_logits=[]
    full_label_ids=[]
    full_aspect_ids = []
    for i, batch in enumerate(data_iter):
        with torch.no_grad():
            if isinstance(batch, dict):
                for k, v in batch.items():
                    batch[k] = batch[k].to(device)
            else:
                batch = batch.to(device)
            
            logits = net(**batch)
            logits = logits.detach().cpu().numpy()
            y = batch['label_id'].cpu().numpy()
            aspect_id = batch['aspect_id'].cpu().numpy()

            full_logits.extend(logits.tolist())
            full_label_ids.extend(y.tolist())
            full_aspect_ids.extend(aspect_id.tolist())
    
    
    y_pred = [np.argmax(logit) for logit in full_logits]
    y_true = full_label_ids
    f1 =sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    acc =sklearn.metrics.accuracy_score(y_true, y_pred)
    print(f"test on {file_path} -- acc: {acc*100 :.2f}, f1-macro: {f1*100 :.2f}")
    if output_dir:
        output_eval_json = os.path.join(output_dir, str(net.__class__.__name__) + "_predictions.json") 
        with open(output_eval_json, "w") as fw:
            json.dump({"logits": full_logits, "label_ids": full_label_ids, "aspect_ids": full_aspect_ids}, fw)


def eval_results(results, prefix, aspects):
    col_1, col_2, col_3 = 'Aspects', prefix + '_Acc', prefix + '_F1'
    outputs = collections.defaultdict(list)
    y_true = results['label_ids']
    y_pred = [np.argmax(logit) for logit in results['logits']]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    outputs[col_1].append('All')
    outputs[col_2].append(acc)
    outputs[col_3].append(f1)

    aspect_dict = {}
    for aspect_id, aspect in enumerate(aspects):
        aspect_dict[aspect_id] = ([], [])

    for i, aspect_id in enumerate(results['aspect_ids']):
        aspect_dict[aspect_id][0].append(y_true[i])
        aspect_dict[aspect_id][1].append(y_pred[i])

    for k, v in aspect_dict.items():
        acc = accuracy_score(v[0], v[1])
        f1 = f1_score(v[0], v[1], average='macro')
        outputs[col_1].append(aspects[k])
        outputs[col_2].append(acc)
        outputs[col_3].append(f1)
    return pd.DataFrame(outputs)

def calculate_mean(results, model, aspects):
    accs = []
    for i, res in enumerate(results):
        if i == 0:
            df_mean = eval_results(res, model, aspects)
            df_mean["Aspects"] = [a.upper() for a in df_mean["Aspects"]]
            df_mean[model + '_Acc'] = df_mean[model + '_Acc'].astype(float)
            df_mean[model + '_F1'] = df_mean[model + '_F1'].astype(float)
            accs.append(df_mean[model + '_Acc'].astype(float).values[0])
        else:
            df_temp = eval_results(res, model, aspects)
            df_mean[model +
                    '_Acc'] = df_mean[model +
                                      '_Acc'] + df_temp[model +
                                                        '_Acc'].astype(float)
            df_mean[model +
                    '_F1'] = df_mean[model +
                                     '_F1'] + df_temp[model +
                                                      '_F1'].astype(float)
            accs.append(df_temp[model + '_Acc'].astype(float).values[0])
    df_mean[model + '_Acc'] = df_mean[model + '_Acc'] / len(results)
    df_mean[model + '_F1'] = df_mean[model + '_F1'] / len(results)
    return df_mean

def read_and_calculate(aspects, paths, model_name):
    results = []
    for path in paths:
        with open(path) as f:
            res = json.load(f)
        results.append(res)
    df = calculate_mean(results, model_name, aspects)
    return df