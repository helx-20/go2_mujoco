"""
Stage2: simple RL fine-tuning scaffold.

This script loads a pretrained stage1 classifier and uses it as a reward-shaping penalty
during RL training of a terrain policy.

Note: this is a scaffold — adapt hyperparams and policy architecture to your setup.
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from criticality.utils.criticality_model import SimpleClassifier
from criticality.stage1.stage1_train import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load classifier
    model = SimpleClassifier(input_dim=56, hidden=256)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval().to(device)

    data_dir = args.test_dir
    pos_test_path = os.path.join(data_dir, 'pos_test.npy')
    neg_test_path = os.path.join(data_dir, 'neg_test.npy')
    pos_test = np.load(pos_test_path)
    neg_test = np.load(neg_test_path)
    X_test = np.concatenate([pos_test, neg_test], axis=0)
    y_test = np.concatenate([np.ones(len(pos_test)), np.zeros(len(neg_test))], axis=0)
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    model.eval()
    total = 0
    correct = 0
    tp = 0
    fp = 0
    fn = 0
    y_true_test = []
    y_score_test = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
            tp += int(((preds == 1) & (yb == 1)).sum().item())
            fp += int(((preds == 1) & (yb == 0)).sum().item())
            fn += int(((preds == 0) & (yb == 1)).sum().item())
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_score_test.extend(probs.cpu().numpy().tolist())
            y_true_test.extend(yb.cpu().numpy().tolist())
    test_acc = correct / total if total > 0 else 0.0
    test_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    test_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec, rec, thr = precision_recall_curve(y_true_test, y_score_test)
    test_auc = auc(rec, prec)

    # save PR arrays
    os.makedirs(args.save_dir, exist_ok=True)
    # plot and save figure if matplotlib available
    if plt is not None and prec.size > 0:
        fig = plt.figure()
        plt.step(rec, prec, where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall (AUC={test_auc:.4f})')
        plt.grid(True)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        point_num = 10
        gap_num = len(thr) // point_num
        points = [(rec[gap_num * i], prec[gap_num * i], thr[gap_num * i]) for i in range(point_num)]
        for x, y, tval in points:
            plt.scatter([x], [y], color='red', s=24)
            plt.annotate(f'{tval:.2f}', xy=(x, y), xytext=(6, -10), textcoords='offset points', fontsize=8, color='red')
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, 'precision_recall.png'), dpi=600)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='criticality/stage2/model/stage2_2_epoch900.pt')
    parser.add_argument('--test_dir', default='/mnt/mnt1/linxuan/go2_data/data/stage1')
    parser.add_argument('--save_dir', default='criticality/stage2/results')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    main(args)
