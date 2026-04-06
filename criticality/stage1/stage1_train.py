import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ensure project root is on sys.path so 'criticality' can be imported regardless of cwd
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from criticality.utils.criticality_model import SimpleClassifier
from sklearn.metrics import auc


def precision_recall_curve(y_true, y_score, num_thresholds=1000):
    """Compute precision-recall curve over a uniform grid of thresholds in [0,1].

    Returns (precision, recall, thresholds) where `thresholds` has length
    `num_thresholds` and `precision`/`recall` have length `num_thresholds + 1` so
    that `precision[i+1]`/`recall[i+1]` correspond to `thresholds[i]` (sklearn-style).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0 or y_score.size == 0:
        thr = np.linspace(1.0, 0.0, num_thresholds, endpoint=False)
        prec = np.concatenate(([1.0], np.zeros(num_thresholds)))
        rec = np.concatenate(([0.0], np.zeros(num_thresholds)))
        return prec, rec, thr

    # descending thresholds so thr[0] maps to precision/recall index 1
    thr = np.linspace(1.0, 0.0, num_thresholds, endpoint=False)
    prec_list = []
    rec_list = []
    positives = int(np.sum(y_true == 1))
    for t in thr:
        preds = (y_score >= t).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = positives - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec_list.append(prec)
        rec_list.append(rec)

    # prepend the "point at infinity" (no predictions positive)
    prec_arr = np.concatenate(([1.0], np.array(prec_list)))
    rec_arr = np.concatenate(([0.0], np.array(rec_list)))
    return prec_arr, rec_arr, thr

def train(args):
    data_dir = args.data_dir
    device = torch.device(args.device)

    pos_train_path = os.path.join(data_dir, 'pos_train.npy')
    neg_train_path = os.path.join(data_dir, 'neg_train.npy')
    pos_val_path = os.path.join(data_dir, 'pos_val.npy')
    neg_val_path = os.path.join(data_dir, 'neg_val.npy')

    pos_train = np.load(pos_train_path)
    neg_train = np.load(neg_train_path)
    if len(neg_train) > len(pos_train) * args.train_ratio:
        neg_train = neg_train[np.random.choice(len(neg_train), size=len(pos_train) * args.train_ratio, replace=False)]

    pos_val = np.load(pos_val_path) if os.path.exists(pos_val_path) else np.zeros((0, pos_train.shape[1]))
    neg_val = np.load(neg_val_path) if os.path.exists(neg_val_path) else np.zeros((0, neg_train.shape[1]))

    X_train = np.concatenate([pos_train, neg_train], axis=0)
    y_train = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))], axis=0)

    X_val = np.concatenate([pos_val, neg_val], axis=0) if (len(pos_val) + len(neg_val)) > 0 else np.zeros((0, X_train.shape[1]))
    y_val = np.concatenate([np.ones(len(pos_val)), np.zeros(len(neg_val))], axis=0) if (len(pos_val) + len(neg_val)) > 0 else np.zeros((0,))

    input_dim = X_train.shape[1]
    model = SimpleClassifier(input_dim=input_dim, hidden=args.hidden).to(device)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_auc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total = 0
        correct = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
        train_acc = correct / total

        # val
        model.eval()
        total = 0
        correct = 0
        tp = 0
        fp = 0
        fn = 0
        y_true_val = []
        y_score_val = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = logits.argmax(dim=1)
                total += yb.size(0)
                correct += (preds == yb).sum().item()
                tp += int(((preds == 1) & (yb == 1)).sum().item())
                fp += int(((preds == 1) & (yb == 0)).sum().item())
                fn += int(((preds == 0) & (yb == 1)).sum().item())
                y_score_val.extend(probs.cpu().numpy().tolist())
                y_true_val.extend(yb.cpu().numpy().tolist())
        val_acc = correct / total if total > 0 else 0.0
        val_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        val_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec, rec, thr = precision_recall_curve(y_true_val, y_score_val)
        val_auc = auc(rec, prec)
        print(f'Epoch {epoch+1}/{args.epochs} train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_p={val_precision:.4f} val_r={val_recall:.4f} val_auc={val_auc:.4f}')

        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'stage1_criticality_best_{args.model_idx}.pt'))

    print('Done. Best val auc:', best_auc)


def test(args):
    data_dir = args.data_dir
    device = torch.device(args.device)

    pos_test_path = os.path.join(data_dir, 'pos_test.npy')
    neg_test_path = os.path.join(data_dir, 'neg_test.npy')
    if not (os.path.exists(pos_test_path) and os.path.exists(neg_test_path)):
        print('No test split found (pos_test.npy / neg_test.npy). Skipping test evaluation.')
        return

    pos_test = np.load(pos_test_path)
    neg_test = np.load(neg_test_path)
    X_test = np.concatenate([pos_test, neg_test], axis=0)
    y_test = np.concatenate([np.ones(len(pos_test)), np.zeros(len(neg_test))], axis=0)

    # build model architecture from test data shape
    input_dim = X_test.shape[1]
    model = SimpleClassifier(input_dim=input_dim, hidden=args.hidden).to(device)

    # if we saved a best model, load it for test
    best_model_path = os.path.join(args.save_dir, f'stage1_criticality_best_{args.model_idx}.pt')
    if os.path.exists(best_model_path):
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        except Exception:
            pass
        test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

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
        print(f'Test acc on {total} samples = {test_acc:.4f} (pos={len(pos_test)} neg={len(neg_test)}) p={test_precision:.4f} r={test_recall:.4f} auc={test_auc:.4f}')

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
    parser.add_argument('--data_dir', default='/mnt/mnt1/linxuan/go2_data/data/stage1')
    parser.add_argument('--save_dir', default='criticality/stage1/model')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--hidden', default=256, type=int)
    parser.add_argument('--model_idx', default=1, type=int, help='index to append to saved model filename for multiple runs')
    parser.add_argument('--train_ratio', default=100, type=int)
    parser.add_argument('--test', action='store_true', help='Run test evaluation after training')
    args = parser.parse_args()
    print('args:', args)
    if args.test:
        test(args)
    else:
        train(args)
