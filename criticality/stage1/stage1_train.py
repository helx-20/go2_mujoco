import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

def compute_auc(y_true, y_score):
    # safe AUC computation using rank method; returns 0.0 if not computable
    try:
        yt = np.asarray(y_true)
        ys = np.asarray(y_score)
        if len(yt) == 0 or len(np.unique(yt)) < 2:
            return 0.0
        P = int((yt == 1).sum())
        N = int((yt == 0).sum())
        if P == 0 or N == 0:
            return 0.0
        ranks = np.argsort(np.argsort(ys)) + 1
        sum_ranks_pos = ranks[yt == 1].sum()
        auc = (sum_ranks_pos - P * (P + 1) / 2.0) / (P * N)
        return float(auc)
    except Exception:
        return 0.0

# ensure project root is on sys.path so 'criticality' can be imported regardless of cwd
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from criticality.utils.criticality_model import SimpleClassifier


def train(args):
    data_dir = args.data_dir
    device = torch.device(args.device)
    # Prefer pre-split files (pos_train/val/test and neg_train/val/test). Fallback to pos.npy/neg.npy if not present.
    pos_train_path = os.path.join(data_dir, 'pos_train.npy')
    neg_train_path = os.path.join(data_dir, 'neg_train.npy')
    pos_val_path = os.path.join(data_dir, 'pos_val.npy')
    neg_val_path = os.path.join(data_dir, 'neg_val.npy')

    pos_train = np.load(pos_train_path)
    neg_train = np.load(neg_train_path)
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
        val_auc = compute_auc(y_true_val, y_score_val)
        print(f'Epoch {epoch+1}/{args.epochs} train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_p={val_precision:.4f} val_r={val_recall:.4f} val_auc={val_auc:.4f}')

        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'stage1_criticality_best.pt'))

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
    best_model_path = os.path.join(args.save_dir, 'stage1_criticality_best.pt')
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
        test_auc = compute_auc(y_true_test, y_score_test)
        print(f'Test acc on {total} samples = {test_acc:.4f} (pos={len(pos_test)} neg={len(neg_test)}) p={test_precision:.4f} r={test_recall:.4f} auc={test_auc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/stage1')
    parser.add_argument('--save_dir', default='criticality/stage1/model')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--hidden', default=256, type=int)
    parser.add_argument('--test', action='store_true', help='Run test evaluation after training')
    args = parser.parse_args()
    if args.test:
        test(args)
    else:
        train(args)
