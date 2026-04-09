import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ensure project root is on sys.path so 'criticality' can be imported regardless of cwd
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from criticality.utils.criticality_model import SimpleClassifier
from sklearn.metrics import auc
from criticality.utils.grad_ops import get_gradient_norms_var_cls_parallel


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
    pos_train_append_path = os.path.join(args.append_data_dir, 'pos_train.npy')
    pos_val_append_path = os.path.join(args.append_data_dir, 'pos_val.npy')

    pos_train = np.load(pos_train_path)
    pos_train_append = np.load(pos_train_append_path) if os.path.exists(pos_train_append_path) else None
    if pos_train_append is not None:
        pos_train = np.concatenate([pos_train, pos_train_append], axis=0)
    neg_train = np.load(neg_train_path)
    if len(neg_train) > len(pos_train) * args.train_ratio:
        neg_train = neg_train[np.random.choice(len(neg_train), size=len(pos_train) * args.train_ratio, replace=False)]

    pos_val = np.load(pos_val_path)
    neg_val = np.load(neg_val_path)
    # pos_val_append = np.load(pos_val_append_path) if os.path.exists(pos_val_append_path) else None
    # if pos_val_append is not None:
    #     pos_val = np.concatenate([pos_val, pos_val_append], axis=0)

    X_train = np.concatenate([pos_train, neg_train], axis=0)
    y_train = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))], axis=0)

    X_val = np.concatenate([pos_val, neg_val], axis=0) if (len(pos_val) + len(neg_val)) > 0 else np.zeros((0, X_train.shape[1]))
    y_val = np.concatenate([np.ones(len(pos_val)), np.zeros(len(neg_val))], axis=0) if (len(pos_val) + len(neg_val)) > 0 else np.zeros((0,))

    input_dim = X_train.shape[1]
    model = SimpleClassifier(input_dim=input_dim, hidden=args.hidden).to(device)
    # optionally initialize from an existing stage1 model
    init_path = getattr(args, 'initial_model_path', '')
    if init_path:
        if os.path.exists(init_path):
            try:
                state = torch.load(init_path, map_location=device)
                model.load_state_dict(state)
                print(f'Loaded initial weights from {init_path}')
            except Exception as e:
                print(f'Warning: failed to load initial model {init_path}: {e}')
        else:
            print(f'Initial model path {init_path} does not exist, continuing with random init')

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

    base_train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_auc = 0.0
    for epoch in range(args.epochs):
        # IS full-mix (dense) sampling: compute per-sample gradient-norms and sample a subset every epoch
        num_samples = getattr(args, 'dense_num_samples', 2048)

        # compute per-sample gradient norms in parallel using functorch/vmap implementation
        model.eval()
        grad_loader = DataLoader(train_ds, batch_size=max(512, args.batch_size), shuffle=False)
        grad_norms = get_gradient_norms_var_cls_parallel(model, grad_loader, device=device, criterion=criterion)
        probs = grad_norms.cpu().numpy().astype(float)
        from criticality.utils.rarity import calculate_rarity 
        rarity = calculate_rarity(probs.reshape(-1, 1))
        print("rarity:", rarity)
        probs = probs / probs.sum()
        n = len(probs)
        k = num_samples
        idx = np.random.choice(n, size=k, replace=False, p=probs)
        mix_num = int(num_samples * args.mix_ratio)
        mix_idx = np.random.choice(n, size=mix_num, replace=False)
        idx = np.concatenate([idx, mix_idx], axis=0)

        if idx.size > 0:
            sub_dataset = torch.utils.data.Subset(train_ds, idx.tolist())
            train_loader = DataLoader(sub_dataset, batch_size=args.batch_size, shuffle=True)
        else:
            train_loader = base_train_loader

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
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'stage1_plus_criticality_best_new_{args.model_idx}.pt'))
            print(f'New best model saved with val_auc={best_auc:.4f} at epoch {epoch+1}')

    print('Done. Best val auc:', best_auc)


def test(args):
    data_dir = args.data_dir
    device = torch.device(args.device)

    pos_test_path = os.path.join(data_dir, 'pos_test.npy')
    neg_test_path = os.path.join(data_dir, 'neg_test.npy')
    pos_test_append_path = os.path.join(args.append_data_dir, 'pos_test.npy')

    pos_test = np.load(pos_test_path)
    neg_test = np.load(neg_test_path)
    # if os.path.exists(pos_test_append_path):
    #     pos_test_append = np.load(pos_test_append_path)
    #     pos_test = np.concatenate([pos_test, pos_test_append], axis=0)
    X_test = np.concatenate([pos_test, neg_test], axis=0)
    y_test = np.concatenate([np.ones(len(pos_test)), np.zeros(len(neg_test))], axis=0)

    # build model architecture from test data shape
    input_dim = X_test.shape[1]
    model = SimpleClassifier(input_dim=input_dim, hidden=args.hidden).to(device)

    # if we saved a best model, load it for test
    best_model_path = os.path.join(args.save_dir, f'stage1_plus_criticality_best_new_{args.model_idx}.pt')
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
            point_num = 20
            idx = []
            for i in range(point_num):
                for tmp in range(len(thr)):
                    if rec[tmp] >= max(rec) - (max(rec) - min(rec)) * i / point_num:
                        idx.append(tmp)
                        break
            points = [(rec[i], prec[i], thr[i]) for i in idx]
            for x, y, tval in points:
                plt.scatter([x], [y], color='red', s=24)
                plt.annotate(f'{tval:.2f}', xy=(x, y), xytext=(0, -10), textcoords='offset points', fontsize=8, color='red')
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, 'precision_recall.png'), dpi=600)
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/mnt/mnt1/linxuan/go2_data/data/stage1')
    parser.add_argument('--append_data_dir', default='/mnt/mnt1/linxuan/go2_data/data/stage1_plus')
    parser.add_argument('--save_dir', default='criticality/stage1_plus/model')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--hidden', default=256, type=int)
    parser.add_argument('--model_idx', default=1, type=int, help='index to append to saved model filename for multiple runs')
    parser.add_argument('--train_ratio', default=100, type=int)
    parser.add_argument('--dense_num_samples', default=512, type=int, help='Number of samples to draw each epoch for dense training')
    parser.add_argument('--mix_ratio', default=8, type=float)
    parser.add_argument('--test', action='store_true', help='Run test evaluation after training')
    parser.add_argument('--initial_model_path', default='criticality/stage1/model/stage1_criticality_best_new_1.pt', help='Path to a stage1 model .pt file to initialize weights from')
    args = parser.parse_args()
    print('args:', args)
    if args.test:
        test(args)
    else:
        train(args)
        test(args)
