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
from criticality.utils.dqn import DQN, ReplayBuffer
from criticality.stage1.stage1_train import precision_recall_curve
from sklearn.metrics import auc

def main(args):
    # load classifier
    device = torch.device('cuda' if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    q_net = SimpleClassifier(input_dim=56, hidden=256)
    if args.stage1_model and os.path.exists(args.stage1_model):
        q_net.load_state_dict(torch.load(args.stage1_model, map_location='cpu'))
    q_net.eval()

    # load replay buffer
    replay_buffer = ReplayBuffer(args.pos_path, args.neg_path, pos_ratio=0.5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = DQN(q_net, learning_rate=args.lr, gamma=args.gamma, target_update=args.target_update, device=device)

    data_dir = args.test_dir
    pos_test_path = os.path.join(data_dir, 'pos_test.npy')
    neg_test_path = os.path.join(data_dir, 'neg_test.npy')
    pos_test = np.load(pos_test_path)
    neg_test = np.load(neg_test_path)
    X_test = np.concatenate([pos_test, neg_test], axis=0)
    y_test = np.concatenate([np.ones(len(pos_test)), np.zeros(len(neg_test))], axis=0)
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    # training loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        inputs, next_obs, rewards, dones = replay_buffer.sample(args.batch_size)
        agent.update(inputs, next_obs, rewards, dones)
        if epoch % args.validate_interval == 0:
            test_auc = test_model(test_loader, agent.q_net, device)
            if test_auc > best_auc:
                best_auc = test_auc
                print(f'New best model with AUC={best_auc:.4f}')
            torch.save(agent.q_net.state_dict(), os.path.join(args.save_dir, f'stage2_{args.worker_id}_epoch{epoch}.pt'))
    torch.save(agent.q_net.state_dict(), os.path.join(args.save_dir, f'stage2_{args.worker_id}.pt'))

def test_model(test_loader, model, device):
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
    print(f'Test acc on {total} samples = {test_acc:.4f} p={test_precision:.4f} r={test_recall:.4f} auc={test_auc:.4f}')
    return test_auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--stage1_model', default='criticality/stage1/model/stage1_criticality_best_1.pt')
    parser.add_argument('--save_dir', default='criticality/stage2/model')
    parser.add_argument('--pos_path', default='/mnt/mnt1/linxuan/go2_data/data/stage2/replay_buffer_pos.npy')
    parser.add_argument('--neg_path', default='/mnt/mnt1/linxuan/go2_data/data/stage2/replay_buffer_neg.npy')
    parser.add_argument('--test_dir', default='/mnt/mnt1/linxuan/go2_data/data/stage1')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--target_update', default=50, type=int)
    parser.add_argument('--validate_interval', default=50, type=int)
    args = parser.parse_args()
    print('args:', args)
    main(args)
