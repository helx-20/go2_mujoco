import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleClassifier(nn.Module):
	"""A small MLP classifier that takes a single-step observation and outputs binary logits."""
	def __init__(self, input_dim: int = 56, hidden: int = 256, hidden_layer=1, num_classes: int = 2):
		super().__init__()
		hidden_layers = [nn.Linear(hidden, hidden), nn.ReLU()] * hidden_layer
		self.net = nn.Sequential(
			nn.Linear(input_dim, hidden),
			nn.ReLU(),
			*hidden_layers,
			nn.Linear(hidden, num_classes)
		)

	def forward(self, x):
		# x: [B, input_dim]
		logits = self.net(x)
		return logits
