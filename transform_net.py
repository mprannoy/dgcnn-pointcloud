import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class input_transform_net(nn.Module):
	"""docstring for input_transform_net"""
	def __init__(self, K=3):
		super(input_transform_net, self).__init__()

		self.conv1 = nn.Conv2d(6, 64, 1)
		self.conv2 = nn.Conv2d(64, 128, 1)
		self.conv3 = nn.Conv2d(128, 1024, 1)


		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, K*K)

		self.bn1 = nn.BatchNorm2d(64)
		self.bn2 = nn.BatchNorm2d(128)
		self.bn3 = nn.BatchNorm2d(1024)
		
		self.bn4 = nn.BatchNorm1d(512)
		self.bn5 = nn.BatchNorm1d(256)


		self.const = Variable(torch.from_numpy(np.eye(K).flatten()).float())
		self.K = K



	def forward(self, edge_feat):

		batch_size, num_points = edge_feat.size()[0], edge_feat.size()[2]

		self.mp1 = nn.MaxPool2d((num_points, 1), stride=2)

		x = self.bn1(F.relu(self.conv1(edge_feat)))
		x = self.bn2(F.relu(self.conv2(x)))
		x,_ = torch.max(x, dim=-1, keepdim=True)

		x = self.bn3(F.relu(self.conv3(x)))
		x = self.mp1(x)

		x = x.view(batch_size, -1)

		x = self.bn4(F.relu(self.fc1(x)))
		x = self.bn5(F.relu(self.fc2(x)))

		x = self.fc3(x) + self.const

		x = x.view(batch_size, self.K, self.K)

		return x