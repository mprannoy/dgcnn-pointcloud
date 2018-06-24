import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as grad

from utils import *
from transform_net import input_transform_net

class classification_net(nn.Module):
 	"""docstring for edge_conv_model"""
 	def __init__(self, k=20):
 		super(classification_net, self).__init__()

 		self.conv1 = nn.Conv2d(6, 64, kernel_size=1)
 		self.conv2 = nn.Conv2d(128, 64, kernel_size=1)
 		self.conv3 = nn.Conv2d(128, 64, kernel_size=1)
 		self.conv4 = nn.Conv2d(128, 128, kernel_size=1)
 		self.conv5 = nn.Conv2d(320, 1024, kernel_size=1)

 		
 		self.bn1 = nn.BatchNorm2d(64)
 		self.bn2 = nn.BatchNorm2d(64)
 		self.bn3 = nn.BatchNorm2d(64)
 		self.bn4 = nn.BatchNorm2d(128)
 		self.bn5 = nn.BatchNorm2d(1024)
 		
 		self.bn6 = nn.BatchNorm1d(512)
 		self.bn7 = nn.BatchNorm1d(256)

 		self.fc1 = nn.Linear(1024, 512)
 		self.fc2 = nn.Linear(512, 256)
 		self.fc3 = nn.Linear(256, 40)

 		self.dropout = nn.Dropout(p=0.5)

 		self.input_transform = input_transform_net()
 		self.k = k
 		
 	

 	def forward(self, point_cloud):
 		batch_size, num_point,_ = point_cloud.size()

 		dist_mat = pairwise_distance(point_cloud)
 		nn_idx = knn(dist_mat, k=self.k)
 		edge_feat = get_edge_feature(point_cloud, nn_idx=nn_idx, k=self.k)

 		edge_feat = edge_feat.permute(0,3,1,2)
 		# point_cloud = point_cloud.permute(0,2,1)

 		transform_mat = self.input_transform(edge_feat)



 		point_cloud_transformed = torch.bmm(point_cloud, transform_mat)
 		dist_mat = pairwise_distance(point_cloud_transformed)
 		nn_idx = knn(dist_mat, k=self.k)
 		edge_feat = get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=self.k)

 		edge_feat = edge_feat.permute(0,3,1,2)


 		net = self.bn1(F.relu(self.conv1(edge_feat)))
 		net,_ = torch.max(net, dim=-1, keepdim=True)
 		net1 = net


 		net = net.permute(0,2,3,1)

 		dist_mat = pairwise_distance(net)
 		nn_idx = knn(dist_mat, k=self.k)
 		edge_feat = get_edge_feature(net, nn_idx=nn_idx, k=self.k)
 		edge_feat = edge_feat.permute(0,3,1,2)


 		net = self.bn2(F.relu(self.conv2(edge_feat)))
 		net,_ = torch.max(net, dim=-1, keepdim=True)
 		net2 = net

 		net = net.permute(0,2,3,1)

 		dist_mat = pairwise_distance(net)
 		nn_idx = knn(dist_mat, k=self.k)
 		edge_feat = get_edge_feature(net, nn_idx=nn_idx, k=self.k)

 		edge_feat = edge_feat.permute(0,3,1,2)


 		net = self.bn3(F.relu(self.conv3(edge_feat)))
 		net,_ = torch.max(net, dim=-1, keepdim=True)
 		net3 = net

 		net = net.permute(0,2,3,1)


 		dist_mat = pairwise_distance(net)
 		nn_idx = knn(dist_mat, k=self.k)
 		edge_feat = get_edge_feature(net, nn_idx=nn_idx, k=self.k)

 		edge_feat = edge_feat.permute(0,3,1,2)


 		net = self.bn4(F.relu(self.conv4(edge_feat)))
 		net,_ = torch.max(net, dim=-1, keepdim=True)
 		net4 = net
 		# import pdb
 		# pdb.set_trace()

 		net = self.bn5(F.relu(self.conv5(torch.cat((net1, net2, net3, net4), 1))))
 		net,_ = torch.max(net, dim=2, keepdim=True)

 		net = net.view(batch_size, -1)


 		net = self.bn6(F.relu(self.fc1(net)))
 		net = self.dropout(net)
 		net = self.bn7(F.relu(self.fc2(net)))
 		net = self.dropout(net)
 		net = self.fc3(net)

 		return net