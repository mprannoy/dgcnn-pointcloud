import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as grad

from utils import *
from transform_net import input_transform_net

class part_seg_net(nn.Module):
    """docstring for part_seg_net"""
    def __init__(self, part_num, k=30, cat_num=16):
        super(part_seg_net, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv7 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv8 = nn.Conv2d(192, 1024, kernel_size=1)
        self.conv9 = nn.Conv2d(cat_num, 128, kernel_size=1)

        self.conv10 = nn.Conv2d(2752, 256, kernel_size=1)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv12 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv13 = nn.Conv2d(128, part_num, kernel_size=1)

        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(64)
        self.bn8 = nn.BatchNorm2d(1024)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(256)
        self.bn11 = nn.BatchNorm2d(256)
        self.bn12 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(p=0.6)

        self.input_transform = input_transform_net()

        self.k = k
        self.part_num = part_num
        self.cat_num = cat_num

    def forward(self, point_cloud, object_label):

        batch_size, num_point,_ = point_cloud.size()
        input_img = point_cloud.unsqueeze(-1)

        self.mp1 = nn.MaxPool2d((num_point, 1), stride=2)

        dist_mat = pairwise_distance(point_cloud)
        nn_idx = knn(dist_mat, k=self.k)
        edge_feat = get_edge_feature(input_img, nn_idx=nn_idx, k=self.k)
        edge_feat = edge_feat.permute(0,3,1,2)
        # point_cloud = point_cloud.permute(0,2,1)

        transform_mat = self.input_transform(edge_feat)

        point_cloud_transformed = torch.bmm(point_cloud, transform_mat)
        input_img = point_cloud_transformed.unsqueeze(-1)
        
        dist_mat = pairwise_distance(point_cloud_transformed)
        nn_idx = knn(dist_mat, k=self.k)
        edge_feat = get_edge_feature(input_img, nn_idx=nn_idx, k=self.k)
        edge_feat = edge_feat.permute(0,3,1,2)


        out1 = self.bn1(F.relu(self.conv1(edge_feat)))
        out1 = self.bn2(F.relu(self.conv2(out1)))
        out_max1,_ = torch.max(out1, dim=-1, keepdim=True)
        out_mean1 = torch.mean(out1, dim=-1, keepdim=True)

        out3 = self.bn3(F.relu(self.conv3(torch.cat((out_max1, out_mean1), dim=1))))

        out = out3.permute(0,2,3,1)
        dist_mat = pairwise_distance(out)
        nn_idx = knn(dist_mat, k=self.k)
        edge_feat = get_edge_feature(out, nn_idx=nn_idx, k=self.k)
        edge_feat = edge_feat.permute(0,3,1,2)

        out = self.bn4(F.relu(self.conv4(edge_feat)))
        out_max2,_ = torch.max(out, dim=-1, keepdim=True)
        out_mean2 = torch.mean(out, dim=-1, keepdim=True)

        out5 = self.bn5(F.relu(self.conv5(torch.cat((out_max2, out_mean2), dim=1))))

        out = out5.permute(0,2,3,1)
        dist_mat = pairwise_distance(torch.squeeze(out, dim=-2))
        nn_idx = knn(dist_mat, k=self.k)
        edge_feat = get_edge_feature(out, nn_idx=nn_idx, k=self.k)
        edge_feat = edge_feat.permute(0,3,1,2)

        out = self.bn6(F.relu(self.conv6(edge_feat)))
        out_max3,_ = torch.max(out, dim=-1, keepdim=True)
        out_mean3 = torch.mean(out, dim=-1, keepdim=True)
        out7 = self.bn7(F.relu(self.conv7(torch.cat((out_max3, out_mean3), dim=1))))

        out8 = self.bn8(F.relu(self.conv8(torch.cat((out3, out5, out7), dim=1))))

        out_max = self.mp1(out8)

        one_hot_label_expand = object_label.view(batch_size, self.cat_num, 1, 1)
        one_hot_label_expand = self.bn9(F.relu(self.conv9(one_hot_label_expand)))
        out_max = torch.cat((out_max, one_hot_label_expand), dim=1)
        out_max = out_max.expand(-1,-1,num_point,-1)

        concat = torch.cat((out_max, out_max1, out_mean1,
                            out3, out_max2, out_mean2,
                            out5, out_max3, out_mean3,
                            out7, out8), dim=1)

        net2 = self.bn10(F.relu(self.conv10(concat)))
        net2 = self.dropout(net2)
        net2 = self.bn11(F.relu(self.conv11(net2)))
        net2 = self.dropout(net2)
        net2 = self.bn12(F.relu(self.conv12(net2)))
        net2 = self.conv13(net2)

        net2 = net2.view(batch_size, self.part_num, num_point, 1)
        net2 = F.log_softmax(net2, dim=1)


        return net2