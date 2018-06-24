import load_dataset
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import json
from time import gmtime, strftime

from part_seg_net import part_seg_net

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

hdf5_data_dir = os.path.join(BASE_DIR, './hdf5_data')

use_cuda = torch.cuda.is_available()
Float = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
Long = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Int = torch.cuda.IntTensor if use_cuda else torch.IntTensor
Double = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

all_obj_cats_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
fin = open(all_obj_cats_file, 'r')
lines = [line.rstrip() for line in fin.readlines()]
all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
fin.close()

all_cats = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))
NUM_CATEGORIES = 16
NUM_PART_CATS = len(all_cats)

MAX_EPOCHS = 50
BATCH_SIZE = 16
NUM_POINT = 2048

TRAINING_FILE_LIST = os.path.join(hdf5_data_dir, 'train_hdf5_file_list.txt')
TESTING_FILE_LIST = os.path.join(hdf5_data_dir, 'val_hdf5_file_list.txt')
TRAIN_FILES = load_dataset.getDataFiles(TRAINING_FILE_LIST)
# num_train_file = len(train_file_list)
test_file_list = load_dataset.getDataFiles(TESTING_FILE_LIST)
num_test_file = len(test_file_list)

def convert_label_to_one_hot(labels):
  label_one_hot = np.zeros((labels.shape[0], NUM_CATEGORIES))
  for idx in range(labels.shape[0]):
    label_one_hot[idx, labels[idx]] = 1
  return label_one_hot

model = part_seg_net(NUM_PART_CATS)
loss_fn = nn.NLLLoss2d()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
if not os.path.exists(RESULTS_DIR):
  os.mkdir(RESULTS_DIR)

exp_date = strftime("%b_%d_%H_%M", gmtime())
file_name = 'results/part_seg/exp_date'

if use_cuda:
    model.cuda()

model.train()
for epoch in range(MAX_EPOCHS):

    train_file_idx = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idx)


    for i in range(len(TRAIN_FILES)):
        cur_train_filename = os.path.join(hdf5_data_dir, TRAIN_FILES[train_file_idx[i]])
        cur_data, cur_labels, cur_seg = load_dataset.load_h5_data_label_seg(cur_train_filename)
        cur_data, cur_labels, order = load_dataset.shuffle_data(cur_data, np.squeeze(cur_labels))
        cur_seg = cur_seg[order, ...]

        cur_labels_one_hot = convert_label_to_one_hot(cur_labels)
        
        file_size = len(cur_labels)
        num_batches = file_size // BATCH_SIZE
        
        total_loss = 0.0
        total_seg_acc = 0.0

        for batch in range(num_batches):
            start_idx = batch * BATCH_SIZE
            end_idx = (batch+1) * BATCH_SIZE

            point_clouds = Variable(torch.from_numpy(cur_data[start_idx: end_idx, ...]))
            object_labels = Variable(torch.from_numpy(cur_labels_one_hot[start_idx: end_idx, ...]))
            true_part_labels = Variable(torch.from_numpy(cur_seg[start_idx: end_idx, ...])).unsqueeze(-1)

            if use_cuda:
                point_clouds = point_clouds.cuda()
                object_labels = object_labels.cuda()
                true_part_labels = true_part_labels.cuda()

            optimizer.zero_grad()

            part_label_probs = model(point_clouds, object_labels)
            _,part_labels = torch.max(part_label_probs, dim=1)

            acc = torch.sum(part_labels==true_part_labels)/float(BATCH_SIZE*NUM_POINT)

            loss = loss_fn(part_label_probs, true_part_labels)

            total_loss += loss.data[0]
            total_seg_acc += acc.data[0]


            loss.backward()

            optimizer.step()

    with open(file_name, 'a') as fp:
        fp.write(
            'Epoch: [{}/{}], '
            'Loss: {}, \n'
            'Train Acc: {}, \n'.format(
                epoch + 1, MAX_EPOCHS, total_loss/float(num_batches), \
                total_seg_acc/float(num_batches)))

    torch.save(model.state_dict(), '{}.pkl'.format(file_name))
    torch.save(model, '{}.pt'.format(file_name))

    