import load_dataset
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from classification_net import classification_net

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILES = load_dataset.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = load_dataset.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
train_file_idxs = np.arange(0, len(TRAIN_FILES))
current_data, current_label = load_dataset.loadDataFile(TRAIN_FILES[train_file_idxs[0]])
# print TRAIN_FILES[train_file_idxs[0]]
# print current_data.shape, np.max(current_label)

MAX_EPOCHS = 1
BATCH_SIZE = 2
NUM_POINT = 2048

model = classification_net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(MAX_EPOCHS):


    for fn in range(len(TRAIN_FILES)):
        current_data, current_label = load_dataset.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:,0:NUM_POINT,:]
        current_data, current_label, _ = load_dataset.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch in range(num_batches):
            start_idx = batch * BATCH_SIZE
            end_idx = (batch+1) * BATCH_SIZE

            rotated_data = load_dataset.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = load_dataset.jitter_point_cloud(rotated_data)
            jittered_data = load_dataset.random_scale_point_cloud(jittered_data)
            jittered_data = load_dataset.rotate_perturbation_point_cloud(jittered_data)
            jittered_data = load_dataset.shift_point_cloud(jittered_data)

            jittered_data = Variable(torch.from_numpy(jittered_data))
            labels = Variable(torch.from_numpy(current_label[start_idx:end_idx]).long())

            optimizer.zero_grad()

            out_labels = model(jittered_data)

            loss = loss_fn(out_labels, labels)

            loss.backward()

            optimizer.step()

            print("Done")




