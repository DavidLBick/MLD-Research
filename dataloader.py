from loading import * 
import torch
import torch.utils.data as Data
import numpy as np
import pdb
from torch.utils.tensorboard import SummaryWriter
import time


############# CONSTANTS #############
FILE_PATH = "D_trans-D_nsb-5_cb-0_empty-8-5-2-2_lp-150_notch-60-120_beats-head-meas_blinks-head-meas_data_array.npz"
MILLISECONDS = 750
BATCH_SIZE = 32
BATCH_PRINT_INTERVAL = 32
MODEL_PATH = "./saved_models/"
N_EPOCHS = 15
PLOT_GRAD_FLOW = True
NORMALIZE = False

#writer = SummaryWriter(log_dir = './tb_logs/' + str(time.time()))
writer = SummaryWriter(log_dir = './tb_logs/' + 'time_conv/' + str(time.time()))
#####################################

class MEG_Dataset(Data.Dataset):
    def __init__(self, data, index_map, ms, channel):
        super(MEG_Dataset, self).__init__()
        self.data = data
        self.index_map = index_map
        self.ms = ms
        self.channel = channel
        #self.index_map = np.load('index_map.npy')

    def __getitem__(self, index):
        i,j = self.index_map[index]
        word_meg = self.data[i, j, self.channel, self.ms]
        target_idx = i
        target_word = inv_stimulus_order_dict[i]

        return word_meg, target_idx, target_word

    def __len__(self):
        return len(self.index_map)

print("Loading MEG data...")
meg_dict = pickle_load(FILE_PATH)
data = meg_dict['data_array']
stimulus_order_dict = meg_dict['stimulus_order_dict']
inv_stimulus_order_dict = meg_dict['inv_stimulus_order_dict']
question_order_dict = meg_dict['question_order_dict']
inv_question_order_dict = meg_dict['inv_question_order_dict']

train_map = []
for i in range(60):
    for j in range(16):
        train_map.append((i,j))
train_map = np.array(train_map)

test_map = []
for i in range(60):
    for j in range(1):
        test_map.append((i,j))
test_map = np.array(test_map)

training_size = int(0.8*data.shape[1])
split_data = np.split(data, [training_size], 1)

train_data = split_data[0]
# add the dimension of 1 in index 1 because the dataset relies on 
# having shape (words, questions, channels, time) so we need a dummy 
# axis for the questions index
test_data = np.mean(split_data[1], axis=1).reshape(split_data[1].shape[0],
                                                   1,
                                                   split_data[1].shape[2],
                                                   split_data[1].shape[3])

train_loaders = []
test_loaders = []
for ms in range(750):
    for channel in range(306):
        train_dataset = MEG_Dataset(train_data, train_map, ms, channel)
        train_loader = Data.DataLoader(train_dataset, 
                                       batch_size = BATCH_SIZE, 
                                       shuffle = True, 
                                       drop_last = True)
        train_loaders.append((ms,channel,train_loader))

        test_dataset = MEG_Dataset(test_data, test_map, ms, channel)
        test_loader = Data.DataLoader(test_dataset, 
                                   batch_size = BATCH_SIZE, 
                                   shuffle = True, 
                                   drop_last = True)
        test_loaders.append((ms,channel,test_loader))
print("done")
        
