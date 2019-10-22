from loading import * 
import torch
import torch.utils.data as Data
import numpy as np
import pdb
from torch.utils.tensorboard import SummaryWriter
import time


############# CONSTANTS #############
FILE_PATH = "G_trans-D_nsb-5_cb-0_empty-8-5-2-2_lp-150_notch-60-120_beats-head-meas_blinks-head-meas_data_array.npz"
MILLISECONDS = 750
BATCH_SIZE = 32
BATCH_PRINT_INTERVAL = 1
MODEL_PATH = "./saved_models/"
N_EPOCHS = 20
PLOT_GRAD_FLOW = True
NORMALIZE = True

#writer = SummaryWriter(log_dir = './tb_logs/' + str(time.time()))
writer = SummaryWriter(log_dir = './tb_logs/' + "embeddings")
#####################################

class MEG_Dataset(Data.Dataset):
    def __init__(self, data):
        super(MEG_Dataset, self).__init__()
        self.data = data
        self.index_map = np.load('index_map.npy')

    def __getitem__(self, index):
        i,j = self.index_map[index]
        word_meg = self.data[i, j, :, :MILLISECONDS]
        target_idx = i
        target_word = inv_stimulus_order_dict[i]

        return word_meg, target_idx, target_word

    def __len__(self):
        return len(self.index_map)

print("Loading MEG data...")
meg_dict = pickle_load(FILE_PATH)
data = meg_dict['data_array']
pdb.set_trace()
stimulus_order_dict = meg_dict['stimulus_order_dict']
inv_stimulus_order_dict = meg_dict['inv_stimulus_order_dict']
question_order_dict = meg_dict['question_order_dict']
inv_question_order_dict = meg_dict['inv_question_order_dict']

train_dataset = MEG_Dataset(data)
train_loader = Data.DataLoader(train_dataset, 
                               batch_size = BATCH_SIZE, 
                               shuffle = True, 
                               drop_last = True)
print("done")
        
