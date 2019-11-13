import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from math import floor
from dataloader import *
import matplotlib.pyplot as plt


##################
### SIMPLE CNN ###
##################

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)

# using pytorch documentation formula
def calc_L_out(L_in, padding, kernel_size, stride, dilation=1):
    num = L_in + (2 * padding) - dilation * (kernel_size - 1) - 1
    frac = num / stride
    out = frac + 1
    L_out = floor(out)
    return L_out

# given all the parameters such as padding kernel_size, etc. 
# figure out the feature dimension of the last layer to automate
# entry of the in hidden size for the final linear classification layer 
def get_last_L_out(params):
    init_L_in = MILLISECONDS
    L_out = None
    for i in range(len(params)):
        L_in = L_out if L_out != None else init_L_in
        L_out = calc_L_out(L_in, params[i][4], params[i][2], 
                               params[i][3])
    return L_out

class Time_Conv(nn.Module):
    def __init__(self, num_classes):
        super(Time_Conv, self).__init__()
        params = [[306, 16, 3, 2, 1],
                  [16,   4, 1, 2, 1] ]

        self.embedding_model = nn.Sequential(
            nn.Conv1d(in_channels = params[0][0], 
                out_channels = params[0][1], 
                kernel_size = params[0][2],
                stride=params[0][3], padding=params[0][4], bias=True), 
            nn.ReLU(), 
            nn.BatchNorm1d(16), 
            nn.Conv1d(in_channels = params[1][0], 
                out_channels = params[1][1], 
                kernel_size = params[1][2],
                stride=params[1][3], padding=params[1][4], bias=True), 
            Flatten()     
            )

        # calculate last length and multiply by the last out channels
        # because we are flattening it
        in_size = get_last_L_out(params) * params[-1][1]
        self.classification_model = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(in_size, num_classes)
        )


    def forward(self, x, embedding=False):
        convolved = self.embedding_model(x)
        y = self.classification_model(convolved)
        return y


class Space_Conv(nn.Module):
    def __init__(self, num_classes):
        super(Space_Conv, self).__init__()
        def get_last_L_out_time(params):
            init_L_in = 306
            L_out = None
            for i in range(len(params)):
                L_in = L_out if L_out != None else init_L_in
                L_out = calc_L_out(L_in, params[i][4], params[i][2], 
                                       params[i][3])
            return L_out

        params = [[MILLISECONDS, 750, 3, 2, 1]]

        self.embedding_model = nn.Sequential(
            nn.Conv1d(in_channels = params[0][0], 
                out_channels = params[0][1], 
                kernel_size = params[0][2],
                stride=params[0][3], padding=params[0][4], bias=True),   
            Flatten()     
            )

        # calculate last length and multiply by the last out channels
        # because we are flattening it
        in_size = get_last_L_out_time(params) * params[-1][1]

        self.classification_model = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(in_size, num_classes)
            )    

    def forward(self, x, embedding=False):
        if embedding:
            convolved = self.embedding_model(x)
            return convolved

        else:
            convolved = self.embedding_model(x)
            return self.classification_model(convolved)


class Time_Space_Conv(nn.Module):
    def __init__(self, num_classes):
        super(Time_Space_Conv, self).__init__()

        def get_last_L_out_2d(params):
            init_L_in = CHANNELS*MILLISECONDS
            L_out = None
            for i in range(len(params)):
                L_in = L_out if L_out != None else init_L_in
                L_out = calc_L_out(L_in, params[i][4], params[i][2],
                                        params[i][3])
            return L_out

        params = [[1, 16, (15,3), 2, 1]]

        self.embedding_model = nn.Sequential(
            nn.Conv2d(in_channels = params[0][0],
                out_channels = params[0][1],
                kernel_size = params[0][2],
                stride = params[0][3],
                padding = params[0][4], bias=True),
            Flatten()
            )


        in_size = 882000

        self.classification_model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_size, num_classes)
            )

    def forward(self, x, embedding=False):
        print('SHAPE: ' + str(x.shape))
        x = x.unsqueeze(1)
        if embedding:
            convolved = self.embedding_model(x)
            return convolved

        else:
            convolved = self.embedding_model(x)
            return self.classification_model(convolved)


def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find("Linear") is -1:
        m.weight.data.uniform_(-1/480, 1/480)
        m.bias.data.fill_(0)

class Logistic_Regression(nn.Module):
    def __init__(self, num_classes):
        super(Logistic_Regression, self).__init__()
        MEG_CHANNELS = 306
        self.logreg = nn.Linear(MEG_CHANNELS*MILLISECONDS, 60)
        self.logreg.apply(weights_init_uniform)        

    def forward(self, x, i):
        batch_logits = self.logreg(x.view(BATCH_SIZE, -1))
        first_row_logits = self.logreg(x.view(BATCH_SIZE, -1)[0, :])
        
        logits_fig = plt.figure()
        plt.imshow(batch_logits.detach().numpy())
        writer.add_figure(tag = 'Logits', 
                          figure = logits_fig, 
                          global_step = i)

        for j in range(BATCH_SIZE):
            plot_i = plt.figure()
            plt.imshow(x[j].numpy())
            writer.add_figure("MEG_Scan", 
                             plot_i, 
                             i)
        return batch_logits
        



