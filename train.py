import torch
from dataloader import * 
import pdb
from model import * 
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_grad_flow_alt(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig('plots/grads_alt_%d.png' % time.time())
    return 

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('plots/grads_%d.png' % time.time())
    return 

def print_stats(batch_idx, after, before, 
                batch_loss, running_loss, accuracy,
                batch_accuracy):
    print("Stats: batch %d" % batch_idx)
    print("Time: ", after - before)
    print("Batch loss", batch_loss)
    print("Running loss", running_loss)
    print("Batch accuracy", batch_accuracy)
    print("Accuracy", accuracy)
    print('\n')

    return

class Trainer(object):
    def __init__(self, model, optimizer):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.gpu = torch.cuda.is_available()
        self.batch_size = BATCH_SIZE

        if self.gpu:
            print("Model to cuda")
            self.model = self.model.cuda()

    def train(self, n_epochs, train_loader):
        for epoch in range(n_epochs):
            correct, epoch_loss, total = 0., 0., 0.

            before = time.time()
            print(len(train_loader), "batches of size", self.batch_size)
            for batch_idx, (data, label, label_word) in enumerate(train_loader):
                # data --> MEG scan 
                # label --> index of word
                # label_word --> actual word (just in case is useful)
                if self.gpu:
                    data = data.float().cuda()
                    label = label.cuda()

                else: data = data.float(); 

                self.optimizer.zero_grad()

                out = self.model(data)

                out = out.view(self.batch_size, -1)
                loss = self.criterion(out, label)
                loss.backward()

                
                plot_grad_flow(self.model.named_parameters())
                plot_grad_flow_alt(self.model.named_parameters())

                self.optimizer.step()

                epoch_loss += loss.item()

                # NOTE: potential speed-up by not moving to numpy
                forward_res = out.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                predictions = np.argmax(forward_res, axis=1)

                batch_correct = np.sum((predictions == label))
                correct += batch_correct
                total += data.size(0)

                if batch_idx % BATCH_PRINT_INTERVAL == 0:
                    after = time.time()
                    print_stats(batch_idx, after, before, 
                                loss.item(), epoch_loss / (batch_idx+1),  
                                float(batch_correct / self.batch_size), 
                                float(correct / total))
                    before = after

            torch.save(self.model, MODEL_PATH + "epoch%d.pt" % epoch)

        return


def main():
    print("Creating model and optimizer...")
    NUM_WORDS = 60
    model = Simple_Conv1d(NUM_WORDS)
    optim = torch.optim.Adam(model.parameters(), 
                             lr = 1e-3)

    trainer = Trainer(model, optim)
    print("done")

    TRAIN_FLAG = True
    if TRAIN_FLAG:
        print("Begin training...")
        trainer.train(N_EPOCHS, train_loader)

    return 



if __name__ == '__main__':
    main() 