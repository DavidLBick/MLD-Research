import torch
import matplotlib
matplotlib.use('TKAgg')
from dataloader import *
import pdb
from model import *
import time
import matplotlib.pyplot as plt
plt.ion()
from matplotlib.lines import Line2D
import sys
import functools

np.random.seed(10)

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
                batch_loss, running_loss, batch_accuracy,
                correct, total):
    print("Stats: batch %d" % batch_idx)
    print("Time: ", after - before)
    print("Batch loss", batch_loss)
    print("Running loss", running_loss)
    print("Batch accuracy", batch_accuracy)
    print("Total Correct", correct)
    print("Total Running Examples", total)
    print('\n')
    return

def enumerate2dList(L):
    enumd = []
    for arr_idx in range(len(L)):
        enumd.append([])
        for i,x in enumerate(L[arr_idx]):
            enumd[arr_idx].append((i,x))
        enumd[arr_idx].sort(key=lambda x: x[1])
    return np.array(enumd, dtype=[('f1', np.uint64), ('f2', float)])

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

    def test(self):
        print('TESTING...')
        correct, epoch_loss, total = 0., 0., 0.

        before = time.time()
        print(len(test_loader), "batches of size", self.batch_size)
        for batch_idx, (data, label, label_word) in enumerate(test_loader):
            # data --> MEG scan
            # label --> index of word
            # label_word --> actual word (just in case is useful)
            if self.gpu:
                data = data.float().cuda()
                label = label.cuda()

            else: data = data.float();

            self.optimizer.zero_grad()

            if NORMALIZE:
                # normalizing each channel by subtracting channel mean
                # and dividing by channel st dev
                means = torch.mean(data, dim = 1).view(BATCH_SIZE, 1, MILLISECONDS)
                sds = torch.std(data, dim = 1).view(BATCH_SIZE, 1, MILLISECONDS)
                data = (data - means) / sds

            # data = data.permute(0, 2, 1)
            out = self.model(data)

            out = out.view(self.batch_size, -1)
            loss = self.criterion(out, label)
            epoch_loss += loss

            # NOTE: potential speed-up by not moving to numpy
            forward_res = out.detach().cpu().numpy()
            labels = label.detach().cpu().numpy()
            forward_res = enumerate2dList(forward_res)

            batch_correct = 0
            for i,label in enumerate(labels):
                for j,x in enumerate(forward_res[i]):
                    word,_ = x
                    if word == label:
                        batch_correct += j/60
                        break;
            print('batch_correct: ' + str(batch_correct))
            # predictions = np.argmax(forward_res, axis=1)

            # batch_correct = np.sum((predictions == labels))
            correct += batch_correct
            total += data.size(0)

            if batch_idx % BATCH_PRINT_INTERVAL == 0:
                after = time.time()
                print('Test stats')
                print_stats(batch_idx, after, before,
                            0, 0,
                            float(batch_correct / self.batch_size),
                            correct,
                            total)
                before = after

        print('Done testing.')
        return float(correct) / total, epoch_loss / batch_idx


    def train(self, n_epochs, train_loader):
        best_val_acc = None
        for epoch in range(n_epochs):
            print('Epoch #%d' % epoch)
            correct, epoch_loss, total = 0., 0., 0.

            before = time.time()
            print(len(train_loader), "batches of size", self.batch_size)
            for batch_idx, (data, label, label_word) in enumerate(train_loader):
                # data --> MEG scan
                # label --> index of word
                # label_word --> actual word (just in case is useful)
                orig_data = data
                if self.gpu:
                    data = data.float().cuda()
                    label = label.cuda()

                else: data = data.float();

                self.optimizer.zero_grad()

                #writer.add_embedding(mat = data.view(BATCH_SIZE, -1),
                #                     global_step = batch_idx,
                #                     metadata = [word for word in label_word],
                #                     tag = 'Unormalized MEG Vecs')

                if NORMALIZE:
                    # normalizing each channel by subtracting channel mean
                    # and dividing by channel st dev
                    means = torch.mean(data, dim = 1).view(BATCH_SIZE, 1, MILLISECONDS)
                    sds = torch.std(data, dim = 1).view(BATCH_SIZE, 1, MILLISECONDS)
                    data = (data - means) / sds

                #writer.add_embedding(mat = data.view(BATCH_SIZE, -1),
                #                     global_step = batch_idx,
                #                     metadata = [word for word in label_word],
                #                     tag = 'Normalized MEG Vecs')

                # FOR SPACE CONVOLUTION
                # want to have the time steps as the channels
                # data = data.permute(0, 2, 1)
                out = self.model(data)

                out = out.view(self.batch_size, -1)
                loss = self.criterion(out, label)
                loss.backward()

                #plot_grad_flow(self.model.named_parameters())
                #plot_grad_flow_alt(self.model.named_parameters())

                self.optimizer.step()

                epoch_loss += loss.item()

                # NOTE: potential speed-up by not moving to numpy
                forward_res = out.detach().cpu().numpy()
                labels = label.detach().cpu().numpy()
                forward_res = enumerate2dList(forward_res)

                batch_correct = 0
                for i,label in enumerate(labels):
                    for j,x in enumerate(forward_res[i]):
                        word,_ = x
                        if word == label:
                            batch_correct += j/60
                            break;
                # predictions = np.argmax(forward_res, axis=1)

                # batch_correct = np.sum((predictions == labels))
                correct += batch_correct
                total += data.size(0)

                if batch_idx % BATCH_PRINT_INTERVAL == 0:
                    after = time.time()
                    print_stats(batch_idx, after, before,
                                loss.item(), epoch_loss / (batch_idx+1),
                                float(batch_correct / self.batch_size),
                                correct,
                                total)

                    for m in self.model.modules():
                        if isinstance(m, nn.Linear):
                            print('Layer type: ' + str(type(m)))
                            print('Shape: ' + str(m.weight.shape))
                            print()
                            writer.add_histogram("Weights",
                                                 m.weight.data,
                                                 batch_idx)
                    before = after


            val_acc, val_loss = self.test()
            if best_val_acc is None or val_acc > best_val_acc:
                torch.save(self.model, MODEL_PATH +
                                       "time_space_conv_epoch%d.pt" % epoch)
                best_val_acc = val_acc

            writer.add_scalar("Loss/Train",
                              epoch_loss / batch_idx,
                              epoch)
            writer.add_scalar("Accuracy/Train",
                              float(correct) / total,
                              epoch)

            writer.add_scalar("Loss/Test",
                              val_loss,
                              epoch)
            writer.add_scalar("Accuracy/Test",
                              val_acc,
                              epoch)

        return


def get_one_patch_intermed(i, word_scan, weights, filter=0):
    patch = word_scan[0, 0, (15*i):(15*(i+1)),(3*i):(3*(i+1))].flatten()
    filter = weights[filter,0,:,:].flatten()
    intermed = np.dot(patch.numpy(), filter.detach().numpy())
    return intermed


def max_points(activations, new_vals, word_idx):
    if len(activations) == 0:
        activations = [None]*55125
    for i in range(len(new_vals)):
        if activations[i] == None or activations[i][0] < new_vals[i]:
            activations[i] = (new_vals[i], word_idx)
    return activations

def main():
    print("Creating model and optimizer...")
    NUM_WORDS = 60
    #model = Logistic_Regression(NUM_WORDS)
    # model = Time_Space_Conv(NUM_WORDS)

    # print(model)
    # print('-'*60)
    # for l in list(model.named_parameters()):
    #     print(l[0], ':', l[1].detach().numpy().shape)

    model = torch.load("saved_models/time_space_conv_725_epoch7.pt")
    # layers = [child for child in model.children()]
    # embedding_model = layers[0]
    # conv_layer = embedding_model[0]
    # weights = conv_layer.weight
    #
    # plt.figure(1)
    # for i in range(8):
    #     plt.subplot(4, 4, i+1)
    #     plt.imshow(weights[i,:,:].detach().numpy())
    # plt.show()

    # pdb.set_trace()

    x = np.mean(data, axis=1)[:, :, :750]
    word_scan = torch.from_numpy(x).float()
    print('word scan dims: ' + str(word_scan.size()))
    means = torch.mean(word_scan, dim = 1).view(60, 1, MILLISECONDS)
    sds = torch.std(word_scan, dim = 1).view(60, 1, MILLISECONDS)
    word_scan = (word_scan - means) / sds
    word_scan = word_scan.unsqueeze(1)
    print('word scan dims: ' + str(word_scan.size()))
    # pdb.set_trace()
    convolved = model.embedding_model(word_scan)
    all_intermediates = convolved.cpu().detach().numpy()
    print('all_intermediates shape: ' + str(all_intermediates.shape))
    all_weights = model.classification_model[1].weight.data.numpy()
    intermediates_times_weights = np.multiply(all_intermediates, all_weights)
    print('convolved shape: ' + str(convolved.size()))

    NUM_FILTERS = 16
    filters = [[] for i in range(NUM_FILTERS)]
    activations = [[] for i in range(NUM_FILTERS)]
    for word_idx in range(60):
        intermediate_vals = convolved[word_idx].cpu().detach().numpy()
        for i in range(len(filters)):
            filters[i] = intermediate_vals[55125*i : 55125*(i+1)]

        ret = []

        filter_vals = [intermediates_times_weights[word_idx][55125*i : 55125*(i+1)] for i in range(len(filters))]
        # f0_vals = intermediates_times_weights[word_idx][:55125]
        # f1_vals = intermediates_times_weights[word_idx][55125:]

        # activations = [max_points(activations[i], filter_vals[i], word_idx) for i in range(len(filters))]
        # filter_0_activations = max_points(filter_0_activations, f0_vals, word_idx)
        # filter_1_activations = max_points(filter_1_activations, f1_vals, word_idx)

        # filter_0_activations.append((word_idx, np.sum(f0_vals)))
        # filter_1_activations.append((word_idx, np.sum(f1_vals)))

        for i in range(55125):
            curr_max = None
            max_filter = None
            for j in range(len(filters)):
                if curr_max == None or filter_vals[j][i] > curr_max:
                    curr_max = filter_vals[j][i]
                    max_filter = j
            ret.append(max_filter)

            # ret.append(0 if f0_vals[i] > f1_vals[i] else 1)
        fig = plt.figure(1)
        ret = np.reshape(ret, (147, 375))
        plt.imshow(ret)
        fig.savefig('filter_per_pt_plots/'  + str(word_idx) + '.png')

        # fig0 = plt.figure(1)
        # filter_0_vals = np.reshape(filter_0_vals, (147, 375))
        # plt.imshow(filter_0_vals)
        # fig0.savefig('intermediate_plots/' + str(word_idx) + '_fig0.png')
        #
        # fig1 = plt.figure(1)
        # filter_1_vals = np.reshape(filter_1_vals, (147, 375))
        # plt.imshow(filter_1_vals)
        # fig1.savefig('intermediate_plots/' + str(word_idx) + '_fig1.png')

    # for i in range(len(filters)):
    #     activations[i] = list(map(lambda x: x[1], activations[i]))
    #     fig = plt.figure(1)
    #     filter_vals = np.reshape(activations[i], (147, 375))
    #     plt.imshow(filter_vals)
    #     fig.savefig('word_per_filter_pt_plots/filter' + str(i) + '.png')

    pdb.set_trace()
    return ret
    sys.exit(0)
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
