# from __future__ import print_function
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataset import DatasetFromFolder
import time, random, os, shutil
from munkres import Munkres
from global_set import u_initial, mot_dataset_dir, model_dir
from m_mot_model import *
from tensorboardX import SummaryWriter

torch.manual_seed(123)
np.random.seed(123)


def deleteDir(del_dir):
    shutil.rmtree(del_dir)


class GN():
    def __init__(self, lr=5e-4, batchs=8, cuda=True):
        '''
        :param tt: train_test
        :param tag: 1 - evaluation on testing data, 0 - without evaluation on testing data
        :param lr:
        :param batchs:
        :param cuda:
        '''
        # all the tensor should set the 'volatile' as True, and False when update the network
        self.hungarian = Munkres()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.nEpochs = 999
        self.lr = lr
        self.batchsize = batchs
        self.numWorker = 4

        self.show_process = 0  # interaction
        self.step_input = 1

        print('     Preparing the model...')
        self.resetU()

        self.Uphi = uphi().to(self.device)
        self.Ephi = ephi().to(self.device)

        self.criterion = nn.MSELoss() if criterion_s else nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)

        self.optimizer = optim.Adam([
            {'params': self.Uphi.parameters()},
            {'params': self.Ephi.parameters()}],
            lr=lr)

        self.writer = SummaryWriter()

        seqs = [2, 4, 5, 9, 10, 11, 13]
        lengths = [600, 1050, 837, 525, 654, 900, 750]

        for i in range(len(seqs)):
            # print '     Loading Data...'
            seq = seqs[i]
            self.seq_index = seq
            start = time.time()
            sequence_dir = mot_dataset_dir + 'MOT16/train/MOT16-%02d' % seq

            self.train_set = DatasetFromFolder(sequence_dir)

            self.train_test = lengths[i]
            # self.train_test = int(self.train_test * 0.8)  # For training the model without the validation set

            self.loss_threhold = 0.03
            self.update('seq/%02d' % seq)

    def resetU(self):
        if u_initial:
            self.u = torch.FloatTensor([random.random() for i in range(u_num)]).view(1, -1)
        else:
            self.u = torch.FloatTensor([0.0 for i in range(u_num)]).view(1, -1)
        self.u = self.u.to(self.device)

    def updateNetwork(self, seqName):
        self.train_set.setBuffer(1)
        step = 1
        loss_step = 0
        average_epoch = 0
        edge_counter = 0.0
        for head in range(1, self.train_test):
            self.train_set.loadNext()  # Get the next frame
            edge_counter += self.train_set.m * self.train_set.n
            start = time.time()
            # print('         Step -', step)
            data_loader = DataLoader(dataset=self.train_set, num_workers=self.numWorker, batch_size=self.batchsize,
                                     shuffle=True)
            for epoch in range(1, self.nEpochs):
                num = 0
                epoch_loss = 0.0
                arpha_loss = 0.0
                for iteration in enumerate(data_loader, 1):
                    index, (e, gt, vs_index, vr_index) = iteration
                    e = e.to(self.device)
                    gt = gt.to(self.device)

                    self.optimizer.zero_grad()

                    u_ = self.Uphi(self.train_set.E, self.train_set.V, self.u)
                    v1 = self.train_set.getMotion(1, vs_index)
                    v2 = self.train_set.getMotion(0, vr_index, vs_index)
                    e_ = self.Ephi(e, v1, v2, u_)

                    if self.show_process:
                        print('-' * 66)
                        print(vs_index, vr_index)
                        print('e:', e.cpu().data.numpy()[0][0], end=' ')
                        print('e_:', e_.cpu().data.numpy()[0][0], end=' ')
                        if criterion_s:
                            print('GT:', gt.cpu().data.numpy()[0][0])
                        else:
                            print('GT:', gt.cpu().data.numpy()[0])

                    # Penalize the u to let its value not too big
                    arpha = torch.mean(torch.abs(u_))
                    arpha_loss += arpha.item()
                    arpha.backward(retain_graph=True)

                    #  The regular loss
                    # print e_.size(), e_
                    # print gt.size(), gt
                    loss = self.criterion(e_, gt.squeeze(1))
                    epoch_loss += loss.item()
                    loss.backward()

                    # update the network: Uphi and Ephi
                    self.optimizer.step()

                    #  Show the parameters of the Uphi and Ephi to check the process of optimiser
                    # print self.Uphi.features[0].weight.data
                    # print self.Ephi.features[0].weight.data
                    # raw_input('continue?')

                    num += e.size()[0]

                if self.show_process and self.step_input:
                    a = input('Continue(0-step, 1-run, 2-run with showing)?')
                    if a == '1':
                        self.show_process = 0
                    elif a == '2':
                        self.step_input = 0

                epoch_loss /= num

                self.writer.add_scalar(seqName, epoch_loss, loss_step)  # Show the training loss
                loss_step += 1
                # print('         Loss of epoch {}: {}.'.format(epoch, epoch_loss))
                if epoch_loss < self.loss_threhold:
                    break

            # print('         Time consuming:{}\n\n'.format(time.time() - start))
            self.updateUE()
            average_epoch += epoch
            step += 1
            self.train_set.swapFC()

    def saveModel(self):
        print('Saving the Uphi model...')
        torch.save(self.Uphi, model_dir + 'uphi_%02d.pth' % self.seq_index)
        print('Saving the Ephi model...')
        torch.save(self.Ephi, model_dir + 'ephi_%02d.pth' % self.seq_index)
        print('Saving the global variable u...')
        torch.save(self.u, model_dir + 'u_%02d.pth' % self.seq_index)
        print('Done!')

    def updateUE(self):
        u_ = self.Uphi(self.train_set.E, self.train_set.V, self.u)

        self.u = u_.data

        # update the edges
        for edge in self.train_set:
            e, gt, vs_index, vr_index = edge
            e = e.to(self.device).view(1, -1)
            v1 = self.train_set.getMotion(1, vs_index)
            v2 = self.train_set.getMotion(0, vr_index, vs_index)
            e_ = self.Ephi(e, v1, v2, u_)
            self.train_set.edges[vs_index][vr_index] = e_.data.view(-1)

    def update(self, seqName):
        print('    Train the model with the sequence: ', seqName)
        self.updateNetwork(seqName)
        self.saveModel()


if __name__ == '__main__':
    try:
        # deleteDir(model_dir)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
            start = time.time()
            print('     Starting Graph Network...')
            gn = GN()
            print('Time consuming:', time.time() - start)
        else:
            # deleteDir(model_dir)
            # os.mkdir(model_dir)
            print('The model has been here!')

    except KeyboardInterrupt:
        print('Time consuming:', time.time() - start)
        print('')
        print('-' * 90)
        print('Existing from training early.')
