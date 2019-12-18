# from __future__ import print_function
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataset import DatasetFromFolder
import time, random, os, shutil
from munkres import Munkres
from global_set import u_initial, mot_dataset_dir, model_dir
from mot_model import *
from tensorboardX import SummaryWriter

torch.manual_seed(123)
np.random.seed(123)


def deleteDir(del_dir):
    shutil.rmtree(del_dir)


class GN():
    def __init__(self, lr=1e-5, batchs=8, cuda=True):
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
        self.Vphi = vphi().to(self.device)
        self.Ephi1 = ephi().to(self.device)
        self.Ephi2 = ephi().to(self.device)

        self.criterion = nn.MSELoss() if criterion_s else nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)

        self.criterion_v = nn.MSELoss().to(self.device)

        self.optimizer1 = optim.Adam([
            {'params': self.Ephi1.parameters()}],
            lr=lr)
        self.optimizer2 = optim.Adam([
            {'params': self.Uphi.parameters()},
            {'params': self.Vphi.parameters()},
            {'params': self.Ephi2.parameters()}],
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

    def getEdges(self):  # the statistic data of the graph among two frames' detections
        self.train_set.setBuffer(1)
        step = 1
        edge_counter = 0.0
        for head in range(1, self.train_test):
            self.train_set.loadNext()  # Get the next frame
            edge_counter += self.train_set.m * self.train_set.n
            step += 1
            self.train_set.swapFC()

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
        edge_counter = 0.0
        for head in range(1, self.train_test):
            self.train_set.loadNext()  # Get the next frame
            edge_counter += self.train_set.m * self.train_set.n
            # print('         Step -', step)
            data_loader = DataLoader(dataset=self.train_set, num_workers=self.numWorker, batch_size=self.batchsize,
                                     shuffle=True)

            for epoch_i in range(1, self.nEpochs):
                num = 0
                epoch_loss_i = 0.0
                for iteration in enumerate(data_loader, 1):
                    index, (e, gt, vs_index, vr_index) = iteration
                    e = e.to(self.device)
                    gt = gt.to(self.device)
                    vs = self.train_set.getApp(1, vs_index)
                    vr = self.train_set.getApp(0, vr_index)

                    self.optimizer1.zero_grad()
                    e1 = self.Ephi1(e, vs, vr, self.u)
                    # update the Ephi1
                    loss = self.criterion(e1, gt.squeeze(1))
                    loss.backward()
                    self.optimizer1.step()
                    num += self.batchsize
                if epoch_loss_i / num < self.loss_threhold:
                    break
            # print('         Updating the Ephi1: %d times.' % epoch_i)

            for epoch in range(1, self.nEpochs):
                num = 0
                epoch_loss = 0.0
                v_loss = 0.0
                arpha_loss = 0.0

                candidates = []
                E_CON, V_CON = [], []
                for iteration in enumerate(data_loader, 1):
                    index, (e, gt, vs_index, vr_index) = iteration
                    e = e.to(self.device)
                    gt = gt.to(self.device)
                    vs = self.train_set.getApp(1, vs_index)
                    vr = self.train_set.getApp(0, vr_index)

                    e1 = self.Ephi1(e, vs, vr, self.u)

                    e1 = e1.data
                    vr1 = self.Vphi(e1, vs, vr, self.u)
                    candidates.append((e1, gt, vs, vr, vr1))
                    E_CON.append(torch.mean(e1, 0))
                    V_CON.append(torch.mean(vs, 0))
                    V_CON.append(torch.mean(vr1.data, 0))

                E = self.train_set.aggregate(E_CON).view(1, -1)  # This part is the aggregation for Edge
                V = self.train_set.aggregate(V_CON).view(1, -1)  # This part is the aggregation for vertex
                # print E.view(1, -1)

                n = len(candidates)
                for i in range(n):
                    e1, gt, vs, vr, vr1 = candidates[i]
                    tmp_gt = 1 - torch.FloatTensor(gt.cpu().numpy()).to(self.device)

                    self.optimizer2.zero_grad()

                    u1 = self.Uphi(E, V, self.u)
                    e2 = self.Ephi2(e1, vs, vr1, u1)

                    # Penalize the u to let its value not too big
                    arpha = torch.mean(torch.abs(u1))
                    arpha_loss += arpha.item()
                    arpha.backward(retain_graph=True)

                    v_l = self.criterion_v(tmp_gt * vr, tmp_gt * vr1)
                    v_loss += v_l.item()
                    v_l.backward(retain_graph=True)

                    #  The regular loss
                    loss = self.criterion(e2, gt.squeeze(1))
                    epoch_loss += loss.item()
                    loss.backward()

                    # update the network: Uphi and Ephi
                    self.optimizer2.step()

                    num += e1.size()[0]

                if self.show_process and self.step_input:
                    a = input('Continue(0-step, 1-run, 2-run with showing)?')
                    if a == '1':
                        self.show_process = 0
                    elif a == '2':
                        self.step_input = 0

                epoch_loss /= num

                # print('         Loss of epoch {}: {}.'.format(epoch, epoch_loss))
                self.writer.add_scalar(seqName, epoch_loss, loss_step)
                loss_step += 1
                if epoch_loss < self.loss_threhold:
                    break

            self.updateUVE()
            step += 1
            self.train_set.swapFC()

    def saveModel(self):
        print('Saving the Uphi model...')
        torch.save(self.Uphi, model_dir + 'uphi_%02d.pth' % self.seq_index)
        print('Saving the Vphi model...')
        torch.save(self.Vphi, model_dir + 'vphi_%02d.pth' % self.seq_index)
        print('Saving the Ephi1 model...')
        torch.save(self.Ephi1, model_dir + 'ephi1_%02d.pth' % self.seq_index)
        print('Saving the Ephi model...')
        torch.save(self.Ephi2, model_dir + 'ephi2_%02d.pth' % self.seq_index)
        print('Saving the global variable u...')
        torch.save(self.u, model_dir + 'u_%02d.pth' % self.seq_index)
        print('Done!')

    def updateUVE(self):
        with torch.no_grad():
            candidates = []
            E_CON, V_CON = [], []
            for edge in self.train_set:
                e, gt, vs_index, vr_index = edge
                e = e.view(1, -1).to(self.device)
                vs = self.train_set.getApp(1, vs_index)
                vr = self.train_set.getApp(0, vr_index)

                e1 = self.Ephi1(e, vs, vr, self.u)
                vr1 = self.Vphi(e1, vs, vr, self.u)
                candidates.append((e1, gt, vs, vr1, vs_index, vr_index))
                E_CON.append(e1)
                V_CON.append(vs)
                V_CON.append(vr1)

            E = self.train_set.aggregate(E_CON).view(1, -1)  # This part is the aggregation for Edge
            V = self.train_set.aggregate(V_CON).view(1, -1)  # This part is the aggregation for vertex
            u1 = self.Uphi(E, V, self.u)
            self.u = u1.data

            nxt = self.train_set.nxt
            for iteration in candidates:
                e1, gt, vs, vr1, vs_index, vr_index = iteration
                e2 = self.Ephi2(e1, vs, vr1, u1)
                if gt.item():
                    self.train_set.detections[nxt][vr_index][0] = vr1.data
                self.train_set.edges[vs_index][vr_index] = e2.data.view(-1)

    def update(self, seqName):
        print('    Train the model with the sequence: ', seqName)
        self.updateNetwork(seqName)
        self.saveModel()


if __name__ == '__main__':
    try:
        deleteDir(model_dir)
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
