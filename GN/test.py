# from __future__ import print_function
import numpy as np
from m_mot_model import *
from munkres import Munkres
import torch.nn.functional as F
import time, os, shutil, random
from global_set import test_gt_det, \
    tau_threshold, gap, f_gap, show_recovering, decay, u_update, mot_dataset_dir, model_dir
from mot_model import appearance
from test_dataset import ADatasetFromFolder
from m_test_dataset import MDatasetFromFolder

torch.manual_seed(123)
np.random.seed(123)


def deleteDir(del_dir):
    shutil.rmtree(del_dir)


year = 17

type = ''  # detection method
t_dir = ''  # the dir of tracking results
sequence_dir = ''  # the dir of the testing video sequence

seqs = [2, 4, 5, 9, 10, 11, 13]  # the set of sequences
lengths = [600, 1050, 837, 525, 654, 900, 750]  # the length of the sequence

test_seqs = [1, 3, 6, 7, 8, 12, 14]
test_lengths = [450, 1500, 1194, 500, 625, 900, 750]

tt_tag = 1  # 1 - test, 0 - train

tau_conf_score = 0.0


class GN():
    def __init__(self, seq_index, seq_len, cuda=True):
        '''
        Evaluating with the MotMetrics
        :param seq_index: the number of the sequence
        :param seq_len: the length of the sequence
        :param cuda: True - GPU, False - CPU
        '''
        self.bbx_counter = 0
        self.seq_index = seq_index
        self.hungarian = Munkres()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.seq_len = seq_len
        self.alpha = 0.3
        self.missingCounter = 0
        self.sideConnection = 0

        print('     Loading the model...')
        self.loadAModel()
        self.loadMModel()

        self.out_dir = t_dir + 'motmetrics_%s/' % (type)

        print(self.out_dir)
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        else:
            deleteDir(self.out_dir)
            os.mkdir(self.out_dir)
        self.initOut()

    def initOut(self):
        print('     Loading Data...')
        self.a_train_set = ADatasetFromFolder(sequence_dir, mot_dataset_dir + 'MOT16/test/MOT16-%02d' % self.seq_index,
                                              tau_conf_score)
        self.m_train_set = MDatasetFromFolder(sequence_dir, mot_dataset_dir + 'MOT16/test/MOT16-%02d' % self.seq_index,
                                              tau_conf_score)

        detection_dir = self.out_dir + 'res_det.txt'
        res_training = self.out_dir + 'res.txt'  # the tracking results
        self.createTxt(detection_dir)
        self.createTxt(res_training)
        self.copyLines(self.seq_index, 1, detection_dir, self.seq_len, 1)

        self.evaluation(1, self.seq_len, detection_dir, res_training)

    def getSeqL(self, info):
        # get the length of the sequence
        f = open(info, 'r')
        f.readline()
        for line in f.readlines():
            line = line.strip().split('=')
            if line[0] == 'seqLength':
                seqL = int(line[1])
        f.close()
        return seqL

    def copyLines(self, seq, head, gt_seq, tail=-1, tag=0):
        '''
        Copy the groun truth within [head, head+num]
        :param seq: the number of the sequence
        :param head: the head frame number
        :param tail: the number the clipped sequence
        :param gt_seq: the dir of the output file
        :return: None
        '''
        if tt_tag:
            basic_dir = mot_dataset_dir + 'MOT%d/test/MOT%d-%02d-%s/' % (year, year, seq, type)
        else:
            basic_dir = mot_dataset_dir + 'MOT%d/train/MOT%d-%02d-%s/' % (year, year, seq, type)
        print('     Testing on', basic_dir, 'Length:', self.seq_len)
        seqL = tail if tail != -1 else self.getSeqL(basic_dir + 'seqinfo.ini')

        det_dir = 'gt/gt_det.txt' if test_gt_det else 'det/det.txt'
        seq_dir = basic_dir + ('gt/gt.txt' if tag == 0 else det_dir)
        inStream = open(seq_dir, 'r')

        outStream = open(gt_seq, 'w')
        for line in inStream.readlines():
            line = line.strip()
            attrs = line.split(',')
            f_num = int(attrs[0])
            if f_num >= head and f_num <= seqL:
                outStream.write(line + '\n')
        outStream.close()

        inStream.close()
        return seqL

    def createTxt(self, out_file):
        f = open(out_file, 'w')
        f.close()

    def loadAModel(self):
        from mot_model import uphi, ephi, vphi
        tail = 13
        self.AUphi = torch.load('../App2/' + model_dir + 'uphi_%02d.pth' % tail).to(self.device)
        self.AVphi = torch.load('../App2/' + model_dir + 'vphi_%02d.pth' % tail).to(self.device)
        self.AEphi1 = torch.load('../App2/' + model_dir + 'ephi1_%02d.pth' % tail).to(self.device)
        self.AEphi2 = torch.load('../App2/' + model_dir + 'ephi2_%02d.pth' % tail).to(self.device)
        self.Au = torch.load('../App2/' + model_dir + 'u_%02d.pth' % tail).to(self.device)

    def loadMModel(self):
        from m_mot_model import uphi, ephi
        tail = 13
        self.MUphi = torch.load('../Motion1/' + model_dir + 'uphi_%d.pth' % tail).to(self.device)
        self.MEphi = torch.load('../Motion1/' + model_dir + 'ephi_%d.pth' % tail).to(self.device)
        self.Mu = torch.load('../Motion1/' + model_dir + 'u_%d.pth' % tail).to(self.device)

    def swapFC(self):
        self.cur = self.cur ^ self.nxt
        self.nxt = self.cur ^ self.nxt
        self.cur = self.cur ^ self.nxt

    def linearModel(self, out, attr1, attr2):
        # print 'I got you! *.*'
        t = attr1[-1]
        self.sideConnection += 1
        if t > f_gap:
            return
        frame = int(attr1[0])
        x1, y1, w1, h1 = float(attr1[2]), float(attr1[3]), float(attr1[4]), float(attr1[5])
        x2, y2, w2, h2 = float(attr2[2]), float(attr2[3]), float(attr2[4]), float(attr2[5])

        x_delta = (x2 - x1) / t
        y_delta = (y2 - y1) / t
        w_delta = (w2 - w1) / t
        h_delta = (h2 - h1) / t

        for i in range(1, t):
            frame += 1
            x1 += x_delta
            y1 += y_delta
            w1 += w_delta
            h1 += h_delta
            attr1[0] = str(frame)
            attr1[2] = str(x1)
            attr1[3] = str(y1)
            attr1[4] = str(w1)
            attr1[5] = str(h1)
            line = ''
            for attr in attr1[:-1]:
                line += attr + ','
            if show_recovering:
                line += '1'
            else:
                line = line[:-1]
            out.write(line + '\n')
            self.bbx_counter += 1
        self.missingCounter += t - 1

    def evaluation(self, head, tail, gtFile, outFile):
        '''
        Evaluation on dets
        :param head: the head frame number
        :param tail: the tail frame number
        :param gtFile: the ground truth file name
        :param outFile: the name of output file
        :return: None
        '''
        gtIn = open(gtFile, 'r')
        self.cur, self.nxt = 0, 1
        line_con = [[], []]
        id_con = [[], []]
        id_step = 1

        a_step = head + self.a_train_set.setBuffer(head)
        m_step = head + self.m_train_set.setBuffer(head)
        if a_step != m_step:
            print('Something is wrong!')
            print('a_step =', a_step, ', m_step =', m_step)
            input('Continue?')

        while a_step < tail:
            # print '*********************************'
            a_t_gap = self.a_train_set.loadNext()
            m_t_gap = self.m_train_set.loadNext()
            if a_t_gap != m_t_gap:
                print('Something is wrong!')
                print('a_t_gap =', a_t_gap, ', m_t_gap =', m_t_gap)
                input('Continue?')
            a_step += a_t_gap
            m_step += m_step
            print(a_step, end=' ')
            if a_step % 100 == 0:
                print('')

            m_u_ = self.MUphi(self.m_train_set.E, self.m_train_set.V, self.Mu)

            # print 'Fo'
            a_m = self.a_train_set.m
            a_n = self.a_train_set.n
            m_m = self.m_train_set.m
            m_n = self.m_train_set.n

            if a_m != m_m or a_n != m_n:
                print('Something is wrong!')
                print('a_m = %d, m_m = %d' % (a_m, m_m), ', a_n = %d, m_n = %d' % (a_n, m_n))
                input('Continue?')
            # print 'm = %d, n = %d'%(m, n)
            if a_n == 0:
                print('There is no detection in the rest of sequence!')
                break

            if id_step == 1:
                out = open(outFile, 'a')
                i = 0
                while i < a_m:
                    attrs = gtIn.readline().strip().split(',')
                    if float(attrs[6]) >= tau_conf_score:
                        attrs.append(1)
                        attrs[1] = str(id_step)
                        line = ''
                        for attr in attrs[:-1]:
                            line += attr + ','
                        if show_recovering:
                            line += '0'
                        else:
                            line = line[:-1]
                        out.write(line + '\n')
                        self.bbx_counter += 1
                        line_con[self.cur].append(attrs)
                        id_con[self.cur].append(id_step)
                        id_step += 1
                        i += 1
                out.close()

            i = 0
            while i < a_n:
                attrs = gtIn.readline().strip().split(',')
                if float(attrs[6]) >= tau_conf_score:
                    attrs.append(1)
                    line_con[self.nxt].append(attrs)
                    id_con[self.nxt].append(-1)
                    i += 1

            # update the edges
            # print 'T',
            candidates = []
            E_CON, V_CON = [], []
            for edge in self.a_train_set.candidates:
                e, vs_index, vr_index = edge
                e = e.view(1, -1).to(self.device)
                vs = self.a_train_set.getApp(1, vs_index)
                vr = self.a_train_set.getApp(0, vr_index)

                e1 = self.AEphi1(e, vs, vr, self.Au)
                vr1 = self.AVphi(e1, vs, vr, self.Au)
                candidates.append((e1, vs, vr1, vs_index, vr_index))
                E_CON.append(e1)
                V_CON.append(vs)
                V_CON.append(vr1)

            E = self.a_train_set.aggregate(E_CON).view(1, -1)
            V = self.a_train_set.aggregate(V_CON).view(1, -1)
            u1 = self.AUphi(E, V, self.Au)

            ret = self.a_train_set.getRet()
            decay_tag = [0 for i in range(a_m)]
            for i in range(a_m):
                for j in range(a_n):
                    if ret[i][j] == 0:
                        decay_tag[i] += 1

            for i in range(len(self.a_train_set.candidates)):
                e1, vs, vr1, a_vs_index, a_vr_index = candidates[i]
                m_e, m_vs_index, m_vr_index = self.m_train_set.candidates[i]
                if a_vs_index != m_vs_index or a_vr_index != m_vr_index:
                    print('Something is wrong!')
                    print('a_vs_index = %d, m_vs_index = %d' % (a_vs_index, m_vs_index))
                    print('a_vr_index = %d, m_vr_index = %d' % (a_vr_index, m_vr_index))
                    input('Continue?')
                if ret[a_vs_index][a_vr_index] == tau_threshold:
                    continue

                e2 = self.AEphi2(e1, vs, vr1, u1)
                self.a_train_set.edges[a_vs_index][a_vr_index] = e1.data.view(-1)

                a_tmp = F.softmax(e2)
                a_tmp = a_tmp.cpu().data.numpy()[0]

                m_e = m_e.to(self.device).view(1, -1)
                m_v1 = self.m_train_set.getMotion(1, m_vs_index)
                m_v2 = self.m_train_set.getMotion(0, m_vr_index, m_vs_index,
                                                  line_con[self.cur][m_vs_index][-1] + a_t_gap - 1)
                m_e_ = self.MEphi(m_e, m_v1, m_v2, m_u_)
                self.m_train_set.edges[m_vs_index][m_vr_index] = m_e_.data.view(-1)
                m_tmp = F.softmax(m_e_)
                m_tmp = m_tmp.cpu().data.numpy()[0]

                t = line_con[self.cur][a_vs_index][-1]
                if decay_tag[a_vs_index] > 0:
                    A = min(float(a_tmp[0]) * pow(decay, t - 1), 1.0)
                    M = min(float(m_tmp[0]) * pow(decay, t - 1), 1.0)
                else:
                    A = float(a_tmp[0])
                    M = float(m_tmp[0])
                ret[a_vs_index][a_vr_index] = A * self.alpha + M * (1 - self.alpha)

            # for j in ret:
            #     print j
            results = self.hungarian.compute(ret)

            out = open(outFile, 'a')
            look_up = set(j for j in range(a_n))
            nxt = self.a_train_set.nxt
            for (i, j) in results:
                # print (i,j)
                if ret[i][j] >= tau_threshold:
                    continue
                e1 = self.a_train_set.edges[i][j].view(1, -1).to(self.device)
                vs = self.a_train_set.getApp(1, i)
                vr = self.a_train_set.getApp(0, j)

                vr1 = self.AVphi(e1, vs, vr, self.Au)
                self.a_train_set.detections[nxt][j][0] = vr1.data

                look_up.remove(j)
                self.m_train_set.updateVelocity(i, j, line_con[self.cur][i][-1], False)

                id = id_con[self.cur][i]
                id_con[self.nxt][j] = id
                attr1 = line_con[self.cur][i]
                attr2 = line_con[self.nxt][j]
                # print attrs
                attr2[1] = str(id)
                if attr1[-1] + a_t_gap - 1 > 1:
                    # for the missing detections
                    self.linearModel(out, attr1, attr2)
                line = ''
                for attr in attr2[:-1]:
                    line += attr + ','
                if show_recovering:
                    line += '0'
                else:
                    line = line[:-1]
                out.write(line + '\n')
                self.bbx_counter += 1

            if u_update:
                self.Mu = m_u_.data
                self.Au = u1.data

            for j in look_up:
                self.m_train_set.updateVelocity(-1, j, tag=False)

            for i in range(a_n):
                if id_con[self.nxt][i] == -1:
                    id_con[self.nxt][i] = id_step
                    attrs = line_con[self.nxt][i]
                    attrs[1] = str(id_step)
                    line = ''
                    for attr in attrs[:-1]:
                        line += attr + ','
                    if show_recovering:
                        line += '0'
                    else:
                        line = line[:-1]
                    out.write(line + '\n')
                    self.bbx_counter += 1
                    id_step += 1
            out.close()

            # For missing & Occlusion
            index = 0
            for (i, j) in results:
                while i != index:
                    attrs = line_con[self.cur][index]
                    # print '*', attrs, '*'
                    if attrs[-1] + a_t_gap <= gap:
                        attrs[-1] += a_t_gap
                        line_con[self.nxt].append(attrs)
                        id_con[self.nxt].append(id_con[self.cur][index])
                        self.a_train_set.moveApp(index)
                        self.m_train_set.moveMotion(index)
                    index += 1
                if ret[i][j] >= tau_threshold:
                    attrs = line_con[self.cur][index]
                    # print '*', attrs, '*'
                    if attrs[-1] + a_t_gap <= gap:
                        attrs[-1] += a_t_gap
                        line_con[self.nxt].append(attrs)
                        id_con[self.nxt].append(id_con[self.cur][index])
                        self.a_train_set.moveApp(index)
                        self.m_train_set.moveMotion(index)
                index += 1
            while index < a_m:
                attrs = line_con[self.cur][index]
                # print '*', attrs, '*'
                if attrs[-1] + a_t_gap <= gap:
                    attrs[-1] += a_t_gap
                    line_con[self.nxt].append(attrs)
                    id_con[self.nxt].append(id_con[self.cur][index])
                    self.a_train_set.moveApp(index)
                    self.m_train_set.moveMotion(index)
                index += 1

            # con = self.m_train_set.cleanEdge()
            # for i in range(len(con)-1, -1, -1):
            #     index = con[i]
            #     del line_con[self.nxt][index]
            #     del id_con[self.nxt][index]

            line_con[self.cur] = []
            id_con[self.cur] = []
            # print head+step, results
            self.a_train_set.swapFC()
            self.m_train_set.swapFC()
            self.swapFC()
        gtIn.close()
        print('     The results:', id_step, self.bbx_counter)


if __name__ == '__main__':
    try:
        types = [['POI', 0.7]]
        # types = [['DPM', -0.6], ['SDP', 0.5], ['FRCNN', 0.5]]

        for t in types:
            type, tau_conf_score = t
            head = time.time()

            f_dir = 'results/'
            if not os.path.exists(f_dir):
                os.mkdir(f_dir)

            for i in range(len(seqs)):
                if tt_tag:
                    seq_index = test_seqs[i]
                    seq_len = test_lengths[i]
                else:
                    seq_index = seqs[i]
                    seq_len = lengths[i]

                print('The sequence:', seq_index, '- The length of the training data:', seq_len)

                s_dir = f_dir + '%02d/' % seq_index
                if not os.path.exists(s_dir):
                    os.mkdir(s_dir)
                    print(s_dir, 'does not exist!')

                t_dir = s_dir + '%d/' % seq_len
                if not os.path.exists(t_dir):
                    os.mkdir(t_dir)
                    print(t_dir, 'does not exist!')

                if tt_tag:
                    seq_dir = 'MOT%d-%02d-%s' % (year, test_seqs[i], type)
                    sequence_dir = mot_dataset_dir + 'MOT%d/test/' % year + seq_dir
                    print(' ', sequence_dir)

                    start = time.time()
                    print('     Evaluating Graph Network...')
                    gn = GN(test_seqs[i], test_lengths[i])
                else:
                    seq_dir = 'MOT%d-%02d-%s' % (year, seqs[i], type)
                    sequence_dir = mot_dataset_dir + 'MOT%d/train/' % year + seq_dir
                    print(' ', sequence_dir)

                    start = time.time()
                    print('     Evaluating Graph Network...')
                    gn = GN(seqs[i], lengths[i])
                print('     Recover the number missing detections:', gn.missingCounter)
                print('     The number of sideConnections:', gn.sideConnection)
                print('Time consuming:', (time.time() - start) / 60.0)
            print('Time consuming:', (time.time() - head) / 60.0)
    except KeyboardInterrupt:
        print('Time consuming:', time.time() - start)
        print('')
        print('-' * 90)
        print('Existing from training early.')
