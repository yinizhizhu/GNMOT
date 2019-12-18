import torch.utils.data as data
import random, torch, shutil, os, gc
from math import *
from PIL import Image
import torch.nn.functional as F
from global_set import edge_initial, test_gt_det, tau_dis, tau_threshold  # , tau_conf_score


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class MDatasetFromFolder(data.Dataset):
    def __init__(self, part, part_I, tau, cuda=True):
        super(MDatasetFromFolder, self).__init__()
        self.dir = part
        self.cleanPath(part)
        self.img_dir = part_I + '/img1/'
        self.gt_dir = part + '/gt/'
        self.det_dir = part + '/det/'
        self.device = torch.device("cuda" if cuda else "cpu")
        self.tau_conf_score = tau

        self.getSeqL()
        if test_gt_det:
            self.readBBx_gt()
        else:
            self.readBBx_det()
        self.initBuffer()

    def cleanPath(self, part):
        if os.path.exists(part + '/gts/'):
            shutil.rmtree(part + '/gts/')
        if os.path.exists(part + '/dets/'):
            shutil.rmtree(part + '/dets/')

    def getSeqL(self):
        # get the length of the sequence
        info = self.dir + '/seqinfo.ini'
        f = open(info, 'r')
        f.readline()
        for line in f.readlines():
            line = line.strip().split('=')
            if line[0] == 'seqLength':
                self.seqL = int(line[1])
        f.close()
        # print 'The length of the sequence:', self.seqL

    def fixBB(self, x, y, w, h, size):
        width, height = size
        w = min(w + x, width)
        h = min(h + y, height)
        x = max(x, 0)
        y = max(y, 0)
        w -= x
        h -= y
        return x, y, w, h

    def readBBx_gt(self):
        # get the gt
        self.bbx = [[] for i in range(self.seqL + 1)]
        bbxs = [[] for i in range(self.seqL + 1)]
        imgs = [None for i in range(self.seqL + 1)]
        for i in range(1, self.seqL + 1):
            imgs[i] = load_img(self.img_dir + '%06d.jpg' % i)
        gt = self.gt_dir + 'gt.txt'
        f = open(gt, 'r')
        pre = -1
        for line in f.readlines():
            line = line.strip().split(',')
            if line[7] == '1':
                index = int(line[0])
                id = int(line[1])
                x, y = float(line[2]), float(line[3])
                w, h = float(line[4]), float(line[5])
                conf_score, l, vr = float(line[6]), int(line[7]), float(line[8])

                # sweep the invisible head-bbx from the training data
                if pre != id and vr == 0:
                    continue

                pre = id
                img = imgs[index]
                x, y, w, h = self.fixBB(x, y, w, h, img.size)
                width, height = float(img.size[0]), float(img.size[1])
                self.bbx[index].append([x / width, y / height, w / width, h / height, id, conf_score, vr])
                bbxs[index].append([x, y, w, h, id, conf_score, vr])
        f.close()

        gt_out = open(self.gt_dir + 'gt_det.txt', 'w')
        for index in range(1, self.seqL + 1):
            for bbx in bbxs[index]:
                x, y, w, h, id, conf_score, vr = bbx
                print >> gt_out, '%d,-1,%d,%d,%d,%d,%f,-1,-1,-1' % (index, x, y, w, h, conf_score)
        gt_out.close()

    def readBBx_det(self):
        # get the gt
        self.bbx = [[] for i in range(self.seqL + 1)]
        imgs = [None for i in range(self.seqL + 1)]
        for i in range(1, self.seqL + 1):
            imgs[i] = load_img(self.img_dir + '%06d.jpg' % i)
        det = self.det_dir + 'det.txt'
        f = open(det, 'r')
        for line in f.readlines():
            line = line.strip().split(',')
            index = int(line[0])
            id = int(line[1])
            x, y = float(line[2]), float(line[3])
            w, h = float(line[4]), float(line[5])
            conf_score = float(line[6])
            if conf_score >= self.tau_conf_score:
                img = imgs[i]
                x, y, w, h = self.fixBB(x, y, w, h, img.size)
                width, height = float(img.size[0]), float(img.size[1])
                self.bbx[index].append([x / width, y / height, w / width, h / height, id, conf_score])
        f.close()

    def initBuffer(self):
        self.f_step = 1  # the index of next frame in the process
        self.cur = 0  # the index of current frame in the detections
        self.nxt = 1  # the index of next frame in the detections
        self.detections = [None, None]  # the buffer to storing images: current & next frame
        self.feature(1)

    def setBuffer(self, f):
        self.m = 0
        counter = -1
        while self.m == 0:
            counter += 1
            self.f_step = f + counter
            self.feature(1)
            self.m = len(self.detections[self.cur])
        if counter > 0:
            print('           Empty in setBuffer:', counter)
        return counter

    def IOU(self, Reframe, GTframe):
        """
        Compute the Intersection of Union
        :param Reframe: x, y, w, h
        :param GTframe: x, y, w, h
        :return: Ratio
        """
        if edge_initial == 1:
            return random.random()
        elif edge_initial == 3:
            return 0.5
        x1 = Reframe[0]
        y1 = Reframe[1]
        width1 = Reframe[2]
        height1 = Reframe[3]

        x2 = GTframe[0]
        y2 = GTframe[1]
        width2 = GTframe[2]
        height2 = GTframe[3]

        endx = max(x1 + width1, x2 + width2)
        startx = min(x1, x2)
        width = width1 + width2 - (endx - startx)

        endy = max(y1 + height1, y2 + height2)
        starty = min(y1, y2)
        height = height1 + height2 - (endy - starty)

        if width <= 0 or height <= 0:
            ratio = 0
        else:
            Area = width * height
            Area1 = width1 * height1
            Area2 = width2 * height2
            ratio = Area * 1. / (Area1 + Area2 - Area)
        return ratio

    def aggregate(self, set):
        if len(set):
            rho = sum(set)
            return rho / len(set)
        print('     The set is empty!')
        return None

    def distance(self, a_bbx, b_bbx):
        w = min(float(a_bbx[2]) * tau_dis, float(b_bbx[2]) * tau_dis)
        dx = float(a_bbx[0] + a_bbx[2] / 2) - float(b_bbx[0] + b_bbx[2] / 2)
        dy = float(a_bbx[1] + a_bbx[3] / 2) - float(b_bbx[1] + b_bbx[3] / 2)
        d = sqrt(dx * dx + dy * dy)
        if d <= w:
            return 0.0
        return tau_threshold

    def getRet(self):
        cur = self.f_step - self.gap
        ret = [[0.0 for i in range(self.n)] for j in range(self.m)]
        for i in range(self.m):
            bbx1 = self.bbx[cur][i]
            for j in range(self.n):
                ret[i][j] = self.distance(bbx1, self.bbx[self.f_step][j])
        return ret

    def getMotion(self, tag, index, pre_index=None, t=None):
        cur = self.cur if tag else self.nxt
        if tag == 0:
            self.updateVelocity(pre_index, index, t)
            return self.detections[cur][index][0][pre_index]
        return self.detections[cur][index][0][0]

    def moveMotion(self, index):
        self.bbx[self.f_step].append(self.bbx[self.f_step - self.gap][index])  # add the bbx: x, y, w, h, id, conf_score
        self.detections[self.nxt].append(
            self.detections[self.cur][index])  # add the motion: [[x, y, w, h, v_x, v_y], id]

    def cleanEdge(self):
        con = []
        index = 0
        for det in self.detections[self.nxt]:
            motion, id = det
            x = motion[0][0].item() + motion[0][4].item()
            y = motion[0][1].item() + motion[0][5].item()
            if (x < 0.0 or x > 1.0) or (y < 0.0 or y > 1.0):
                con.append(index)
            index += 1

        for i in range(len(con) - 1, -1, -1):
            index = con[i]
            del self.bbx[self.f_step][index]
            del self.detections[self.nxt][index]
        return con

    def swapFC(self):
        self.cur = self.cur ^ self.nxt
        self.nxt = self.cur ^ self.nxt
        self.cur = self.cur ^ self.nxt

    def updateVelocity(self, i, j, t=None, tag=True):
        v_x = 0.0
        v_y = 0.0
        if i != -1:
            if test_gt_det:
                x1, y1, w1, h1, id1, conf_score1, vr1 = self.bbx[self.f_step - self.gap][i]
                x2, y2, w2, h2, id2, conf_score2, vr2 = self.bbx[self.f_step][j]
            else:
                x1, y1, w1, h1, id1, conf_score1 = self.bbx[self.f_step - self.gap][i]
                x2, y2, w2, h2, id2, conf_score2 = self.bbx[self.f_step][j]
            v_x = (x2 + w2 / 2 - (x1 + w1 / 2)) / t
            v_y = (y2 + h2 / 2 - (y1 + h1 / 2)) / t
        if tag:
            # print 'm=%d,n=%d; i=%d, j=%d'%(len(self.detections[self.cur]), len(self.detections[self.nxt]), i, j)
            self.detections[self.nxt][j][0][i][0][4] = v_x
            self.detections[self.nxt][j][0][i][0][5] = v_y
        else:
            cur_m = self.detections[self.nxt][j][0][0]
            cur_m[0][4] = v_x
            cur_m[0][5] = v_y
            self.detections[self.nxt][j][0] = [cur_m]

    def getMN(self, m, n):
        cur = self.f_step - self.gap
        ans = [[None for i in range(n)] for i in range(m)]
        for i in range(m):
            Reframe = self.bbx[cur][i]
            for j in range(n):
                GTframe = self.bbx[self.f_step][j]
                p = self.IOU(Reframe, GTframe)
                # 1 - match, 0 - mismatch
                ans[i][j] = torch.FloatTensor([(1 - p) / 100.0, p / 100.0])
        return ans

    def feature(self, tag=0):
        '''
        Getting the appearance of the detections in current frame
        :param tag: 1 - initiating
        :param show: 1 - show the cropped & src image
        :return: None
        '''
        motions = []
        with torch.no_grad():
            m = 1 if tag else self.m
            for bbx in self.bbx[self.f_step]:
                """
                Bellow Conditions needed be taken into consideration:
                    x, y < 0 and x+w > W, y+h > H
                """
                if test_gt_det:
                    x, y, w, h, id, conf_score, vr = bbx
                else:
                    x, y, w, h, id, conf_score = bbx
                cur_m = []
                for i in range(m):
                    cur_m.append(torch.FloatTensor([[x, y, w, h, 0.0, 0.0]]).to(self.device))
                motions.append([cur_m, id])
        if tag:
            self.detections[self.cur] = motions
        else:
            self.detections[self.nxt] = motions

    def loadNext(self):
        self.m = len(self.detections[self.cur])

        self.gap = 0
        self.n = 0
        while self.n == 0:
            self.f_step += 1
            self.feature()
            self.n = len(self.detections[self.nxt])
            self.gap += 1

        if self.gap > 1:
            print('           Empty in loadNext:', self.f_step - self.gap + 1, '-', self.gap - 1)

        self.candidates = []
        self.edges = self.getMN(self.m, self.n)

        es = []
        # vs_index = 0
        for i in range(self.m):
            # vr_index = self.m
            for j in range(self.n):
                e = self.edges[i][j]
                es.append(e)
                self.candidates.append([e, i, j])
            #     vr_index += 1
            # vs_index += 1

        vs = []
        for i in range(2):
            n = len(self.detections[i])
            for j in range(n):
                v = self.detections[i][j][0][0]
                vs.append(v)

        self.E = self.aggregate(es).to(self.device).view(1, -1)
        self.V = self.aggregate(vs).to(self.device)

        # print '     The index of the next frame', self.f_step, len(self.bbx)
        return self.gap

    def __getitem__(self, index):
        return self.candidates[index]

    def __len__(self):
        return len(self.candidates)
