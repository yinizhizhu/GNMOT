import torch.utils.data as data
import cv2, random, torch, shutil, os
import numpy as np
from math import *
from PIL import Image
import torch.nn.functional as F
from mot_model import appearance
from global_set import edge_initial, test_gt_det, tau_dis, app_fine_tune, fine_tune_dir, \
    tau_threshold  # , tau_conf_score
from torchvision.transforms import ToTensor


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class ADatasetFromFolder(data.Dataset):
    def __init__(self, part, part_I, tau, cuda=True, show=0):
        super(ADatasetFromFolder, self).__init__()
        self.dir = part
        self.cleanPath(part)
        self.img_dir = part_I + '/img1/'
        self.gt_dir = part + '/gt/'
        self.det_dir = part + '/det/'
        self.device = torch.device("cuda" if cuda else "cpu")
        self.tau_conf_score = tau
        self.show = show

        self.loadAModel()
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

    def loadAModel(self):
        if app_fine_tune:
            self.Appearance = torch.load(fine_tune_dir)
        else:
            self.Appearance = appearance()
        self.Appearance.to(self.device)
        self.Appearance.eval()  # fixing the BatchN layer

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

    def readBBx_gt(self):
        # get the gt
        self.bbx = [[] for i in range(self.seqL + 1)]
        gt = self.gt_dir + 'gt.txt'
        f = open(gt, 'r')
        pre = -1
        for line in f.readlines():
            line = line.strip().split(',')
            if line[7] == '1':
                index = int(line[0])
                id = int(line[1])
                x, y = int(line[2]), int(line[3])
                w, h = int(line[4]), int(line[5])
                conf_score, l, vr = float(line[6]), int(line[7]), float(line[8])

                # sweep the invisible head-bbx from the training data
                if pre != id and vr == 0:
                    continue

                pre = id
                self.bbx[index].append([x, y, w, h, id, conf_score, vr])
        f.close()

        gt_out = open(self.gt_dir + 'gt_det.txt', 'w')
        for index in range(1, self.seqL + 1):
            for bbx in self.bbx[index]:
                x, y, w, h, id, conf_score, vr = bbx
                print >> gt_out, '%d,-1,%d,%d,%d,%d,%f,-1,-1,-1' % (index, x, y, w, h, conf_score)
        gt_out.close()

    def readBBx_det(self):
        # get the gt
        self.bbx = [[] for i in range(self.seqL + 1)]
        det = self.det_dir + 'det.txt'
        f = open(det, 'r')
        for line in f.readlines():
            line = line.strip().split(',')
            index = int(line[0])
            id = int(line[1])
            x, y = int(float(line[2])), int(float(line[3]))
            w, h = int(float(line[4])), int(float(line[5]))
            conf_score = float(line[6])
            if conf_score >= self.tau_conf_score:
                self.bbx[index].append([x, y, w, h, conf_score])
        f.close()

    def initBuffer(self):
        if self.show:
            cv2.namedWindow('view', flags=0)
            cv2.namedWindow('crop', flags=0)
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

    def fixBB(self, x, y, w, h, size):
        width, height = size
        w = min(w + x, width)
        h = min(h + y, height)
        x = max(x, 0)
        y = max(y, 0)
        w -= x
        h -= y
        return x, y, w, h

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

    def getMN(self, m, n):
        ans = [[None for i in range(n)] for i in range(m)]
        for i in range(m):
            Reframe = self.bbx[self.f_step - self.gap][i]
            for j in range(n):
                GTframe = self.bbx[self.f_step][j]
                p = self.IOU(Reframe, GTframe)
                # 1 - match, 0 - mismatch
                ans[i][j] = torch.FloatTensor([(1 - p) / 100.0, p / 100.0]).to(self.device)
        return ans

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

    def getApp(self, tag, index):
        cur = self.cur if tag else self.nxt
        if torch.is_tensor(index):
            n = index.numel()
            if n < 0:
                print('The tensor is empyt!')
                return None
            if n == 1:
                return self.detections[cur][index[0]][0]
            ans = torch.cat((self.detections[cur][index[0]][0], self.detections[cur][index[1]][0]), dim=0)
            for i in range(2, n):
                ans = torch.cat((ans, self.detections[cur][index[i]][0]), dim=0)
            return ans
        return self.detections[cur][index][0]

    def moveApp(self, index):
        self.bbx[self.f_step].append(self.bbx[self.f_step - self.gap][index])  # add the bbx
        self.detections[self.nxt].append(self.detections[self.cur][index])  # add the appearance

    def swapFC(self):
        self.cur = self.cur ^ self.nxt
        self.nxt = self.cur ^ self.nxt
        self.cur = self.cur ^ self.nxt

    def resnet34(self, img):
        bbx = ToTensor()(img)
        bbx = bbx.to(self.device)
        bbx = bbx.view(-1, bbx.size(0), bbx.size(1), bbx.size(2))
        ret = self.Appearance(bbx)
        ret = ret.view(1, -1)
        return ret

    def feature(self, tag=0):
        '''
        Getting the appearance of the detections in current frame
        :param tag: 1 - initiating
        :param show: 1 - show the cropped & src image
        :return: None
        '''
        apps = []
        with torch.no_grad():
            bbx_container = []
            for bbx in self.bbx[self.f_step]:
                """
                Bellow Conditions needed be taken into consideration:
                    x, y < 0 and x+w > W, y+h > H
                """
                img = load_img(self.img_dir + '%06d.jpg' % self.f_step)  # initial with loading the first frame
                if test_gt_det:
                    x, y, w, h, id, conf_score, vr = bbx
                else:
                    x, y, w, h, conf_score = bbx
                x, y, w, h = self.fixBB(x, y, w, h, img.size)
                if test_gt_det:
                    bbx_container.append([x, y, w, h, id, conf_score, vr])
                else:
                    bbx_container.append([x, y, w, h, conf_score])
                crop = img.crop([x, y, x + w, y + h])
                bbx = crop.resize((224, 224), Image.ANTIALIAS)
                ret = self.resnet34(bbx)
                app = ret.data
                apps.append([app, conf_score])

                if self.show:
                    img = np.asarray(img)
                    crop = np.asarray(crop)
                    if test_gt_det:
                        print('%06d' % self.f_step, id, vr, '***', )
                    else:
                        print('%06d' % self.f_step, conf_score, vr, '***', )
                    print(w, h, '-', )
                    print(len(crop[0]), len(crop))
                    cv2.imshow('crop', crop)
                    cv2.imshow('view', img)
                    cv2.waitKey(34)
                    input('Continue?')
                    # cv2.waitKey(34)
            self.bbx[self.f_step] = bbx_container
        if tag:
            self.detections[self.cur] = apps
        else:
            self.detections[self.nxt] = apps

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

        for i in range(self.m):
            for j in range(self.n):
                e = self.edges[i][j]
                self.candidates.append([e, i, j])

        # print '     The index of the next frame', self.f_step, len(self.bbx)
        return self.gap

    def __getitem__(self, index):
        return self.candidates[index]

    def __len__(self):
        return len(self.candidates)
