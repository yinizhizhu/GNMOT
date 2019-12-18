import torch.utils.data as data
import cv2, random, torch, shutil, os
import numpy as np
from PIL import Image
import torch.nn.functional as F
from mot_model import appearance
from global_set import edge_initial, app_fine_tune, fine_tune_dir, overlap
from torchvision.transforms import ToTensor


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, part, cuda=True, show=0):
        super(DatasetFromFolder, self).__init__()
        self.dir = part
        self.cleanPath(part)
        self.img_dir = part + '/img1/'
        self.gt_dir = part + '/gt/'
        self.det_dir = part + '/det/'
        self.device = torch.device("cuda" if cuda else "cpu")
        self.show = show

        self.loadAModel()
        self.getSeqL()
        self.readBBx()
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
        print('     The length of the sequence:', self.seqL)

    def generator(self, bbx):
        if random.randint(0, 1):
            x, y, w, h = bbx
            x, y, w = float(x), float(y), float(w),
            tmp = overlap * 2 / (1 + overlap)
            n_w = random.uniform(tmp * w, w)
            n_h = tmp * w * float(h) / n_w

            direction = random.randint(1, 4)
            if direction == 1:
                x = x + n_w - w
                y = y + n_h - h
            elif direction == 2:
                x = x - n_w + w
                y = y + n_h - h
            elif direction == 3:
                x = x + n_w - w
                y = y - n_h + h
            else:
                x = x - n_w + w
                y = y - n_h + h
            ans = [int(x), int(y), int(w), h]
            return ans
        return bbx

    def readBBx(self):
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
                x, y, w, h = self.generator([x, y, w, h])
                self.bbx[index].append([x, y, w, h, id, vr])
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
        self.f_step = f
        self.feature(1)

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
            Reframe = self.bbx[self.f_step - 1][i]
            for j in range(n):
                GTframe = self.bbx[self.f_step][j]
                p = self.IOU(Reframe, GTframe)
                # 1 - match, 0 - mismatch
                ans[i][j] = torch.FloatTensor([(1 - p) / 100.0, p / 100.0])
        return ans

    def aggregate(self, set):
        if len(set):
            rho = sum(set)
            return rho / len(set)
        print('     The set is empty!')
        return None

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
            img = load_img(self.img_dir + '%06d.jpg' % self.f_step)  # initial with loading the first frame
            for bbx in self.bbx[self.f_step]:
                """
                Condition needed be taken into consideration:
                    x, y < 0 and x+w > W, y+h > H
                """
                x, y, w, h, id, vr = bbx
                x, y, w, h = self.fixBB(x, y, w, h, img.size)
                bbx_container.append([x, y, w, h, id, vr])
                crop = img.crop([x, y, x + w, y + h])
                bbx = crop.resize((224, 224), Image.ANTIALIAS)
                ret = self.resnet34(bbx)
                app = ret.data
                apps.append([app, id])

                if self.show:
                    img = np.asarray(img)
                    crop = np.asarray(crop)
                    print('%06d' % self.f_step, id, vr, '***', end=' ')
                    print(w, h, '-', end=' ')
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

    def initEC(self):
        self.m = len(self.detections[self.cur])
        self.n = len(self.detections[self.nxt])
        self.candidates = []
        self.edges = self.getMN(self.m, self.n)
        self.gts = [[None for j in range(self.n)] for i in range(self.m)]
        self.step_gt = 0.0
        for i in range(self.m):
            for j in range(self.n):
                tag = int(self.detections[self.cur][i][1] == self.detections[self.nxt][j][1])
                self.gts[i][j] = torch.LongTensor([tag])
                self.step_gt += tag * 1.0

        for i in range(self.m):
            for j in range(self.n):
                e = self.edges[i][j]
                gt = self.gts[i][j]
                self.candidates.append([e, gt, i, j])

    def loadNext(self):
        self.f_step += 1
        self.feature()
        self.initEC()
        # print '     The index of the next frame', self.f_step
        # print self.detections[self.cur]
        # print self.detections[self.nxt]

    def __getitem__(self, index):
        return self.candidates[index]

    def __len__(self):
        return len(self.candidates)
