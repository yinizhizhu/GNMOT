# Graph Networks for Multiple Object Tracking (WACV 2020)

## Introduction
This is the official code of '**Graph Networks for Multiple object Tracking**'.
Multiple object tracking (MOT) task requires reasoning the states of all targets and associating these targets in a global way. However, existing MOT methods mostly focus on the local relationship among objects and ignore the global relationship. Some methods formulate the MOT problem as a graph optimization problem. However, these methods are based on static graphs, which are seldom updated. To solve these problems, we design a new near-online MOT method with an end-to-end graph network. Specifically, we design an appearance graph network and a motion graph network to capture the appearance and the motion similarity separately. The updating mechanism is carefully designed in our graph network, which means that nodes, edges and the global variable in the graph can be updated. The global variable can capture the global relationship to help tracking. Finally, a strategy to handle missing detections is proposed to remedy the defect of the detectors. Our method is evaluated on both the MOT16 and the MOT17 benchmarks, and experimental results show the encouraging performance of our method.

<p align='center'>
    <img src="/Pic/Pipeline.png">
</p>


## Main Results
### Results on MOT16 with private detection

| Method             | Detection | MOTA | IDF1 | MT |   ML | FP | FN | IDs | FM |
|--------------------|----------|------------|---------|--------|-------|-------|--------|--------|--------| 
| GN without SOT        | POI  | 58.4      |  54.8  | 27.3%   | 23.2%  | 5731  |  68630  |  1454  |  1730  |

<p align='center'>
    <img src="/Pic/mot.gif">
</p>

## Environment
The code is developed using python 2.7 on Ubuntu 16.04. NVIDIA GPU is needed. The code is developed and tested using GTX 1070 GPU card. Other platforms or GPU cards are not fully tested.

## Quick start
### Installation
1. Install pytorch >= 0.4 following [official instruction](https://pytorch.org/).
2. Clone this repo, and we'll call the directory that you cloned as ${GN_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Your directory tree should look like this:

   ```
   ${GN_ROOT}
   ├── MOT
   ├── App2
   |   ├── model
   ├── Motion1
   |   ├── model
   ├── GN
   ├── output
   ├── ReadMe.md
   └── requirements.txt
   ```
5. Download pretrained models from our model zoo([BaiduPan](https://pan.baidu.com/s/1QlYAd-wi5IdnJCoBc4tr3g)(extract code: um4c))
   ```
   ${GN_ROOT}
    `-- App2
        `-- model
            |-- ephi1_13
            |-- ephi2_13
            |-- u_13
            |-- uphi_13
            |-- vphi_13
    `-- Motion1
        `-- model
            |-- ephi_13
            |-- u_13
            |-- uphi_13

   ```
6. **For MOTChallenge data**, please download from [MOTChallenge Dataset](https://motchallenge.net).
Download and extract them under {GN_ROOT}/MOT, and make them look like this:
   ```
	${GN_ROOT}
	 `-- MOT
	     `-- MOT16
		 |-- train
		 |   |-- MOT16-02
		 |   |   |-- seqinfo.ini
		 |   |   |-- gt
		 |   |   |   |-- gt.txt
		 |   |   |-- det
		 |   |   |   |-- det.txt
		 |   |   |-- img1
		 |   |   |   |-- 000001.jpg
		 |   |   |   |-- 000002.jpg
		 |   |   |   |-- 000003.jpg
		 |   |-- ...
		 |-- test
		 |   |-- MOT16-01
		 |   |   |-- seqinfo.ini
		 |   |   |-- det
		 |   |   |   |-- det.txt
		 |   |   |-- img1
		 |   |   |   |-- 000001.jpg
		 |   |   |   |-- 000002.jpg
		 |   |   |   |-- 000003.jpg
		 |   |-- ...
	     `-- MOT17
		 |-- train
		 |   |-- MOT17-02-DPM
		 |   |   |-- seqinfo.ini
		 |   |   |-- gt
		 |   |   |   |-- gt.txt
		 |   |   |-- det
		 |   |   |   |-- det.txt
		 |   |-- MOT17-02-FRCNN
		 |   |   |-- seqinfo.ini
		 |   |   |-- gt
		 |   |   |   |-- gt.txt
		 |   |   |-- det
		 |   |   |   |-- det.txt
		 |   |-- MOT17-02-SDP
		 |   |   |-- seqinfo.ini
		 |   |   |-- gt
		 |   |   |   |-- gt.txt
		 |   |   |-- det
		 |   |   |   |-- det.txt
		 |   |-- MOT17-02-POI
		 |   |   |-- seqinfo.ini
		 |   |   |-- gt
		 |   |   |   |-- gt.txt
		 |   |   |-- det
		 |   |   |   |-- det.txt
		 |   |-- ...
		 |-- test
		 |   |-- MOT17-01-DPM
		 |   |   |-- seqinfo.ini
		 |   |   |-- det
		 |   |   |   |-- det.txt
		 |   |-- MOT17-01-FRCNN
		 |   |   |-- seqinfo.ini
		 |   |   |-- det
		 |   |   |   |-- det.txt
		 |   |-- MOT17-01-SDP
		 |   |   |-- seqinfo.ini
		 |   |   |-- det
		 |   |   |   |-- det.txt
		 |   |-- MOT17-01-POI
		 |   |   |-- seqinfo.ini
		 |   |   |-- det
		 |   |   |   |-- det.txt
		 |   |-- ...
   ```

### Training and Testing

#### Testing on MOTChallenge dataset using model zoo's models ([BaiduPan](https://pan.baidu.com/s/1QlYAd-wi5IdnJCoBc4tr3g)(extract code: um4c))

```
cd ${GN_ROOT}/GN
python test.py
```

#### Training Appearance Graph Network on MOTChallenge dataset

```
python App2/train.py
```
#### Training Motion Graph Network on MOTChallenge dataset

```
python Motion1/train.py
```
