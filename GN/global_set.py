#mot_dataset_dir = '/media/codinglee/DATA/Ubuntu16.04/Desktop/MOT/'
mot_dataset_dir = '../MOT/'

model_dir = 'model/'

u_initial = 1  # 1 - random, 0 - 0

edge_initial = 0  # 1 - random, 0 - IoU

criterion_s = 0  # 1 - MSELoss, 0 - CrossEntropyLoss

test_gt_det = 0  # 1 - detections of gt, 0 - detections of det

u_update = 1  # 1 - update when testing, 0 - without updating
if u_update:
    u_dir = '_uupdate'
else:
    u_dir = ''

app_fine_tune = 0  # 1 - fine-tunedthe appearance model, 0 - pre-trained appearance model
if app_fine_tune:
    fine_tune_dir = '../MOT/Fine-tune_GPU_5_3_60_aug/appearance_19.pth'
    app_dir = 'Finetuned'
else:
    fine_tune_dir = ''
    app_dir = 'Pretrained'

decay = 1.3
decay_dir = '_decay'

f_gap = 5
if f_gap:
    recover_dir = '_Recover'
else:
    recover_dir = '_NoRecover'

tau_threshold = 1.0  # The threshold of matching cost
tau_dis = 2.0  # The times of the current bbx's scale
gap = 25  # max frame number for side connection

show_recovering = 0  # 1 - 11, 0 - 10

overlap = 0.85  # the IoU
