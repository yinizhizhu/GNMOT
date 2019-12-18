import shutil, os
from global_set import mot_dataset_dir

name = 'motmetrics'
types = ['POI']

seqs = [2, 4, 5, 9, 10, 11, 13]  # the set of sequences
lengths = [600, 1050, 837, 525, 654, 900, 750]  # the length of the sequence

test_seqs = [1, 3, 6, 7, 8, 12, 14]
test_lengths = [450, 1500, 1194, 500, 625, 900, 750]

# copy the results for testing sets
for type in types:
    for i in range(len(seqs)):
        src_dir = 'results/%02d/%d/%s_%s/res.txt' % (test_seqs[i], test_lengths[i], name, type)

        t = type
        if type == 'DPM0' or type == 'POI':
            t = 'DPM'

        des_d = 'mot16/'
        if not os.path.exists(des_d):
            os.mkdir(des_d)
        des_dir = des_d + 'MOT16-%02d.txt' % (test_seqs[i])

        print(src_dir)
        print(des_dir)
        shutil.copyfile(src_dir, des_dir)

# Copy the ground truth of training sets
# types = ['POI']
# for type in types:
#     for i in range(len(seqs)):
#         src_dir = mot_dataset_dir + 'MOT17/train/MOT17-%02d-%s/gt/gt.txt' % (seqs[i], type)
#
#         des_dir = des_d + 'MOT16-%02d.txt' % (seqs[i])
#
#         print(src_dir)
#         print(des_dir)
#         shutil.copyfile(src_dir, des_dir)
