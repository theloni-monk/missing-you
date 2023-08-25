import os
import cv2
import glob
import numpy as np
#list1
#label_list = ['skin', 'neck', 'hat', 'eye_g', 'hair', 'ear_r', 'neck_l', 'cloth', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'nose', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip']
#list2 
label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

folder_base = 'CelebAMask-HQ-mask-anno'
folder_save = 'label_data/label'
img_num = 2000


for k in range(28000, 28000+img_num):
    folder_num = 14
    im_base = np.zeros((512, 512))
    for idx, label in enumerate(label_list):
        filename = os.path.join(folder_base, str(folder_num), str(k) + '_' + label + '.png')
        if (os.path.exists(filename)):
            print(label, idx+1)
            im = cv2.imread(filename)
            im = im[:, :, 0]
            im_base[im != 0] = (idx + 1)
        else:
            print(f"could not find {filename}")

    filename_save = os.path.join(folder_save, str(k) + '.jpg')
    print(filename_save)
    cv2.imwrite(filename_save, im_base)
