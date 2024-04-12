'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Minhyeok Sun, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, hyeok0809@kaist.ac.kr
'''

import os
import sys
auto_label_file_path = os.path.abspath(__file__)
main_dir = os.path.abspath(os.path.join(auto_label_file_path, '..', '..','..'))
sys.path.append(main_dir)
import os.path as osp
os.environ['CUDA_VISIBLE_DEVICES']= '0'

from project.auto_label.pipelines.pipeline_detection_v1_1 import PipelineDetection_v1_1

### To train specific conditional enviornment, *_cond.yml config file are required.
### you can choose the scene environment of (normal, overcast, rain, sleet, fog, lightsnow, heavysnow) in the config file

PATH_CONFIG = './configs/cfg_RTNH_wide_cond.yml' # to test original RTNH or RTNH using auto_labels
# PATH_CONFIG = './configs/cfg_PVRCNNPP_cond.yml' # to test LODN PVRCNNPP
# PATH_CONFIG = './configs/cfg_SECOND_cond.yml' # to test LODN SECOND

# PATH_MODEL = './RTNH_model_log/RTNH.pt' # original RTNH 
PATH_MODEL = './RTNH_model_log/RTNH-PVRCNN.pt' # RTNH trained with PVRCNNPP auto-labels
# PATH_MODEL = './LODN_model_log/LODN_PVRCNNPP.pt' # PVRCNNPP LODN network
# PATH_MODEL = './LODN_model_log/LODN_SECOND.pt' # SECOND LODN network

if __name__ == '__main__':
    pline = PipelineDetection_v1_1(path_cfg=PATH_CONFIG, mode='train')

    memo = 'RTNH' + PATH_MODEL


    print('\033[92m\n', memo, '\n\033[0m')
    pline.load_dict_model(PATH_MODEL)
    pline.validate_kitti_conditional(list_conf_thr=[0.3], is_subset=False, is_print_memory=False)
    print('\033[92m\n', memo, '\n\033[0m')
