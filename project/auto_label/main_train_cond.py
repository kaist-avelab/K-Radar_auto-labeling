'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
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

PATH_CONFIG = './configs/cfg_RTNH_wide_cond.yml' # to train original RTNH or RTNH using auto_labels (label version : v_P)
# PATH_CONFIG = './configs/cfg_PVRCNNPP_cond.yml' # to train LODN PVRCNNPP
# PATH_CONFIG = './configs/cfg_SECOND_cond.yml' # to train LODN SECOND

if __name__ == '__main__':


    memo = '' + PATH_CONFIG
    pline = PipelineDetection_v1_1(path_cfg=PATH_CONFIG, mode='train')
    ### Save this file for checking ###
    import shutil
    shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))

    ### Save this file for checking ###
    print('\033[92m\n', memo, '\n\033[0m')
    pline.train_network()
    
    ### conditional evaluation for last epoch
    print('\033[92m\n', memo, '\n\033[0m')
    pline.validate_kitti_conditional(list_conf_thr=[0.3], is_subset=False, is_print_memory=False)


    ### memo
    print('\033[92m\n', memo, '\n\033[0m')
