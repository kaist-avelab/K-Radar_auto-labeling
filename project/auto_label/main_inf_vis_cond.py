'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Minhyeok Sun, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, hyeok0809@kaist.ac.kr
'''

'''
* Inference various network (SECOND, PVRCNNPP, RTNH and RTNH trained with auto-labels)
* Visualize the output using open3D, matplotlib
'''

import os
import sys
auto_label_file_path = os.path.abspath(__file__)
main_dir = os.path.abspath(os.path.join(auto_label_file_path, '..', '..','..'))
sys.path.append(main_dir)
import torch
from tqdm import tqdm
from torch.utils.data import Subset
from project.auto_label.pipelines.pipeline_detection_v1_1 import PipelineDetection_v1_1
import numpy as np
from project.auto_label.utils.util_auto_label import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__': 
    SAMPLE_INDICES = []

    # Setting ======================================================================================================|
    ### model setting

    PATH_CONFIG = './configs/cfg_RTNH_wide_cond.yml'    # --- cfg for RTNH or RTNH-*
    # PATH_CONFIG = './configs/cfg_PVRCNNPP_cond.yml'   # --- cfg for LODN-PVRCNN
    # PATH_CONFIG = './configs/cfg_SECOND_cond.yml'     # --- cfg for LODN-SECOND

    # PATH_MODEL = './RTNH_model_log/RTNH.pt'              # --- original RTNH 
    PATH_MODEL = './RTNH_model_log/RTNH-PVRCNN.pt'     # --- RTNH trained with PVRCNNPP auto-labels
    # PATH_MODEL = './LODN_model_log/LODN_PVRCNNPP.pt'   # --- PVRCNNPP LODN network
    # PATH_MODEL = './LODN_model_log/LODN_SECOND.pt'     # --- SECOND LODN network

    CONFIDENCE_THR = 0.3

    is_all_sample = True
    if is_all_sample != True: SAMPLE_INDICES = [] # --- if not is_all_sample, add specific sample indicies in the list
    is_shuffle = True


    ### for visualization setting
    is_3D_vis = True            # --- for visualization using open3D (3D view)
    if is_3D_vis:
        is_vis_rdr_points = True # --- only work with RTNH model
        is_3D_label_vis = True
        is_3D_inference_vis = True

    is_plt_vis = False           # --- for visualization using matplotlib (Bird eye view)
    if is_plt_vis:
        is_plt_label = True     # --- for vis label data with sensor data
        is_plt_inference = True # --- for vis inference output with sensor data 

    is_plt_save = False          # --- for save visulization output
    plt_save_path = './plt_save_folder/your_folder_name'

    idx = 0                     # --- start with the indices that you enter in "SAMPLE_INDICES = []"
    max_iter = 100              # --- 100 inference outputs will be generated / max_iter = 34994 if you want entire dataset to inference 



    ### =================================================================================================================|
    ### =================================================================================================================|
    ### network and dataset setting =====================================================================================|



    pline = PipelineDetection_v1_1(PATH_CONFIG, mode='all') # --- mode : train, test, all
    pline.load_dict_model(PATH_MODEL)
    pline.network.eval()

    dataset_loaded = pline.dataset_test
    if is_all_sample == True:
        for i in (range(len(dataset_loaded))):
            SAMPLE_INDICES.append(i+idx)
    if is_shuffle == True:
        np.random.seed(10)
        np.random.shuffle(SAMPLE_INDICES)
    subset = Subset(dataset_loaded, SAMPLE_INDICES)
    data_loader = torch.utils.data.DataLoader(subset,
            batch_size = 1, shuffle = False,
            collate_fn = dataset_loaded.collate_fn,
            num_workers = 1)
    print("Number of workers in DataLoader:", data_loader.num_workers)

    ### for inference model ============================================================================================|
    for idx, dict_item in enumerate(tqdm(data_loader)):
        if max_iter is not None and idx >= max_iter: exit()

        ### Inference data and find dict_item seq and label ID
        dict_item_seq = dict_item['meta'][0]['seq']
        dict_item_label_id = dict_item['meta'][0]['label_v1_0'].split('/')[-1]

        print('Seq : %s / Label ID : %s / ' %(dict_item_seq,dict_item_label_id) + '\033[0m')
                
        ### Run network inference
        with torch.no_grad(): dict_item = pline.network(dict_item)


        ### NMS post-processing
        pred_dicts = dict_item['pred_dicts'][0]
        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
        pred_dicts = post_processing_nms(pred_dicts)


        ### CONFIDENCE_THR filtering
        try:
            low_score_indices = pred_dicts['pred_scores'] < CONFIDENCE_THR # roi_scores
            # 높은 점수를 받은 요소들만 유지
            filtered_nms_output = pred_dicts['pred_boxes'][~low_score_indices]
            filtered_roi_scores = pred_dicts['pred_scores'][~low_score_indices]
            filtered_cls = pred_dicts['pred_labels'][~low_score_indices]

            # pred_dicts 업데이트
            dict_item['pred_dicts'][0]['pred_boxes'] = filtered_nms_output
            dict_item['pred_dicts'][0]['pred_scores'] = filtered_roi_scores
            dict_item['pred_dicts'][0]['pred_labels'] = filtered_cls

            predicted_objs = dict_item['pred_dicts'][0]['pred_boxes'].detach().cpu().numpy()
            dict_item['pred_dicts'][0]['num_pred_output'] = len(predicted_objs)
            print('\n','\033[92m' + 'Seq : %s / Label ID : %s / Inference success' %(dict_item_seq,dict_item_label_id) + '\033[0m')
            print('predicted output : ',filtered_cls, filtered_roi_scores)


        except RuntimeError as re:
            print(f'\033[91mSeq : {dict_item_seq} / Label ID : {dict_item_label_id} / RuntimeError occurred: {re}\033[0m')
            pred_dicts['pred_boxes'] = None
            pred_dicts['pred_scores'] = None
            pred_dicts['pred_labels'] = None
            pred_dicts['num_pred_output'] = 0

        except IndexError as ie:
            print(f'\033[91mSeq : {dict_item_seq} / Label ID : {dict_item_label_id} / IndexError occurred: {ie}\033[0m')
            pred_dicts['pred_boxes'] = None
            pred_dicts['pred_scores'] = None
            pred_dicts['pred_labels'] = None
            pred_dicts['num_pred_output'] = 0

        


        ### for 3D view visualization ============================================================================================|
        
        if is_3D_vis == True: vis_rdr_data_using_o3d(dict_item, dataset_loaded, predicted_objs, filtered_cls, is_3D_label_vis=is_3D_label_vis, is_3D_inference_vis=is_3D_inference_vis, 
                                                        is_vis_rdr_points=is_vis_rdr_points)

        ### for bird eye view visualization ======================================================================================|
    
        if is_plt_vis == True: vis_lpc_data_using_plt(dict_item, is_plt_vis=is_plt_vis, is_plt_label=is_plt_label, is_plt_inference=is_plt_inference, predicted_objs=predicted_objs, filtered_cls=filtered_cls, 
                                                        is_plt_save=is_plt_save, plt_save_path=plt_save_path)
        elif is_plt_vis == False and is_plt_save == True: vis_lpc_data_using_plt(dict_item, is_plt_vis=is_plt_vis, is_plt_label=True, is_plt_inference=True, predicted_objs=predicted_objs, filtered_cls=filtered_cls, 
                                                                                    is_plt_save=is_plt_save, plt_save_path=plt_save_path)


        torch.cuda.empty_cache() # empty vram 
        print('\n','==================================================================================================','\n')
