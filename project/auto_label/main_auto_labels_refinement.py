'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Minhyeok Sun, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, hyeok0809@kaist.ac.kr
'''

'''
* refine the generated auto-labels using t-1, t+1 label information
* compare the object IoU, filter out the false alarms and supplement the label information of false negatives
'''

import os
import sys
auto_label_file_path = os.path.abspath(__file__)
main_dir = os.path.abspath(os.path.join(auto_label_file_path, '..', '..','..'))
sys.path.append(main_dir)
import os.path as osp
from tqdm import tqdm
from project.auto_label.utils.util_auto_label_evaluation import *
from project.auto_label.utils.util_auto_label import ensure_directory
from project.auto_label.utils.util_auto_label_refinement import *


if __name__=='__main__':

    # Setting ======================================================================================================|
    is_all_seq = True
    get_seq_list = [] # for specific seq


    pvrcnn_label_path = './auto_labels/pvrcnn_inferenced_label_0_3_youradditionalname' # path of target auto-labels
    is_plot_refinement_result = False # if you want to vis refinement result

    is_saving_refinement_label = True # if you want to save refined auto-labels
    revise_pvrcnn_label_path = './auto_labels/pvrcnn_inferenced_label_0_3_refinedfoldername' # path for saving refined auto-labels
    

    lenght_factor = 9 # lenght offset for objects (meters to add)
    width_factor = 1.1 # width offset for objects (scale)
    print(lenght_factor, revise_pvrcnn_label_path)

    if not os.path.exists(revise_pvrcnn_label_path):
    # 폴더가 없으면 생성
        os.makedirs(revise_pvrcnn_label_path)

    # Refinement ======================================================================================================|
    all_seq_list = []
    for i in range(58):
        all_seq_list.append(i+1)

    seq_list = []
    if is_all_seq:
        seq_list = all_seq_list
    else:
        seq_list = get_seq_list

    for s, seq in enumerate(tqdm(seq_list)):
        pvrcnn_label_seq_path = osp.join(pvrcnn_label_path, str(seq))
        pvrcnn_label_list = sorted(os.listdir(pvrcnn_label_seq_path))

        for idx, label_name in enumerate(pvrcnn_label_list):
            filtered_objects = []

            current_label_path = osp.join(pvrcnn_label_path,str(seq),label_name)
            previous_label_path = osp.join(pvrcnn_label_path,str(seq),pvrcnn_label_list[max(0,idx-1)])
            next_label_path = osp.join(pvrcnn_label_path,str(seq),pvrcnn_label_list[min(len(pvrcnn_label_list)-1,idx+1)])


            filtered_objects.extend(remove_false_alarms(current_label_path, previous_label_path, next_label_path, 
                                is_bus_or_truck=False, lenght_factor=lenght_factor, width_factor=width_factor,is_plot_result=is_plot_refinement_result))
            filtered_objects.extend(remove_false_alarms(current_label_path, previous_label_path, next_label_path, is_bus_or_truck=True,is_plot_result=is_plot_refinement_result))

            if is_saving_refinement_label:
                with open(current_label_path, 'r') as file_to_read:
                    for line in file_to_read:
                        meta_data = line.strip()
                        break
                output_file_path = osp.join(revise_pvrcnn_label_path,str(seq),label_name)
                ensure_directory(output_file_path)
                with open(output_file_path, 'w') as file:
                    file.write(meta_data)
                    for obj in filtered_objects:
                        line = f"\n*, P, -1, {obj['confidence']}, {obj['type']}, {obj['x']}, {obj['y']}, {obj['z']}, {obj['theta']}, {obj['length']}, {obj['width']}, {obj['height']}"
                        file.write(line)


