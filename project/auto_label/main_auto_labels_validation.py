'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Minhyeok Sun, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, hyeok0809@kaist.ac.kr
'''

'''
* Calculate TP, FP, FN, Precision, Recall, and F1 scores of generated auto-labels
'''

import os
import sys
auto_label_file_path = os.path.abspath(__file__)
main_dir = os.path.abspath(os.path.join(auto_label_file_path, '..', '..','..'))
sys.path.append(main_dir)
import os.path as osp
from project.auto_label.utils.util_auto_label_evaluation import *


if __name__=='__main__':
    is_all_seq = True
    all_seq_list = []
    for i in range(58):
        all_seq_list.append(i+1)
    normal = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    overcast = [22]
    rain = [21,23,24,25,26,32,33,34]
    sleet = [27,28,29,30,31,35,36,37,50,51,52,53]
    fog = [38,39,40,41,44,45]
    lightsnow = [42,43,48,49]
    heavysnow = [46,47,54,55,56,57,58]

    all = [all_seq_list,normal,overcast,fog, rain,sleet,lightsnow,heavysnow]

    get_seq_list = all
    hand_label_path = '../../tools/revise_label/kradar_revised_label_v2_0/KRadar_refined_label_by_UWIPL'
    pvrcnn_label_path = './auto_labels/pvrcnn_inferenced_label_0_3_refinedfoldername'

    memo = 'auto_label path : ' + pvrcnn_label_path
    print('\033[92m\n', memo, '\n\033[0m')

    seq_list = []

    if is_all_seq:
        seq_list = all

    else:
        seq_list = get_seq_list

    for l, list in enumerate(get_seq_list):
        seq_list = list
        sedan_output_precision = []
        sedan_output_recall = []
        bus_output_precision = []
        bus_output_recall = []

        for s, seq in enumerate(seq_list):
            if s == 0:
                if len(seq_list) > 50: condition = 'all'
                else:
                    if seq in normal: condition = 'normal'
                    elif seq in overcast: condition = 'overcast'
                    elif seq in rain: condition = 'rain'
                    elif seq in sleet: condition = 'sleet'
                    elif seq in fog: condition = 'fog'
                    elif seq in lightsnow: condition = 'lightsnow'
                    elif seq in heavysnow: condition = 'heavysnow'
                    else: condition = 'None'

            hand_label_seq_path = osp.join(hand_label_path, str(seq))
            pvrcnn_label_seq_path = osp.join(pvrcnn_label_path, str(seq))

            hand_label_list = sorted(os.listdir(hand_label_seq_path))
            pvrcnn_label_list = sorted(os.listdir(pvrcnn_label_seq_path))

            for i in (range(len(pvrcnn_label_list))):
                hand_label_file_path = osp.join(hand_label_seq_path,hand_label_list[i])
                pvrcnn_label_file_path = osp.join(pvrcnn_label_seq_path,pvrcnn_label_list[i])

                output = calculcalte_precision_and_recall_of_pvrcnn_label(hand_label_file_path,pvrcnn_label_file_path)
                sedan_output_precision.extend(output[0])
                sedan_output_recall.extend(output[1])
                bus_output_precision.extend(output[2])
                bus_output_recall.extend(output[3])


        sedan_fp = len([x for x in sedan_output_precision if x < 0.1])
        sedan_tp = len(sedan_output_precision) - sedan_fp 
        sedan_fn = len([x for x in sedan_output_recall if x < 0.1]) 
        sedan_tp2 = len(sedan_output_recall) - sedan_fn

        sedan_precision = sedan_tp2 / (sedan_tp+sedan_fp+0.0000001)
        sedan_recall = sedan_tp2 / (sedan_tp2 + sedan_fn+0.0000001)
        sedan_f1_score = 2/((1/(sedan_precision+0.0000001)) + 1/((sedan_recall + 0.000001))+0.0000001)
        sedan_accuracy = sedan_tp2 / (sedan_tp2 + sedan_fp + sedan_fn +0.0000001)

        bus_fp = len([x for x in bus_output_precision if x < 0.1]) 
        bus_tp = len(bus_output_precision) - bus_fp 
        bus_fn = len([x for x in bus_output_recall if x < 0.1])
        bus_tp2 = len(bus_output_recall) - bus_fn 

        bus_precision = bus_tp2 / (bus_tp+bus_fp+0.0000001)
        bus_recall = bus_tp2 / (bus_tp2 + bus_fn+0.0000001)
        bus_f1_score = 2/(1/(bus_precision+0.0000001) + 1/(bus_recall + 0.000001) + 0.000001)
        bus_accuracy = bus_tp2 / (bus_tp2 + bus_fp + bus_fn +0.0000001)

        print('\n================================\n')
        print(condition)
        print('\nSEDAN')
        print('FP, FN, TP, total : ', sedan_fp, sedan_fn, sedan_tp2, sedan_fn + sedan_tp2)
        print('Pre, Rec, F1, Acc : {:.3f} {:.3f} {:.3f} {:.3f}'.format(sedan_precision, sedan_recall, sedan_f1_score, sedan_accuracy))
        print('\nBUS or TRUCK')
        print('FP, FN, TP, total : ', bus_fp, bus_fn, bus_tp2, bus_fn + bus_tp2)
        print('Pre, Rec, F1, Acc : {:.3f} {:.3f} {:.3f} {:.3f}'.format(bus_precision, bus_recall, bus_f1_score, bus_accuracy))


    print('\033[92m\n', memo, '\n\033[0m')