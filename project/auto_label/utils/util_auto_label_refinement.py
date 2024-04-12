"""
# -*- coding: utf-8 -*-
--------------------------------------------------------------------------------
'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Minhyeok Sun, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, hyeok0809@kaist.ac.kr
'''
--------------------------------------------------------------------------------
# description: util fuctions for auto-label refinement process
"""
import os
import sys
auto_label_file_path = os.path.abspath(__file__)
main_dir = os.path.abspath(os.path.join(auto_label_file_path, '..', '..','..'))
sys.path.append(main_dir)
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from project.auto_label.utils.util_auto_label_evaluation import *

def adjust_box_dimensions(box, conf_score, lenght_factor=6, width_factor=1.1):
    box['width'] *= width_factor
    box['length'] += lenght_factor
    return box


def get_adjusted_boxes(objects, is_bus_or_truck=False, lenght_factor=6, width_factor=1.1):
    adjusted_boxes = []
    for obj in objects:
        if is_bus_or_truck and obj['type'] != 'Bus or Truck':
            continue
        elif not is_bus_or_truck and obj['type'] == 'Bus or Truck':
            continue
        
        if is_bus_or_truck: 
            adjusted_boxes.append(obj)
        else:
            adjusted_box = adjust_box_dimensions(obj, obj.get('conf_score', 1.0),lenght_factor, width_factor)
            adjusted_boxes.append(adjusted_box)

    return adjusted_boxes


def plot_2d_boxes_inv(ax, objects, color):
    for obj in objects:
        x, y, theta, length, width = obj['x'], obj['y'], obj['theta'], obj['length'], obj['width']

        rect = Rectangle((x - length/2, y - width/2), length, width, linewidth=1, edgecolor=color, facecolor='none')
        t = transforms.Affine2D().rotate_around(x, y, theta) + ax.transData
        rect.set_transform(t)

        ax.add_patch(rect)


def plot_2d_boxes(ax, objects, color, size=1, alpha=1):
    for obj in objects:
        x, y, theta, length, width = obj['y'], obj['x'], obj['theta'], obj['width'], obj['length']
        rect = Rectangle((x - length/2, y - width/2), length, width, linewidth=size, edgecolor=color, facecolor='none', alpha=alpha)
        t = transforms.Affine2D().rotate_around(x, y, theta) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)


def remove_duplicate_objects(objects, iou_threshold=0.5):
    filtered_objects = []
    for i, obj1 in enumerate(objects):
        duplicate = False
        for j, obj2 in enumerate(objects):
            if i != j:
                iou = calculate_iou(obj1, obj2)
                if iou >= iou_threshold:
                    if obj1['confidence'] < obj2['confidence']:
                        duplicate = True
                        break
        if not duplicate:
            filtered_objects.append(obj1)

    return filtered_objects


def remove_false_alarms(current_label_path, previous_label_path, next_label_path, is_bus_or_truck=False, thr=0.1,lenght_factor=6,width_factor=1.1,is_plot_result=False):
    current_objects = parse_label_file(current_label_path)
    if is_bus_or_truck == False: current_objects = [obj for obj in current_objects if obj['type'] == 'Sedan']
    else:current_objects = [obj for obj in current_objects if obj['type'] == 'Bus or Truck']

    previous_objects = parse_label_file(previous_label_path)
    if is_bus_or_truck == False: previous_objects = [obj for obj in previous_objects if obj['type'] == 'Sedan']
    else:previous_objects = [obj for obj in previous_objects if obj['type'] == 'Bus or Truck']

    next_objects = parse_label_file(next_label_path)
    if is_bus_or_truck == False: next_objects = [obj for obj in next_objects if obj['type'] == 'Sedan']
    else:next_objects = [obj for obj in next_objects if obj['type'] == 'Bus or Truck']

    if is_plot_result:
        fig, ax = plt.subplots(figsize=(5, 10))
        plot_2d_boxes(ax, current_objects, 'green', size=2)   # 현재 객체들을 파란색으로 그림
        plot_2d_boxes(ax, previous_objects, 'red', size=1, alpha=0.5) # 이전 객체들을 녹색으로 그림
        plot_2d_boxes(ax, next_objects, 'blue',size=1, alpha=0.5)       # 다음 객체들을 빨간색으로 그림


    # 객체들의 차원을 조정합니다.
    current_adjusted = get_adjusted_boxes(current_objects, is_bus_or_truck, lenght_factor=lenght_factor,width_factor=width_factor)
    previous_adjusted = get_adjusted_boxes(previous_objects, is_bus_or_truck)
    next_adjusted = get_adjusted_boxes(next_objects, is_bus_or_truck)

    # IOU 계산과 False Alarm 제거
    check_list = []
    for current_obj in current_adjusted:
        max_iou = 0
        for other_obj in previous_adjusted + next_adjusted:
            iou = calculate_iou(current_obj, other_obj)
            max_iou = max(max_iou, iou)
        check_list.append(max_iou)

    # 0.1보다 작은 IOU 값을 가진 객체 제거
    filtered_objects = []
    for obj, iou in zip(current_objects, check_list):
        if obj['confidence']>0.5:
            filtered_objects.append(obj)
        else:
            if iou>= thr:
                filtered_objects.append(obj)

    missing_objects = add_missing_objects_with_iou_threshold(previous_adjusted, current_adjusted, next_adjusted)
    filtered_objects.extend(missing_objects)
    filtered_objects = remove_duplicate_objects(filtered_objects)


    if not is_bus_or_truck:
        for obj in filtered_objects:
            obj['length'] = obj['length'] -lenght_factor
            obj['width'] = obj['width']/width_factor
    
    if is_plot_result:
        plot_2d_boxes(ax, filtered_objects, 'yellow',size=3) 
        plt.ylim([0, 72])
        plt.xlim([-16, 16])
        def on_key(event):
            if event.key == ' ':
                plt.close()
        plt.gcf().canvas.mpl_connect('key_press_event', on_key)
        plt.show()

    return filtered_objects


def add_missing_objects_with_iou_threshold(prev_objects, current_objects, next_objects, threshold=0.01):
    missing_objects = []

    for next_obj in next_objects:
        if not any(calculate_iou(next_obj, cur_obj) >= threshold for cur_obj in current_objects):
            similar_prev_obj = next((prev_obj for prev_obj in prev_objects if prev_obj['type'] == next_obj['type']
                                     and calculate_iou(prev_obj, next_obj) >= threshold), None)
            
            if similar_prev_obj:
                avg_obj = {
                    'type': next_obj['type'],
                    'x': (similar_prev_obj['x'] + next_obj['x']) / 2,
                    'y': (similar_prev_obj['y'] + next_obj['y']) / 2,
                    'z': (similar_prev_obj['z'] + next_obj['z']) / 2,
                    'theta': (similar_prev_obj['theta'] + next_obj['theta']) / 2,
                    'length': (similar_prev_obj['length'] + next_obj['length']) / 2,
                    'width': (similar_prev_obj['width'] + next_obj['width']) / 2,
                    'height': (similar_prev_obj['height'] + next_obj['height']) / 2,
                    'confidence':(similar_prev_obj['confidence'] + next_obj['confidence']) / 2,
                }
                missing_objects.append(avg_obj)

    return missing_objects


