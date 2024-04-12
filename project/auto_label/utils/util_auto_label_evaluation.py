"""
# -*- coding: utf-8 -*-
--------------------------------------------------------------------------------
'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Minhyeok Sun, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, hyeok0809@kaist.ac.kr
'''
--------------------------------------------------------------------------------
# description: util fuctions to evaulate auto-label accuracy
"""
import os
import sys
auto_label_file_path = os.path.abspath(__file__)
main_dir = os.path.abspath(os.path.join(auto_label_file_path, '..', '..','..'))
sys.path.append(main_dir)
import numpy as np
from shapely.geometry import Polygon



def parse_label_file(file_path):
    objects = []
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:] # 첫 줄(meta 정보) 제외

    for line in lines:
        parts = line.strip().split(',')
        obj_type = parts[4].strip() if ' P' in parts else parts[2].strip()
        confidence_score = float(parts[3])
        x, y, z, theta, l, w, h = map(float, parts[-7:])
        objects.append({'type': obj_type, 'x': x, 'y': y, 'z':z, 'theta': theta, 'length': l, 'width': w, 'height':h, 'confidence':confidence_score})
    return objects


def parse_inference_file(file_path):# pvrcnn label form : *, P, -1, confidence, Sedan, x, y, z, theta, w, l, h
    objects = []
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]  # 첫 줄(meta 정보) 제외

    for line in lines:
        parts = line.strip().split(',')
        obj_type = parts[4].strip() if ' P' in parts else parts[2].strip()
        x, y, z, theta, l, w, h = map(float, parts[-7:])
        objects.append({'type': obj_type, 'x': x+2.3, 'y': y-0.3, 'theta': theta, 'length': l, 'width': w})
    return objects


def parse_label_v2_1_file(file_path): # v2_1 label form : *, R, 0, Sedan, x, y, z, theta, w, l, h
    objects = []
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]  # 첫 줄(meta 정보) 제외

    for line in lines:
        parts = line.strip().split(',')
        obj_type = parts[3].strip()
        x, y, z, theta, l, w, h = map(float, parts[-7:])
        objects.append({'type': obj_type, 'x': x, 'y': y, 'theta': theta, 'length': l, 'width': w})
    return objects


def rotate_point(x, y, theta):
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    return x_rot, y_rot


def get_box_corners(box):
    x, y, theta, length, width = box['x'], box['y'], box['theta'], box['length'], box['width']
    corners = []
    for dx, dy in [(-length/2, -width/2), (-length/2, width/2), (length/2, width/2), (length/2, -width/2)]:
        rx, ry = rotate_point(dx, dy, theta)
        corners.append((x + rx, y + ry))
    return corners


def calculate_iou(box1, box2):
    

    poly1 = Polygon(get_box_corners(box1))
    poly2 = Polygon(get_box_corners(box2))

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    inter_area = poly1.intersection(poly2).area
    union_area = poly1.area + poly2.area - inter_area
    return inter_area / union_area if union_area else 0.0


def calculate_max_iou_precision(a_objects, b_objects, obj_type): #pvrcnn 라벨 기준 정답 라벨과 비교
    max_iou_list = []
    for b_obj in b_objects:
        if b_obj['type'] == obj_type:
            max_iou = 0
            for a_obj in a_objects:
                if a_obj['type'] == obj_type:
                    iou = calculate_iou(a_obj, b_obj)
                    max_iou = max(max_iou, iou)
            max_iou_list.append(max_iou)
    return max_iou_list


def calculate_max_iou_recall(b_objects, a_objects, obj_type): #정답 라벨 기준 pvrcnn 과 비교
    max_iou_list = []
    for b_obj in b_objects:
        if b_obj['type'] == obj_type:
            max_iou = 0
            for a_obj in a_objects:
                if a_obj['type'] == obj_type:
                    iou = calculate_iou(a_obj, b_obj)
                    max_iou = max(max_iou, iou)
            max_iou_list.append(max_iou)
    return max_iou_list


def calculcalte_precision_and_recall_of_pvrcnn_label(hand_label=None, pvrcnn_label=None, thr=0.01):
# 파일 경로
    a_label_path = hand_label
    b_label_path = pvrcnn_label

    # 객체 데이터 파싱
    a_objects = parse_label_file(a_label_path)
    b_objects = parse_inference_file(b_label_path)

    # Sedan과 Bus or Truck 객체에 대한 최대 IOU 계산
    max_iou_sedan_precision = calculate_max_iou_precision(a_objects, b_objects, 'Sedan')
    max_iou_bus_truck_precision = calculate_max_iou_precision(a_objects, b_objects, 'Bus or Truck')
    max_iou_sedan_recall = calculate_max_iou_recall(a_objects, b_objects, 'Sedan')
    max_iou_bus_truck_recall = calculate_max_iou_recall(a_objects, b_objects, 'Bus or Truck')


    output = [max_iou_sedan_precision,max_iou_sedan_recall,max_iou_bus_truck_precision,max_iou_bus_truck_recall]
    return output


def calculate_max_iou_precision(a_objects, b_objects, obj_type): #pvrcnn 라벨 기준 정답 라벨과 비교
    max_iou_list = []
    for b_obj in b_objects:
        if b_obj['type'] == obj_type:
            max_iou = 0
            for a_obj in a_objects:
                if a_obj['type'] == obj_type:
                    iou = calculate_iou(a_obj, b_obj)
                    max_iou = max(max_iou, iou)
            max_iou_list.append(max_iou)
    return max_iou_list


def calculate_max_iou_recall(b_objects, a_objects, obj_type): #정답 라벨 기준 pvrcnn 과 비교
    max_iou_list = []
    for b_obj in b_objects:
        if b_obj['type'] == obj_type:
            max_iou = 0
            for a_obj in a_objects:
                if a_obj['type'] == obj_type:
                    iou = calculate_iou(a_obj, b_obj)
                    max_iou = max(max_iou, iou)
            max_iou_list.append(max_iou)
    return max_iou_list


def calculcalte_precision_and_recall_of_pvrcnn_label(hand_label=None, pvrcnn_label=None, thr=0.01):
# 파일 경로
    a_label_path = hand_label
    b_label_path = pvrcnn_label

    # 객체 데이터 파싱
    a_objects = parse_label_file(a_label_path)
    b_objects = parse_inference_file(b_label_path)

    # Sedan과 Bus or Truck 객체에 대한 최대 IOU 계산
    max_iou_sedan_precision = calculate_max_iou_precision(a_objects, b_objects, 'Sedan')
    max_iou_bus_truck_precision = calculate_max_iou_precision(a_objects, b_objects, 'Bus or Truck')
    max_iou_sedan_recall = calculate_max_iou_recall(a_objects, b_objects, 'Sedan')
    max_iou_bus_truck_recall = calculate_max_iou_recall(a_objects, b_objects, 'Bus or Truck')


    output = [max_iou_sedan_precision,max_iou_sedan_recall,max_iou_bus_truck_precision,max_iou_bus_truck_recall]
    return output


