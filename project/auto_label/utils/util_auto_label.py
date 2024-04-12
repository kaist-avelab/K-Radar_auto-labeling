"""
# -*- coding: utf-8 -*-
--------------------------------------------------------------------------------
'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Minhyeok Sun, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, hyeok0809@kaist.ac.kr
'''
--------------------------------------------------------------------------------
# description: util functions to visualize the inference output and generate auto-labels
"""
import os
import sys
auto_label_file_path = os.path.abspath(__file__)
main_dir = os.path.abspath(os.path.join(auto_label_file_path, '..', '..','..'))
sys.path.append(main_dir)
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import open3d as o3d
import torch
from torchvision.ops import nms
import glob

def ensure_directory(file_path):
    ### ensure the directory exists
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)




def check_point(output, main_path_to_save = None):
    generate_label_name = 'pvrcnn_inferenced_label'
    seq_to_save = (output['meta'][0]['seq'])
    id_to_save = output['meta'][0]['label_v1_0'].split('/')[-1]
    save_label_path = osp.join(main_path_to_save,generate_label_name,seq_to_save,id_to_save)
    if os.path.exists(save_label_path):
        return False
    else: return True




def create_cylinder_mesh(radius, p0, p1, color=[1, 0, 0]):
            # Create a cylinder
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(np.array(p1)-np.array(p0)))
            cylinder.paint_uniform_color(color)

            # Rotate and translate the cylinder to align with the points
            frame = np.array(p1) - np.array(p0)
            frame /= np.linalg.norm(frame)
            R = o3d.geometry.get_rotation_matrix_from_xyz((np.arccos(frame[2]), np.arctan2(-frame[0], frame[1]), 0))
            cylinder.rotate(R, center=[0, 0, 0])
            cylinder.translate((np.array(p0) + np.array(p1)) / 2)
            return cylinder




def draw_3d_box_in_cylinder(vis, center, theta, l, w, h, color=[1, 0, 0], radius=0.1, in_cylinder=True):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,              0,             1]])

    # 8 corners of a 3D box
    corners = np.array([[l/2, w/2, h/2], [l/2, w/2, -h/2], [l/2, -w/2, h/2], [l/2, -w/2, -h/2],
                        [-l/2, w/2, h/2], [-l/2, w/2, -h/2], [-l/2, -w/2, h/2], [-l/2, -w/2, -h/2]])

    # Rotate and translate corners
    corners_rotated = np.dot(corners, R.T) + center

    # Create lines between the corners to form the box edges
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
            [0, 4], [1, 5], [2, 6], [3, 7]]

    # Create a line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_rotated)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])

    # Add the box to the visualizer
    if in_cylinder:
        for line in lines:
            cylinder = create_cylinder_mesh(radius, corners_rotated[line[0]], corners_rotated[line[1]], color)
            vis.add_geometry(cylinder)
    else:
        vis.add_geometry(line_set)




def get_ldr64_from_path(path_ldr64):
    with open(path_ldr64, 'r') as f:
        lines = [line.rstrip('\n') for line in f][13:]
        pc_lidar = [point.split() for point in lines]
        f.close()
    pc_lidar = np.array(pc_lidar, dtype = float).reshape(-1, 9)


    pc_lidar = pc_lidar[np.where(
        (pc_lidar[:, 0] > 0.01) | (pc_lidar[:, 0] < -0.01) |
        (pc_lidar[:, 1] > 0.01) | (pc_lidar[:, 1] < -0.01))]


    n_pts, _ = pc_lidar.shape
    calib_vals = np.array([-2.54,0.3,0.7]).reshape(-1,3).repeat(n_pts, axis=0)
    pc_lidar[:,:3] = pc_lidar[:,:3] + calib_vals

    return pc_lidar




def post_processing_nms(pred_dicts): 
    # for nms
    try:
        x = pred_dicts['pred_boxes'][:, 0]  # 첫 번째 열
        y = pred_dicts['pred_boxes'][:, 1]  # 두 번째 열
        l = pred_dicts['pred_boxes'][:, 3]  # 네 번째 열
        w = pred_dicts['pred_boxes'][:, 4]  # 다섯 번째 열

        # x1, y1, x2, y2 계산
        x1 = x - l / 2
        y1 = y - w / 2
        x2 = x + l / 2
        y2 = y + w / 2

        # 새로운 2D 박스 텐서 생성
        boxes_2d = torch.stack((x1, y1, x2, y2), dim=1).squeeze(0) # [N, 4] 크기로 조정

        # roi_scores 텐서 조정
        scores = pred_dicts['pred_scores'].squeeze(0)  # [N] 크기로 조정

        # NMS 적용
        nms_indices = nms(boxes_2d, scores, iou_threshold=0.01)

        # NMS를 통과한 박스와 점수
        nms_boxes = pred_dicts['pred_boxes'][nms_indices]
        nms_label = pred_dicts['pred_labels'][nms_indices]
        nms_scores = pred_dicts['pred_scores'][nms_indices]
        pred_dicts['pred_boxes']=nms_boxes
        pred_dicts['pred_labels']=nms_label
        pred_dicts['pred_scores']=nms_scores

        
    except RuntimeError as e:
        if pred_dicts['pred_boxes'].numel() == 0:
            print(f'\033[91mRuntimeError occurred: {e}\033[0m')
            print('error msg from pv_pp ln.129 : no predicted object existed')
            pred_dicts['pred_boxes']=None
            pred_dicts['pred_labels']=None
            pred_dicts['pred_scores']=None

    return(pred_dicts)




def get_color(confidence_score, color_1='red', color_2='green', confidnce_thr = 0.5, thr_gap=0.2): 
    ### for color scale using in vis_lpc_data_using_plt 
    low_thr = max(0,confidnce_thr)
    high_thr = min(confidnce_thr+thr_gap,1.0)
    if confidence_score < low_thr: return color_1
    elif confidence_score > high_thr: return color_2
    else:
        cmap = LinearSegmentedColormap.from_list('custom_cmap', [color_1, color_2])
        return cmap((confidence_score - low_thr) / (high_thr - low_thr))




def vis_rdr_data_using_o3d(dict_item, dataset_loaded, predicted_objs, filtered_cls, is_3D_label_vis=False, is_3D_inference_vis=False, is_vis_rdr_points=False):    
    vis = o3d.visualization.Visualizer()
    window_width = 700 
    window_height = 900 
    vis.create_window(width=window_width, height=window_height)
    label = dict_item['label'][0]

    if is_vis_rdr_points:
        rdr_points = dict_item['rdr_sparse']
        rdr_x_points = rdr_points[:, 0]
        rdr_y_points = rdr_points[:, 1]
        rdr_z_points = rdr_points[:, 2]
        rdr_intensity = rdr_points[:, 3]
        rdr_xyz_points = np.hstack((rdr_x_points.reshape(-1, 1), rdr_y_points.reshape(-1, 1), rdr_z_points.reshape(-1, 1)))     
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(rdr_xyz_points)
        intensities = rdr_intensity
        max_intensity = 0.2
        min_intensity = 0
        cmap = plt.get_cmap('viridis_r')
        colors = [cmap((i - min_intensity) / (max_intensity - min_intensity))[:3] for i in intensities]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(pcd)


        pc_lidar = dataset_loaded.get_ldr64_from_path(dict_item['meta'][0]['path']['ldr64'])
        pcd_l = o3d.geometry.PointCloud()
        pcd_l.points = o3d.utility.Vector3dVector(pc_lidar[:,:3])
        intensities = pc_lidar[:, 3]
        light_gray_color = [0.75, 0.75, 0.75]  # 연한 회색 색상
        colors_l = np.tile(light_gray_color, (len(pcd_l.points), 1))
        pcd_l.colors = o3d.utility.Vector3dVector(colors_l)
        vis.add_geometry(pcd_l)
        render_size = 0.2


    else:
        pc_lidar = dataset_loaded.get_ldr64_from_path(dict_item['meta'][0]['path']['ldr64'])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_lidar[:,:3])
        intensities = pc_lidar[:, 3]
    
        max_intensity = 20
        min_intensity = 0

        cmap = plt.get_cmap('viridis_r')
        colors = [cmap((i - min_intensity) / (max_intensity - min_intensity))[:3] for i in intensities]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(pcd)
        render_size = 0.3
    
    
    render_option = vis.get_render_option()
    render_option.point_size *= render_size

    if is_3D_label_vis == True:
        for obj in label:
            cls_name, _, (x, y, z, th, l, w, h), trk_id = obj
            if cls_name == 'Sedan': rgb = [0,0,1]
            elif cls_name == 'Bus or Truck': rgb = [0.4, 0, 1]
            else: [0.3,0.3,0.3]
            draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=rgb, radius=0.18)

    if is_3D_inference_vis == True:
        for i, obj in enumerate(predicted_objs):
            cls_name = filtered_cls[i]
            if cls_name.item() == 1: rgb = [1,0,0]
            elif cls_name.item() == 2: rgb = [1,0.7,0]
            else: rgb = [0,0,0]
            x, y, z, l, w, h, th = obj
            draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h,  color=rgb,radius=0.18, in_cylinder=True)

    
    view_control = vis.get_view_control()
    view_settings = {
        "lookat": [ 20.801952763510624, 6.9942054386173291, 4.2808899887458649 ],
        "up": [ 0.7175963976970039, 0.022781242878260506, 0.69608650682598738 ],
        "front": [ -0.69305284665717293, -0.075371796850189715, 0.71693573211269301 ],
        "zoom": 0.059999999999999998,
        "field_of_view": 60.0
    }


    view_control.set_lookat(np.array(view_settings["lookat"]))
    view_control.set_up(np.array(view_settings["up"]))
    view_control.set_front(np.array(view_settings["front"]))
    view_control.set_zoom(view_settings["zoom"])


    vis.run()
    vis.destroy_window()
    del render_option
    del vis




def vis_lpc_data_using_plt(dict_item, is_plt_vis=False, is_plt_label=False, is_plt_inference=False, predicted_objs=None, filtered_cls=None, is_plt_save=False, plt_save_path=None):
    ### plt PVRCNN inference output with label
    dict_item_seq = dict_item['meta'][0]['seq']
    dict_item_label_id = dict_item['meta'][0]['label_v1_0'].split('/')[-1]
    ldr_points = None

    if 'ldr64' in dict_item:
        points = dict_item['ldr64']
    elif 'rdr_sparse' in dict_item:
        print('here')
        rdr_points = dict_item['rdr_sparse']
        points = get_ldr64_from_path(dict_item['meta'][0]['path']['ldr64'])
    else:
        raise Exception('no lidar and radar points are input')

    x_points = points[:, 0]
    y_points = points[:, 1]
    intensity = points[:, 3]

    # BEV plot
    plt.figure(figsize=(5, 10))
    if 'ldr64' in dict_item: 
        scatter = plt.scatter(y_points, x_points, s=0.5, c=intensity, cmap='viridis_r', vmin=0, vmax=20, alpha=0.8)
    elif 'rdr_sparse' in dict_item: 
        rdr_x_points = rdr_points[:, 0]
        rdr_y_points = rdr_points[:, 1]
        rdr_intensity = rdr_points[:, 3]+0.1
        
        scatter_rdr = plt.scatter(rdr_y_points, rdr_x_points, s=2, c=rdr_intensity, cmap='viridis_r', vmin=0.1, vmax=0.3, alpha=0.7)
        scatter = plt.scatter(y_points, x_points, s=0.1, c='black', cmap='viridis_r', vmin=0, vmax=20, alpha=0.05)
    
    if is_plt_label == True:
        for o_idx, object in enumerate(dict_item['label'][0]):
            x, y, z, th, l, w, h = object[2]
            corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            rotated_corners = np.dot(corners, R.T) + np.array([x, y])
            if object[0] == 'Sedan': label_color = 'blue'
            elif object[0] == 'Bus or Truck': label_color = 'purple'
            else: label_color = 'black'
            x_coords = rotated_corners[:,1]
            y_coords = rotated_corners[:,0]

            for i in range(4):
                j = (i + 1) % 4
                plt.plot([rotated_corners[i][1], rotated_corners[j][1]], 
                        [rotated_corners[i][0], rotated_corners[j][0]], 
                        color=label_color, linewidth=2, alpha=1.0)

    if is_plt_inference==True:
        if predicted_objs is not None:
            for idx, obj in enumerate(predicted_objs):
                x, y, z, l, w, h, th = obj
                corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
                R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                rotated_corners = np.dot(corners, R.T) + np.array([x, y])
                if filtered_cls[idx] == 1: label_color = 'red'
                elif filtered_cls[idx] == 2: label_color = 'orange'
                for i in range(4):
                    j = (i + 1) % 4
                    plt.plot([rotated_corners[i][1], rotated_corners[j][1]], 
                            [rotated_corners[i][0], rotated_corners[j][0]], 
                            color=label_color,linewidth=2)
        else:
            raise ValueError("no inference objects are input")
        
    plt.title(str('label : ' + dict_item_seq + ' / ' + dict_item_label_id))
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.xlim([0, 72])
    plt.ylim([-16, 16])
    plt.axis([-17,17,-1,73])
    plt.gca().invert_xaxis()

    if 'ldr64' in dict_item: plt.colorbar(scatter, label='LDR Intensity')
    else: plt.colorbar(scatter_rdr, label='RDR Power')
    
    def on_key(event):
        if event.key == ' ':
            plt.close()

    plt.gcf().canvas.mpl_connect('key_press_event', on_key)
    
    if is_plt_save:
        if plt_save_path == None: raise ValueError("no save path for plt is given")
        else:
            plt_save_path = osp.join(plt_save_path,(dict_item_seq+'_'+dict_item_label_id.split('.txt')[0]+'.png'))
            ensure_directory(plt_save_path)
            plt.savefig(plt_save_path,format='png',dpi=300)
            print(plt_save_path)
    if is_plt_vis: plt.show()





def generate_label_with_LODN_inference(output, main_path_to_save = None, generate_label_name='pvrcnn_inferenced_label'): 
    ### generate label using LODN inference output (label version = v1_1)

    # find path to save the inference output
    if main_path_to_save == None:  raise ValueError("no save path for inf_label is given")
    seq_to_save = (output['meta'][0]['seq'])
    id_to_save = output['meta'][0]['label_v1_0'].split('/')[-1]
    save_label_path = osp.join(main_path_to_save,generate_label_name,seq_to_save,id_to_save) 

    if os.path.exists(save_label_path):
        return
    else:
        # make first meta line using existing revised_label
        revised_label_path = (output['meta'][0]['label_v2_1'])
        data_to_save = []
        with open(revised_label_path, 'r') as file_to_read:
            for line in file_to_read:
                data_to_save.append(line.strip())
                break

        # make lable information with pvrcnn inference output
        try:
            if output['pred_dicts'][0]['num_pred_output'] > 0:
                predicted_item = output['pred_dicts'][0]['pred_boxes']
                for idx, item in enumerate(predicted_item):
                    predict_obj = output['pred_dicts'][0]['pred_labels'][idx].item()
                    score = "{:.2f}".format(output['pred_dicts'][0]['pred_scores'][idx].item())
                    if predict_obj == 1: label = 'Sedan'
                    elif predict_obj == 2: label = 'Bus or Truck'
                    else: continue

                    x,y,z,l,w,h,theta = item
                    x,y,z,l,w,h,theta = x.item(),y.item(),z.item(),l.item(),w.item(),h.item(),theta.item()
                    w,h,l = w/2, h/2, l/2
                    theta = theta % 3.14
                    if theta > 1.57: theta = theta - 3.14
                    else: theta = theta
                    data_line = '*, P, -1, %s, %s, %f, %f, %f, %f, %f, %f, %f' % (score, label, x, y, z, theta, l, w, h)
                    data_to_save.append(data_line)
            else: pass


        except KeyError:
            print('error when generating pvrcnn label')

        print('Inf    label save path : ',save_label_path)
        ensure_directory(save_label_path)
        with open(save_label_path, 'w') as file_to_write:
            for line in data_to_save:
                file_to_write.write(line + '\n')




def generate_label_of_no_object(dict_item, main_path_to_save, generate_label_name): 
    ### to generate empty inference label 
    list_seq = []
    for i in range(58):
        list_seq.append(i+1)
     
    for seq in list_seq:
        #try:
        revise_label_seq_path = osp.join('/',*dict_item['meta'][0]['label_v1_0'].split('/')[0:7], str(seq), 'info_label')
        inference_label_seq_path = osp.join(main_path_to_save,generate_label_name, str(seq))

        revise_files = set([os.path.basename(f) for f in glob.glob(os.path.join(revise_label_seq_path, "*.txt"))])
        inference_files = set([os.path.basename(f) for f in glob.glob(os.path.join(inference_label_seq_path, "*.txt"))])

        missing_files = revise_files.difference(inference_files)
        missing_files = sorted(missing_files)

        for file in missing_files:
            data_to_save = []
            with open(osp.join(revise_label_seq_path,file), 'r') as file_to_read:
                for line in file_to_read:
                    data_to_save.append(line.strip())
                    break

            with open(osp.join(inference_label_seq_path,file), 'w') as file_to_write:
                for line in data_to_save:
                    file_to_write.write(line + '\n')
            print('missing label generated : ', osp.join(inference_label_seq_path,file))
        #except: print('seq is not exist : ', seq)
    print('end of saving process')



