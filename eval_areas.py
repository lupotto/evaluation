import sys
import os
import glob
import cv2
import numpy as np
import math
import pickle
import argparse
import logging
from tools.utils import draw_plot_func,  compute_ap, bbox_iou, range_areas


root_path = '/home/alupotto/data/autel'


def mainParser():
    parser = argparse.ArgumentParser(description='Autel Dataset')
    parser.add_argument('--ground_truth', type=str,  help='path ground_truth labels')
    parser.add_argument('--predictions' , type=str,  help='path prediction files')
    parser.add_argument('--images'      , type=str,  help='path file image')

    return parser




#TODO: change paths to arguments (mainParser)
#TODO: check if labels and images are the same id (preproces)
def evaluation_areas():

    ROOT_PATH = '/home/alupotto/data/autel'

    #read images path
    with open(os.path.join(ROOT_PATH,'autel.txt'),'r') as f_imgs:
         images_list= f_imgs.readlines()
    images_list.sort()

    #read labels path
    gt_path = os.path.join(ROOT_PATH, 'new_labels')
    gt_list = glob.glob("{}/*.txt".format(gt_path))
    gt_list.sort()

    #dict areas
    areas_keys = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
    dict_areas = dict.fromkeys(areas_keys, 0)

    #file image id != label id
    f_not_match = open('output/gt_img_notmatch.txt', 'w')
    for idx, gt_file in enumerate(gt_list):

        #check if image_id == label_id
        if gt_file.strip('.txt').split('/')[-1] != \
                images_list[idx].strip('.jpg\n').split('/')[-1]:

            f_not_match.write(gt_file.strip('.txt').split('/')[-1])

        #prompt
        print("{}: {}".format(idx, gt_file))
        with open(gt_file) as f_gt:
            gt_lines = f_gt.readlines()

        #clean \n & convert to a list
        gt_lines = [line.strip('\n').split(' ')[1:] for line in gt_lines]

        #convert string to float
        gt_lines = [list(map(float,gt_line))for gt_line in gt_lines]

        #clean \n images
        images_list = [img.strip('\n') for img in images_list]

        # read image size for extract area
        img = cv2.imread(os.path.join(ROOT_PATH, 'images', images_list[idx]))
        height, width, _ = img.shape
        area_img = height * width

        for x0, y0, x1, y1 in gt_lines:
            #calcul area & normalize
            area = (x1 - x0) * (y1 - y0)
            area_norm = area/area_img

            #transform to name
            area_name = areas_keys[range_areas(area_norm)]

            if area_name in dict_areas:
                dict_areas[area_name] += 1
            else:
                dict_areas[area_name] = 1

    with open('pickles/dict_areas.pkl', 'wb') as area_pkl:
        pickle.dump(dict_areas, area_pkl, protocol=pickle.HIGHEST_PROTOCOL)

    return dict_areas

def histogram_areas(dict_areas = None):


    if dict_areas is None:
        with open('pickles/dict_areas.pkl', 'rb') as areas_pkl:
            dict_areas = pickle.load(areas_pkl)

    window_title = "areas"
    plot_title = "number of objects per areas"
    x_label = "number of objects per area"
    output_path = "output/areas_wide.png"
    to_show = True
    plot_color = 'forestgreen'
    draw_plot_func(
        dict_areas,
        len(dict_areas.keys()),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
    )

def evaluation_areas_AP():

    MIN_OVERLAP = 0.5

    ROOT_PATH = '/home/alupotto/data/autel'

    # read images path
    with open(os.path.join(ROOT_PATH, 'autel.txt'), 'r') as f_imgs:
        images_list = f_imgs.readlines()
    images_list.sort()

    # read labels path
    gt_path = os.path.join(ROOT_PATH, 'new_labels')
    gt_list = glob.glob("{}/*.txt".format(gt_path))
    gt_list.sort()

    # dict areas
    areas_keys = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
    classes = dict.fromkeys(areas_keys)
    dict_areas_AP = {k: [] for k in classes}

    #pred_path = os.path.join(ROOT_PATH, 'new_predicted')
    pred_path = os.path.join(ROOT_PATH, 'predicted_085')
    APs = []

    for idx, gt_file in enumerate(gt_list):

        #prompt
        print("{}: {}".format(idx, gt_file))
        with open(gt_file) as f_gt:
            gt_lines = f_gt.readlines()

        #clean \n & convert to a list
        gt_lines = [line.strip('\n').split(' ') for line in gt_lines]
        #convert to string to float
        gt_lines = [list(map(float, gt_line)) for gt_line in gt_lines]



        #read prediction
        pred_file = gt_file.replace(gt_path.split('/')[-1], pred_path.split('/')[-1])
        with open(pred_file) as f_pred:
            f_pred = f_pred.readlines()

        # clean \n & convert to a list
        pred_lines = [line.strip('\n').split(' ') for line in f_pred]
        #convert string to float
        pred_lines = [list(map(float, pred_line)) for pred_line in pred_lines]

        if len(pred_lines) == 0:
            if len(gt_lines) != 0:
                APs.append(0)
            continue

        gt_lines = np.array(gt_lines)
        bbox_gt = gt_lines[:, 1:]


        # read image size for extract area
        img = cv2.imread(os.path.join(ROOT_PATH, 'images', images_list[idx].strip('\n')))
        height, width, _ = img.shape
        area_img = height * width

        for idx, (_, x0, y0, x1, y1) in enumerate(gt_lines):
            area = (x1 - x0) * (y1 - y0)
            area_norm = area / area_img
            # transform to name
            area_cls =range_areas(area_norm)
            #change class to area
            gt_lines[idx, 0] = area_cls



        AP_temp = {}
        correct = []
        detected = []


        tpos = len(pred_lines) * [0]

        for idx, (_, conf, x0, y0, x1, y1) in enumerate(pred_lines):
            area = (x1 - x0) * (y1 - y0)
            area_norm = area / area_img
            area_cls = range_areas(area_norm)

            # transform to name
            bbox_pred = np.array((x0, y0, x1, y1))
            bbox_pred = bbox_pred.reshape((1, bbox_pred.shape[0]))

            iou = bbox_iou(bbox_pred, bbox_gt)
            # extract index of largest overlap
            best_i = np.argmax(iou)

            if iou[best_i] > MIN_OVERLAP and area_cls == gt_lines[best_i, 0] and best_i not in detected:

                correct.append(1)
                detected.append(best_i)
                tpos[idx] = int(1)
                AP_temp[areas_keys[area_cls]] = tpos

            else:
                correct.append(0)
                tpos[idx] = int(0)

                AP_temp[areas_keys[area_cls]] = tpos


        for cls, value in AP_temp.items():

            true_positives = np.array(value)
            false_positives = 1 - true_positives

            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            recall = true_positives / len(gt_lines) if len(gt_lines) else true_positives
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
            AP = compute_ap(recall, precision)
            APs.append(AP)
            if cls in dict_areas_AP:
                dict_areas_AP[cls].append(AP)
            else:
                dict_areas_AP[cls] = AP
        print("Sample [%d/%d] AP: %.4f mAP: %.4f" % (len(APs), len(gt_list), AP, np.mean(APs)))
    dict_areas_AP['mAP'] = np.mean(APs)
    with open('pickles/dict_areas_AP.pkl', 'wb') as cls_pkl:
        pickle.dump(dict_areas_AP, cls_pkl, protocol=pickle.HIGHEST_PROTOCOL)
    
    return dict_areas_AP

def histogram_areas_AP(dict_areas_AP = None):


    if dict_areas_AP is None:
        with open('pickles/dict_areas_AP.pkl', 'rb') as cls_pkl:
            dict_areas_AP = pickle.load(cls_pkl)


    for key, value in dict_areas_AP.items():
        AP = np.mean(value)
        if math.isnan(AP):
            dict_areas_AP[key] = 0
        else:
            dict_areas_AP[key] = AP
         #   dict_areas_AP[key] = 0

#        dict_areas_AP[key] = np.mean(value)

    print(dict_areas_AP)
    mAP = dict_areas_AP['mAP']
    dict_areas_AP.pop('mAP')
    n_classes = len(dict_areas_AP.keys())
    window_title = "AP per area"
    plot_title = "mAP = {0:.2f}%".format(mAP * 100)
    x_label = "Average Precision"
    output_path = "output/areas_AP.png"
    to_show = False
    plot_color = 'royalblue'
    draw_plot_func(
        dict_areas_AP,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
    )

def main():

    if len(sys.argv) != 2:
        logging.error("Usage: python eval_areas.py --ground_truth <path_gt_folder> --predictions <path_to_predicts> --images <image file>")
        sys.exit()
    params_path = sys.argv[1]


if __name__ == '__main__':

    #count area
    dict_area = evaluation_areas()
    #histogram_areas()
    #AP areas
   # dict_classes = evaluation_areas_AP()
    #histogram_areas_AP()