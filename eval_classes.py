import json
import os
import sys
import glob
import shutil
import logging
import argparse
import matplotlib.pyplot as plt
import operator
import numpy as np
import pickle
from collections import defaultdict
from tools.utils import draw_plot_func,  compute_ap, bbox_iou
from tools.utils import load_classes



def mainParser():

    parser = argparse.ArgumentParser(description='Autel Dataset')
    parser.add_argument('--ground_truth', type=str,  help='path ground_truth labels')
    parser.add_argument('--predictions' , type=str,  help='path prediction files')
    parser.add_argument('--images'      , type=str,  help='path file image')
    opt = parser.parse_args()

    return opt


def load_classes(classes_path):

    fp = open(classes_path, "r")
    names = fp.read().split("\n")[:-1]

    return names

#TODO: change paths to arguments (mainParser)
def evaluation_gt():

    #read labels path
    gt_path = '/home/alupotto/data/autel/new_labels'
    gt_list = glob.glob("{}/*.txt".format(gt_path))

    #declare dictionary gt
    classes = load_classes('data/autel.names')
    dict_gt = dict.fromkeys(classes, 0)

    for gt_file in gt_list:

        with open(gt_file) as f_gt:
            gt_lines = f_gt.readlines()

        #clean \n & convert to a list
        gt_lines = [line.strip('\n').split(' ') for line in gt_lines]
        #convert to string to float
        gt_lines = [ int(gt_line[0])for gt_line in gt_lines]


        for class_id in gt_lines:
            class_name = classes[class_id]

            if class_name in dict_gt:
                dict_gt[class_name] += 1

            else:
                dict_gt[class_name] = 1

    with open('pickles/dict_gt.pkl', 'wb') as gt_pkl:
        pickle.dump(dict_gt, gt_pkl, protocol=pickle.HIGHEST_PROTOCOL)

    return dict_gt

def histogram_classes_gt(dict_gt = None):


    if dict_gt is None:
        with open('pickle/dict_gt.pkl', 'rb') as gt_pkl:
            dict_gt = pickle.load(gt_pkl)

    classes = load_classes('data/autel.names')
    gt_path = '/home/alupotto/data/autel/new_labels'
    gt_list = glob.glob("{}/*.txt".format(gt_path))

    window_title = "ground truth"
    plot_title = "Ground-Truth\n"
    plot_title += "(" + str(len(gt_list)) + " files and " + str(len(classes)) + " classes)"
    x_label = "number ground truth objects"
    output_path = "output/ground_truth.png"
    to_show = False
    plot_color = 'forestgreen'
    draw_plot_func(
        dict_gt,
        len(classes),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
    )
    return dict_gt

def evaluation_classes(gt_path, pred_path):

    MIN_OVERLAP = 0.5
    ROOT_PATH =  '/home/alupotto/data/autel/'


    #AP: 0.1667(0.0556)
    #AP: 0.85

    #read labels path
    #gt_path = '/home/alupotto/data/autel/new_labels'
    gt_list = glob.glob("{}/*.txt".format(gt_path))
    gt_list.sort()

    #paths predicted
    #pred_path = '/home/alupotto/data/autel/new_predicted'
    #pred_path = '/home/alupotto/data/autel/experiments/predicted_2420'



    APs = []

    #declare dictionary classes
    classes = load_classes('data/autel.names')
    AP_classes = dict.fromkeys(classes)
    AP_classes = {k: [] for k in AP_classes}


    for idx, gt_file in enumerate(gt_list):

        #read images path
        with open(gt_file) as f_gt:
            gt_lines = f_gt.readlines()

        #clean \n & convert to a list
        gt_lines = [line.strip('\n').split(' ') for line in gt_lines]
        #convert to string to float
        gt_lines = [list(map(float, gt_line)) for gt_line in gt_lines]

        #extract gt_id
        gt_id = gt_file.split('/')[-1]
        pred_file = os.path.join(pred_path, gt_id)

        #read prediction
        with open(pred_file) as f_pred:
            f_pred = f_pred.readlines()

        # clean \n and covnert to a list
        pred_lines = [line.strip('\n').split(' ') for line in f_pred]
        #convert string to float
        pred_lines = [list(map(float, pred_line)) for pred_line in pred_lines]


        gt_lines = np.array(gt_lines)
        bbox_gt = gt_lines[:,1:]

        AP_temp = {}
        correct = []
        detected = []

        if len(pred_lines) == 0:
            if len(gt_lines) != 0:
                tpos_non = len(gt_lines) * [0]
                for idx, (cls_gt, _, _, _, _) in enumerate(gt_lines):
                    cls_name = classes[int(cls_gt)]
                    tpos_non[idx] = int(0)
                    APs.append(0)
                    AP_temp[cls_name] = tpos_non

                continue

        tpos = len(pred_lines) * [0]

        for idx, (cls_pred, conf, x1, y1, x2, y2) in enumerate(pred_lines):

            bbox_pred = np.array((x1, y1, x2, y2))
            bbox_pred = bbox_pred.reshape((1, bbox_pred.shape[0]))

            iou = bbox_iou(bbox_pred, bbox_gt)
            # extract index of largest overlap
            best_i = np.argmax(iou)
            cls_name = classes[int(cls_pred)]
            if iou[best_i] > MIN_OVERLAP and cls_pred == gt_lines[best_i, 0] and best_i not in detected:

                correct.append(1)
                detected.append(best_i)
                tpos[idx] = int(1)
                AP_temp[cls_name] = tpos

            else:
                correct.append(0)
                tpos[idx] = int(0)

                AP_temp[cls_name] = tpos

        for cls, value in AP_temp.items():

            true_positives = np.array(value)
            false_positives = 1 - true_positives

            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            recall = true_positives / len(gt_lines) if len(gt_lines) else true_positives
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
            AP = compute_ap(recall, precision)
            APs.append(AP)

            if cls in AP_classes:
                AP_classes[cls].append(AP)
            else:
                AP_classes[cls] = AP


        print("Sample [%d/%d] AP: %.4f mAP: %.4f" % (len(APs), len(gt_list), AP, np.mean(APs)))
    AP_classes['mAP'] = np.mean(APs)
    with open('pickles/dict_classes.pkl', 'wb') as cls_pkl:
        pickle.dump(AP_classes, cls_pkl, protocol=pickle.HIGHEST_PROTOCOL)

    return AP_classes

def histogram_classes_AP(dict_classes = None):


    if dict_classes is None:
        with open('pickles/dict_classes.pkl', 'rb') as cls_pkl:
            dict_classes = pickle.load(cls_pkl)

    classes = load_classes('data/autel.names')
    for key, value in dict_classes.items():
        dict_classes[key] = np.mean(value)

    mAP = dict_classes['mAP']
    dict_classes.pop('mAP')

    window_title = "AP per class"
    plot_title = "mAP = {0:.2f}%".format(mAP * 100)
    x_label = "Average Precision"
    output_path = "output/classes_AP.png"
    to_show = True
    plot_color = 'royalblue'
    draw_plot_func(
        dict_classes,
        len(classes),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
    )
    return dict_classes


def main():


    args = mainParser()

    dict_classes = evaluation_classes(args.ground_truth, args.predictions)
    histogram_classes_AP(dict_classes)

if __name__ == '__main__':


    main()
    #ground truth
    #dict_gt = evaluation_gt()
    #histogram_classes_gt(dict_gt)
    #AP classes
    #dict_classes = evaluation_classes(args['ground_truth'])
    #histogram_classes_AP(dict_classes)