import json
import os
import sys
import glob
import shutil
import matplotlib.pyplot as plt
import operator
import numpy as np
import argparse
from tools.utils import draw_plot_func,  compute_ap, bbox_iou
from tools.utils import load_classes


def mainParser():
    parser = argparse.ArgumentParser(description='Autel Dataset')
    parser.add_argument('--ground_truth', type=str,  help='path ground_truth labels')
    parser.add_argument('--predictions' , type=str,  help='path prediction files')
    parser.add_argument('--images'      , type=str,  help='path file image')
    opt = parser.parse_args()

    return opt


def compute_mAP(gt_path, pred_path):

    MIN_OVERLAP = 0.5
    root_path =  '/home/alupotto/autel/'
    #gt_path = '/home/alupotto/data/autel/new_labels'
    gt_list = glob.glob("{}/*.txt".format(gt_path))
    pred_list = glob.glob("{}/*.txt".format(pred_path))

    AP = 0
    APs = []

    c = 0

    gt_list.sort()
    for gt_file in gt_list:

        gt_id = gt_file.split('/')[-1]


        with open(gt_file) as f_gt:
            gt_lines = f_gt.readlines()

        #clean \n and covnert to a list
        gt_lines = [line.strip('\n').split(' ') for line in gt_lines]
        #convert to string to float
        gt_lines = [list(map(float, gt_line)) for gt_line in gt_lines]

        # extract gt_id
        gt_id = gt_file.split('/')[-1]
        pred_file = os.path.join(pred_path, gt_id)

        #read prediction
        #pred_file = gt_file.replace(gt_path.split('/')[-1], pred_path.split('/')[-1])
        with open(pred_file) as f_pred:
            f_pred = f_pred.readlines()

        # clean \n and covnert to a list
        pred_lines = [line.strip('\n').split(' ') for line in f_pred]
        #convert string to float
        pred_lines = [list(map(float, pred_line)) for pred_line in pred_lines]

        if len(pred_lines) == 0:
            if len(gt_lines) != 0:
                APs.append(0)
            continue

        gt_lines = np.array(gt_lines)
        bbox_gt = gt_lines[:,1:]


        correct = []
        detected = []

        for cls_pred, conf, x1, y1, x2, y2 in pred_lines:
            bbox_pred = np.array((x1, y1, x2, y2))
            bbox_pred = bbox_pred.reshape((1, bbox_pred.shape[0]))

            iou = bbox_iou(bbox_pred, bbox_gt)
            # extract index of largest overlap
            best_i = np.argmax(iou)

            if iou[best_i] > MIN_OVERLAP and cls_pred == gt_lines[best_i, 0] and best_i not in detected:
                correct.append(1)
                detected.append(best_i)
            else:
                correct.append(0)


        true_positives = np.array(correct)
        false_positives = 1 - true_positives

        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)


        recall = true_positives / len(gt_lines) if len(gt_lines) else true_positives
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        AP = compute_ap(recall, precision)
        APs.append(AP)


        print("Sample [%d/%d] AP: %.4f mAP: %.4f" % (len(APs),len(gt_list), AP, np.mean(APs)))


    print("Mean Average Precision: %.4f" % np.mean(APs))



def main():

    args = mainParser()
    print(args)
    compute_mAP(args.ground_truth, args.predictions)


if __name__ == '__main__':
    main()
