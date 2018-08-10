import os
import cv2

file = open('/data/shared/autel/total_autel.txt')
root = '/home/alupotto/autel/'

file_w = open('/home/alupotto/autel/autel.txt', 'w')

for item in file.readlines():
        temp = item.strip('\n').split('/')[-1]
        index = item.split('/')[5] + '/' + temp

        file_w.write(index + '\n')
        image_file = root + 'images/' + index
        print(image_file)

        img = cv2.imread(image_file)
        h, w, c = img.shape
        label_file = root + 'labels/' + temp.strip('.jpg') + '.txt'
        n_label_file = root + 'new_labels/' + temp.strip('.jpg') + '.txt'
        f = open(label_file)
        f_w = open(n_label_file, 'w')
        for line in f.readlines():
                t = line.split(' ')
                cls = int(t[0])
                l1 = float(t[1])
                l2 = float(t[2])
                l3 = float(t[3])
                l4 = float(t[4])

                t1 = (l1 * w + 1) * 2
                t2 = l3 * w
                t3 = (l2 * h + 1) * 2
                t4 = l4 * h

                x1 = (t1 + t2) / 2
                x0 = t1 - x1
                y1 = (t3 + t4) / 2
                y0 = t3 - y1

                new_l = str(cls) + ' ' + str(x0) + ' ' + str(y0) + ' ' + str(x1) + ' ' + str(y1) + '\n'
                f_w.write(new_l)
