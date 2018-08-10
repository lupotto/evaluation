
import cv2
import glob
import sys
import os

file_img_path = '/home/alupotto/resources/autel/autel.txt'
root = '/home/alupotto/resources/autel/'

#file_w = open('/home/jimuyang/data/autel/autel.txt', 'w')


path_predictions = '/home/alupotto/resources/autel/predicted'

label_files = sorted(glob.glob('{}**/*.txt'.format(path_predictions)))

with open(file_img_path) as f:
    img_files = f.readlines()

    for img_path in img_files:

        id_img =img_path.strip('\n')
        image_file = os.path.join(root,'images',id_img)

        img = cv2.imread(image_file)
        h, w, _ = img.shape

        print(image_file)
        new_predicted = root + 'new_predicted/' + id_img.split('/')[-1].replace('.jpg','.txt')
        file_new = open(new_predicted,'w')

        old_predicted = root + 'predicted/' + id_img.split('/')[-1].replace('.jpg','.txt')
        file_old = open(old_predicted)

        for line in file_old.readlines():
            line = line.strip('\n')
            value = line.split(' ')

            cls = int(value[0])
            conf = float(value[1])
            l1 = float(value[2])
            l2 = float(value[3])
            l3 = float(value[4])
            l4 = float(value[5])

            t1 = (l1 * w + 1) * 2
            t2 = l3 * w
            t3 = (l2 * h + 1) * 2
            t4 = l4 * h

            x1 = (t1 + t2) / 2
            x0 = t1 - x1
            y1 = (t3 + t4) / 2
            y0 = t3 - y1

            file_new.write(str(cls) + ' ' + str(conf)+ ' ' + str(x0) + ' ' + str(y0) + ' ' + str(x1) + ' ' + str(y1) + '\n')


