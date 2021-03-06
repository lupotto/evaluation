import cv2
import os
import sys

ROOT_PATH = '/home/alupotto/data/autel/'



def show_predictions(folder_images):

	with open(os.path.join(ROOT_PATH, 'autel.txt')) as f:
		files_list = f.readlines()

	id_img = 1
	for file_path in files_list:

		if file_path.strip('\n').split('/')[-2] == folder_images:
			print( file_path.strip('\n').split('/')[-1])

			img = cv2.imread(ROOT_PATH + 'images/'+ file_path.strip('\n'))
			#print(root + item.strip('\n'))
			label_f = ROOT_PATH + 'experiments/predicted_3915/' + file_path.strip('\n').split('/')[-1].strip('.jpg') + '.txt'

			f = open(label_f)
			for i in f.readlines():
				temp = i.strip('\n').split(' ')
				x0 = float(temp[2])
				y0 = float(temp[3])
				x1 = float(temp[4])
				y1 = float(temp[5])
				#print(x0, y0, x1, y1)
				img_new = cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0,255,0), 3)

			#print(img.shape)
			cv2.imshow('img', img_new)
			cv2.waitKey(1)

			#id consecutive for generate video with ffmpeg
			animal_id = file_path.strip('\n').split('/')[-1]

			if (len(str(id_img))) == 1:
				animal_id_ = animal_id.split('_')[-2] + '_0000' + str(id_img) + '.jpg'

			elif (len(str(id_img))) == 2:
				animal_id_ = animal_id.split('_')[-2] + '_000' + str(id_img) + '.jpg'

			elif (len(str(id_img))) == 3:
				animal_id_ = animal_id.split('_')[-2] + '_00' + str(id_img) + '.jpg'

			cv2.imwrite(os.path.join(ROOT_PATH, 'sequences_out', 'predictions',
									 folder_images, animal_id_), img_new)

			id_img += 1

if __name__ == '__main__':

	show_predictions('Zebra')