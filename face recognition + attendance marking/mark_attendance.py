import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# folder name for all employees/workers
path = 'images'
images = []
class_names = []
my_list = os.listdir(path)
# print(my_list)

# a simple construct to extract all image names as labels
# it is assumed that all image names are respective names of individual i.e employees of the current org
for cl in my_list:
	# read image
	cur_img = cv2.imread(f'{path}/{cl}')
	# store image in a list
	images.append(cur_img)
	# get label names
	class_names.append(os.path.splittext(cl)[0])
# print(class_names)

# compute encodings of faces present in the 'images' folder
def compute_encodings(images):
	encoded_list = []
	# loop thru all images
	for img in images:
		# convert orig img to gray for ease of recog
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# encode images using face_recog algo face_encodings
		# face_encoding is a pretrained algo that has been trained to detect faces and give mapping to entities that make up a face about 128 of such entities
		# note that this entities are numbers as that what a computer feeds on
		# since it returns an array/list subsetting is required hence [0]
		encoded = face_recognition.face_encodings(img[0])
		encoded_list.append(encoded)
		# return all encodings for all images in the dataset
	return encoded_list

# function that reads and writes to a csv file to mark attendance for employees
def mark_attendance(name):
	with open('attendance.csv', 'r+') as f:
		present_stud = f.readlines()
		student_name = []
		# print(present_stud)
		for line in present_stud:
			entry = line.split(',')
			student_name.append(entry[0])
		if name not in student_name:
			now = datetime.now()
			dt_str = now.strftime("%H:%M:%S")
			f.writelines(f'\n{name}, {df_str}') 

# calculate all encodings and store in a list
encod_list_images = compute_encodings(images)
print('Encoding Complete')

# now open webcam
video_capture = cv2.video_capture(0)

# an infinite loop to keep getting frames aka images
while True:
	success, img = video_capture.read()
	# resize for faster computation and recognition
	img_small = cv2.resize(img, (0,0), None, 0.25,0.25)
	# convert each frame to gray
	img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

	# recognize/ detect face for each frame then encode
	face_curframe = face_recognition.face_locations(img_small)
	encode_curframe = face_recognition.face_encodings(img, face_curframe)

	# for each frame
	for encode_face, face_loc in zip(encode_curframe, face_curframe):
		# 1. get matches i.e with the already encoded list
		matches = face_recognition.compare_faces(encod_list_images, encode_face)
		# 2. compute similarities aka how much do they differ
		face_dis = face_recognition.face_distance(encod_list_images, encode_face)
		# print(face_dis)
		# get only the image with the lowest distance value
		match_index = np.argmin(face_dis)


		# if since matches returns all faces detected get the match with the lowest distance
		if matches[match_index]:
			# get the name of such image
			name = class_names[match_index].upper()
			# print(name)
			y1,x2,y2,x2 = face_loc
			# convert back to orig size for image
			x1*=4
			x2*=4
			y1*=4
			# draw the rectangle around the face
			cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
			cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
			# include the text
			cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
			# and mark attendance
			mark_attendance(name)

	# keep rendering images
	cv2.imshow("WebCam", img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
        break   
        # to quit
        video_capture.release()
        cv2.destroyAllWindows()