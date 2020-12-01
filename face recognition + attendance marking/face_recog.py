import cv2
import numpy as np
import face_recognition

img_abu = face_recognition.load_image('images/abubakar.jpg')
img_abu = cv2.cvtColor(img_abu, cv2.COLOR_BGR2RGB)

img_test = face_recognition.load_image('images/abubakar_test.jpg')
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)


face_location = face_recognition.face_location(img_abu)
encode_abu = face_recognition.face_encodings(img_abu)[0]
cv2.rectangle(img_abu, (face_location[3], face_location[0]), (face_location[1], face_location[2]), (255, 0, 255, 0), 2)

face_location_test = face_recognition.face_location(img_test)
encode_test = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_abu, (face_location_test[3], face_location_test[0]), (face_location_test[1], face_location_test[2]), (255, 0, 255, 0), 2)

results = face_recognition.compare_faces([encode_abu], encode_test)
face_dis = face_recognition.face_distance([encode_abu], encode_test)
print(results, face_dis)

cv2.putText(img_test, f'{results} {round(face_dis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

cv2.imshow('Abubakar', img_abu)
cv2.imshow('Abubakar', img_test)
cv2.waitKey(0)