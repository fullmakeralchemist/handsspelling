import cv2 
import uuid
import os
import time

labels = ['thumbsup', 'hi', 'loveyou', 'livelong'] #modify the labels as you need
number_imgs = 20#number of images that will take

IMAGES_PATH = os.path.join('images')

if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)

for label in labels:#Loop that creates folders for the labels
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.makedirs(path)

for label in labels:#Loop that takes the pictures for each label
    cap = cv2.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(10)#Time before start taking pictures
    for imgnum in range(number_imgs):
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH, label, label + '.' + '{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)#Time between each camera shot

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

