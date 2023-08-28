import cv2
from facenet_pytorch import MTCNN
import torch
from datetime import datetime
import os

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

IMG_PATH = './data/test_images/'
# count = 100
usr_name = input("Input ur name: ")
USR_PATH = os.path.join(IMG_PATH, usr_name)
leap = 1

mtcnn = MTCNN(margin = 20, keep_all=False, select_largest = True, post_process=False, device = device)
# cap = cv2.VideoCapture(r'C:\Users\Admin\OneDrive\Desktop\New folder (2)\FaceNet-Infer\data\video-to-train\4817238053258128907.mp4')
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
image_folder = './data/picture-to-train/'
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg'))]
count = len(image_files)
image = cv2.imread(image_folder + 'IMG_20200303_063033.jpg')


new_width = int(image.shape[1] * 1/5)
new_height = int(image.shape[0] * 1/5)
image = cv2.resize(image, (new_width, new_height))

path = str(USR_PATH+'/{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-")+str(count)))
face_img = mtcnn(image, save_path = path)
print("face_img", face_img)



# count = len(image_files)

# for image_file in image_files:
#     if image_file and leap%2:
#         path = str(USR_PATH+'/{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-")+str(count)))
#         image = cv2.imread(image_folder + image_file)
#         #thay đổi tỷ lệ
#         new_width = int(image.shape[1] * 1/5)
#         new_height = int(image.shape[0] * 1/5)

#         # Sử dụng hàm cv2.resize() để thay đổi kích thước hình ảnh
#         image = cv2.resize(image, (new_width, new_height))

#         face_img = mtcnn(image, save_path = path)
#         count-=1
#     leap+=1
#     # cv2.imshow('Face Capturing', frame)
#     if cv2.waitKey(1)&0xFF == 27:
#         break
# # cap.release()
# cv2.destroyAllWindows()