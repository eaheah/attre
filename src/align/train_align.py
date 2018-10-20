import numpy as np
import cv2
import dlib
import os
from matplotlib import pyplot as plt
from align import Aligner



def resize(image, new_width, is_square=False, interpolation=cv2.INTER_AREA):
    if is_square:
        return cv2.resize(image, (new_width, new_width), interpolation=interpolation)
    else:
        img_height, img_width = image.shape[:2]
        new_height = int(new_width / float(img_width) * img_height)
        return cv2.resize(image, (new_width, new_height))


class TrainAligner(Aligner):
    def __init__(self, output_path, df, save=False):
        self.df = df # might be a memory issue (generate together?)
        self.save = save
        self.output_path = output_path
        self.output_left_eye = (0.35, 0.35)
        self.output_width = 256
        # self.output_height = self.get_height()
        self.bbox_shape = 95

    def get_height(self):
        # print (self.input_width)
        # print(self.input_height)
        new_height = int(self.output_width / float(self.input_width) * self.input_height)
        return new_height


    def resize_point(self, old_x, old_y):
        new_x = old_x * (float(self.output_width) / float(self.input_width))
        new_y = old_y * (float(self.output_height) / float(self.input_height))
        return int(new_x), int(new_y)

    def save_figure(self, original_image, aligned_image, box_args, image_path, row):
        image_path = self.mod_path(image_path)
        filename = os.path.join(self.output_path, image_path)
        rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        for center in (self.left_center, self.right_center):
            cv2.circle(rgb_image, center, 5, color=(0,0,255), thickness=-1)
        original_image = self.bound(rgb_image, box_args)
        aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)




        plt.figure(figsize=(15,15))
        plt.subplot(121)
        plt.imshow(original_image)
        plt.subplot(122)
        plt.imshow(aligned_image)
        plt.savefig(filename)

    def get_datum(self, row, label):
        return row.get(label).values[0]

    def get_coordinates(self, row):
        lefteye = self.get_datum(row, 'lefteye_x'), self.get_datum(row, 'lefteye_y')
        righteye = self.get_datum(row, 'righteye_x'), self.get_datum(row, 'righteye_y')
        return lefteye, righteye


    def get_rotation(self, coordinates):
        left_center = coordinates[1]
        right_center = coordinates[0]

        self.left_center = tuple(left_center)
        self.right_center = tuple(right_center)

        x = right_center[0] - left_center[0]
        y = right_center[1] - left_center[1]

        angle = np.degrees(np.arctan2(y, x)) - 180

        new_right_x = 1.0 - self.output_left_eye[0]

        distance = np.sqrt((x**2) + (y**2))
        new_distance = new_right_x - self.output_left_eye[0]
        new_distance = new_distance * self.output_width
        scale = new_distance / distance

        center = (left_center[0] + right_center[0]) // 2, (left_center[1] + right_center[1]) // 2
        return center, angle, scale

    def align(self, img, img_gray, bbox, image_path, row):
        box_args = self.get_bbox(bbox)
        self.input_height, self.input_width = img.shape[:2]
        self.output_height = self.get_height()
        coordinates = self.get_coordinates(row)

        center, angle, scale = self.get_rotation(coordinates)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        # cv2.transform(pointsToTransform, M)

        # print(rotation_matrix)

        # landmarks_rs = np.reshape(np.array(landmarks), (68, 1, 2))
        # landmarks_t = cv2.transform(landmarks_rs, transform)

        self.new_left = cv2.transform(np.reshape(np.array(self.left_center), (1,1,2)), rotation_matrix)
        self.new_right = cv2.transform(np.reshape(np.array(self.right_center), (1,1,2)), rotation_matrix)

        # print(center, angle, scale)

        tX = self.output_width * 0.5
        tY = self.output_height * self.output_left_eye[1]

        rotation_matrix[0, 2] += (tX - center[0])
        rotation_matrix[1,2] += (tY - center[1])

        aligned_face = cv2.warpAffine(img, rotation_matrix, (self.output_width, self.output_height), flags=cv2.INTER_CUBIC)

        if self.save:
            self.save_figure(img, aligned_face, box_args, image_path, row)
        return aligned_face 

    def generate(self, path, face_detector):
        for image_path in os.listdir(path):

            image = cv2.imread(os.path.join(path, image_path))
            print(image.shape)
            # image = resize(image, new_width=800)
            row = self.df.loc[df['image_id'] == image_path]
            
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            boxes = face_detector(gray_image, 1)# number of times image is upsampled (more means more faces, but may not need much here)

            if boxes:
                assert len(boxes) == 1 # dataset is single faces

                yield face_aligner.align(image, gray_image, boxes[0], image_path, row)         

if __name__ == '__main__':    
    import pandas as pd

    face_detector = dlib.get_frontal_face_detector()

    path = '/vagrant/imgs/small/images'
    landmarks_path = '/vagrant/imgs/small/list_landmarks_align_celeba.csv'
    directory = os.listdir(path)
    df = pd.read_csv(landmarks_path)
    face_detector = dlib.get_frontal_face_detector()
    face_aligner = TrainAligner(output_path='/vagrant/src/align/aligned-train/',df=df, save=True)
    generator = face_aligner.generate('/vagrant/imgs/small/images/', face_detector)
    
    for i in generator:
        pass
