import numpy as np
import cv2
import dlib
import os
import pandas as pd

def resize(image, new_width, is_square=False, interpolation=cv2.INTER_AREA):
    if is_square:
        return cv2.resize(image, (new_width, new_width), interpolation=interpolation)
    else:
        img_height, img_width = image.shape[:2]
        new_height = int(new_width / float(img_width) * img_height)
        return cv2.resize(image, (new_width, new_height))

class Aligner:
    def __init__(self, output_path, predictor_path, df, save=False):
        self.df = df
        self.save = save
        self.output_path = output_path
        self.predictor = dlib.shape_predictor(predictor_path)
        self.output_left_eye = (0.35, 0.35)
        # self.output_width = 256
        # self.output_height = 256
        self.bbox_shape = 95

    def get_bbox(self, dlib_bbox):
        left_x = dlib_bbox.left()
        top_y = dlib_bbox.top()
        width = dlib_bbox.width()
        height = dlib_bbox.height()

        return left_x, top_y, width, height

    def mod_path(self, image_path):
        return image_path.replace('jpg', 'png') 

    def get_height(self):
        # print (self.input_width)
        # print(self.input_height)
        new_height = int(self.output_width / float(self.input_width) * self.input_height)
        return new_height

    def save_figure(self, original_image, aligned_image, box_args, image_path):
        image_path = self.mod_path(image_path)
        # print(original_image.dtype)
        filename = os.path.join(self.output_path, image_path)
        # rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        # for center in (self.left_center, self.right_center):
        #     cv2.circle(rgb_image, center, 5, color=(0,0,255), thickness=-1)
        # original_image = self.bound(rgb_image, box_args)

        gray_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)

        boxes = face_detector(gray_image, 1)

        if boxes:
            if not len(boxes) == 1:
                box = self.get_largest_bbox(boxes)
            else:
                box = boxes[0]   

            box_args = self.get_bbox(box)
            try:
                aligned_image = self.bound(aligned_image, box_args)
                cv2.imwrite(filename, aligned_image)
            except:
                pass

            # plt.figure(figsize=(15,15))
            # plt.subplot(121)
            # plt.imshow(original_image)
            # plt.subplot(122)
            # plt.imshow(aligned_image)
            # plt.savefig(filename)
        else:
            pass

    def get_datum(self, row, label):
        return row.get(label).values[0]

    def get_coordinates(self, row):
        # coordinates = np.zeros((prediction.num_parts, 2), dtype="int")
        # for i in range(0, prediction.num_parts):
        #     coordinates[i] = (prediction.part(i).x, prediction.part(i).y)
        # return coordinates
        lefteye = self.get_datum(row, 'lefteye_x'), self.get_datum(row, 'lefteye_y')
        righteye = self.get_datum(row, 'righteye_x'), self.get_datum(row, 'righteye_y')
        return lefteye, righteye

    def bound(self, image, args):
        return resize(image[args[1]:args[1] + args[3], args[0]:args[0] + args[2]], self.bbox_shape, is_square=True)

    def get_rotation(self, coordinates):
        # left_eye = coordinates[42:48]
        # right_eye = coordinates[36:42]

        # left_center = left_eye.mean(axis=0).astype("int")
        # right_center = right_eye.mean(axis=0).astype('int')
        # print("left center, right center")
        # print(left_center, right_center)

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

        # predition = self.predictor(img_gray, bbox)
        # coordinates = self.get_coordinates(predition)
        coordinates = self.get_coordinates(row)
        
        center, angle, scale = self.get_rotation(coordinates)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        # print(center, angle, scale)

        tX = self.output_width * 0.5
        tY = self.output_height * self.output_left_eye[1]

        rotation_matrix[0, 2] += (tX - center[0])
        rotation_matrix[1,2] += (tY - center[1])

        aligned_face = cv2.warpAffine(img, rotation_matrix, (self.output_width, self.output_height), flags=cv2.INTER_CUBIC)

        if self.save:
            self.save_figure(img, aligned_face, box_args, image_path)
        return aligned_face

    def get_largest_bbox(self, boxes):
        area = 0
        box = None
        for box in boxes:
            box_area = box.width() * box.height()
            box = box if box_area > area else box
            area = box_area if box_area > area else area 
        return box         

    def generate(self, path, face_detector):
        for image_path in os.listdir(path):
            print(image_path)
            image = cv2.imread(os.path.join(path, image_path))
            # image = resize(image, new_width=800)
            row = self.df.loc[df['image_id'] == image_path]

            self.output_width, self.output_height = image.shape[:2]
            
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            boxes = face_detector(gray_image, 1)# number of times image is upsampled (more means more faces, but may not need much here)

            if boxes:
                if not len(boxes) == 1:
                    box = self.get_largest_bbox(boxes)
                else:
                    box = boxes[0]              
                yield face_aligner.align(image, gray_image, box, image_path, row) 
            else:
                yield None

    def single_image(self, path, image_path, face_detector):
        image = cv2.imread(os.path.join(path, image_path))
        # image = resize(image, new_width=800)
        self.output_width, self.output_height = image.shape[:2]  
        row = self.df.loc[df['image_id'] == image_path]      
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        boxes = face_detector(gray_image, 1)# number of times image is upsampled (more means more faces, but may not need much here)

        if boxes:
            if not len(boxes) == 1:
                box = self.get_largest_bbox(boxes)
            else:
                box = boxes[0]              
            return face_aligner.align(image, gray_image, box, image_path, row) 
        else:
            return None
if __name__ == '__main__':
    landmarks_path = '/vagrant/imgs/list_landmarks_align_celeba.csv'
    df = pd.read_csv(landmarks_path)

    face_aligner = Aligner(
        output_path='/vagrant/src/training_data/aligned/',
        predictor_path='/vagrant/src/shape_predictor_68_face_landmarks.dat',
        df=df,
        save=True)
    face_detector = dlib.get_frontal_face_detector()

    generator = face_aligner.generate('/vagrant/imgs/orig_images/img_align_celeba/', face_detector)

    import time
    # s = time.time()
    try:
        for _ in generator:
            pass
    except:
        print("\a")
        raise
    print("\a")

    # face_aligner.single_image('/vagrant/imgs/orig_images/img_align_celeba/', '202379.jpg', face_detector)