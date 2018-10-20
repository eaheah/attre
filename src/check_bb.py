import os
import cv2
import dlib

def get_largest(boxes):
	area = 0
	box = None
	for box in boxes:
		box_area = box.width() * box.height()
		box = box if box_area > area else box
		area = box_area if box_area > area else area 
	return box 

def count(length=None):
	images_path = "/vagrant/imgs/orig_images/img_align_celeba"
	directory = os.listdir(images_path)
	length = len(directory) if not length else length
	face_detector = dlib.get_frontal_face_detector()
	i = 1
	width = 0
	height = 0
	num_boxes = []
	for image_path in directory:
		if i == length:
			break
		print("Image {} of {}".format(i, length))
		i += 1
		image = cv2.imread(os.path.join(images_path, image_path))
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		boxes = face_detector(gray_image, 1)

		# print(boxes)
		if boxes:
			if not len(boxes) == 1:
				box = get_largest(boxes)
			else:
				box = boxes[0]
			width += box.width()
			height += box.height()


		else:
			num_boxes.append(image_path)
			

	return width, height, length, num_boxes

if __name__ == '__main__':
	width, height, length, num_boxes = count()
	total = length - len(num_boxes)

	width2, height2, length2, num_boxes2 = count()
	total2 = length2 - len(num_boxes2)

	width3, height3, length3, num_boxes3 = count()
	total3 = length3 - len(num_boxes3)

	print(width)
	print(width/total)
	print(height/total)
	print(num_boxes)
	print()

	print(width2/total2)
	print(height2/total2)
	print(num_boxes2)
	print()

	print(width3/total3)
	print(height3/total3)
	print(num_boxes3)
	print()

	assert sorted(num_boxes) == sorted(num_boxes2)
	assert sorted(num_boxes2) == sorted(num_boxes3)
	

	print("\a")