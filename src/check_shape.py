import os
import cv2
images_path = "/vagrant/imgs/orig_images/img_align_celeba"
directory = os.listdir(images_path)
length = len(directory)
i = 1
for image_path in directory:
	print("Image {} of {}".format(i, length))
	i += 1
	image = cv2.imread(os.path.join(images_path, image_path))
	assert image.shape == (218, 178, 3)