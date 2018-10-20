import cv2
import os

class ImageData:
	def __init__(self, images_path, gt_path, do_shuffle=False):
		self.images_path = images_path
		self.gt_path = gt_path
		self.image_files = [f for f in os.listdir(images_path) \
							if os.path.isfile(os.path.join(images_path, f))]
		assert(len(self.image_files) > 0)
		if do_shuffle:
			random.seed(11)
		else:
			self.image_files.sort()
		self.num_examples = len(self.image_files)
		self.do_shuffle = do_shuffle

	def __iter__(self):
		if self.do_shuffle:
			random.shuffle(self.image_files)
			files = self.image_files 
		else:
			files = self.image_files 
		for f in files:
			image = cv2.imread(os.path.join(self.images_path, f))
			if image is None:
				exit(0)
			# if len(image.shape) > 1:
			# 	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)# get rid of this
			print(image.shape)
			yield('a')

if __name__ == "__main__":
	image_data = ImageData('/vagrant/imgs/images/small', None, False)
	for _ in image_data:
		pass