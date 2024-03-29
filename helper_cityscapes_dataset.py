from glob import glob
import os
import random
import scipy.misc
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import time
import tensorflow as tf
from urllib.request import urlretrieve
from tqdm import tqdm
import os.path
import shutil
import zipfile

# Label name and its corresponding color code
Label = namedtuple('Label', ['name', 'color'])
label_classes = [
	Label('unlabelled', (0,0,0)),
	Label('dynamic', (111, 74,  0)),
	Label('ground', ( 81,  0, 81)),
	Label('road', (128, 64,128)),
	Label('sidewalk', (244, 35,232)),
	Label('parking', (250,170,160)),
	Label('rail track', (230,150,140)),
	Label('building', ( 70, 70, 70)),
	Label('wall', (102,102,156)),
	Label('fence', (190,153,153)),
	Label('guard rail', (180,165,180)),
	Label('bridge', (150,100,100)),
	Label('tunnel', (150,120, 90)),
	Label('pole', (153,153,153)),
	Label('traffic light', (250,170, 30)),
	Label('traffic sign', (220,220,  0)),
	Label('vegetation', (107,142, 35)),
	Label('terrain', (152,251,152)),
	Label('sky', ( 70,130,180)),
	Label('person', (220, 20, 60)),
	Label('rider', (255,  0,  0)),
	Label('car', (0,  0,142)),
	Label('truck', (0,  0, 70)),
	Label('bus', (0, 60,100)),
	Label('caravan', (0,  0, 90)),
	Label('trailer', (0,  0,110)),
	Label('train', (0, 80,100)),
	Label('motorcycle', ( 0,  0,230)),
	Label('bicycle', (119, 11, 32))
	]

# Loads data to the memory
def load_data(image_paths, label_paths, data_type):
	image_files = glob(image_paths + data_type + '/**/*.png')
	label_files = glob(label_paths + data_type + '/**/*color.png')

	gt_images = []
	for img in image_files:
		img_base = os.path.basename(img)
		img_city = os.path.basename(os.path.dirname(img))
		label_base = img_base.replace('leftImg8bit.png', "gtFine_color.png")
		label = label_paths + data_type + '/' + img_city + '/' + label_base

		gt_images.append(label)

	return image_files, gt_images

# Generate batch images for training
def gen_batches_fn(img_shape, image_paths, label_paths, data_type):

	def get_batches_fn(batch_size):

		image_files = glob(image_paths + data_type + '/**/*.png')
		label_files = glob(label_paths + data_type + '/**/*color.png')
	
		gt_images, train_images = [], []
		for img in image_files: #[0:79]
			img_base = os.path.basename(img)
			img_city = os.path.basename(os.path.dirname(img))
			label_base = img_base.replace('leftImg8bit.png', "gtFine_color.png")
			# Changing the last term in the training images to the label base because they have the same name up to that point. 
			label = label_paths + data_type + '/' + img_city + '/' + label_base

			train_images.append(img)
			gt_images.append(label)

		train_image_paths, gt_image_paths = shuffle(train_images, gt_images)

		for batch_i in range(0, len(train_image_paths), batch_size):

			train_images, gt_images = [], []

			for img, label in zip(train_image_paths[batch_i:batch_i+batch_size], gt_image_paths[batch_i:batch_i+batch_size]):
				 image = scipy.misc.imresize(scipy.misc.imread(img), img_shape)
				 gt_image = scipy.misc.imresize(scipy.misc.imread(label, mode='RGB'), img_shape)
				 label_bg = np.zeros([img_shape[0], img_shape[1]], dtype=bool)
				 label_list = []
				 for l in label_classes[1:]:

				 	current_class = np.all(gt_image == np.array(l.color), axis=2)
				 	label_bg = current_class | label_bg
				 	label_list.append(current_class)

				 # ~ changes 0 to 1 and 1 to 0 so we find everything else not considered a class and stack this
				 # onto the label_list.
				 label_bg = ~label_bg 
				 # Now we stack labels depth wise. For example, 2 classes would result in a shape (256, 512, 3) where
				 # each depth slice (pixel) might look like [False, False, True] or [0, 0, 1].
				 label_all = np.dstack([label_bg, *label_list])

				 train_images.append(image)
				 gt_images.append(label_all)


			yield np.array(train_images), np.array(gt_images)

	return get_batches_fn

# Generate the segmented images using the original image and the mask (segments from the last layer)
def gen_test_output(sess, logits, keep_prob, image_pl, image_test, gt_test, image_shape, label_colors):

    for f in image_test:
        image_file = f
        gt_image_file = gt_test[0]

        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
        labels = sess.run([tf.argmax(tf.nn.softmax(logits), axis=-1)], {keep_prob: 1.0, image_pl: [image]})
        labels = labels[0].reshape(image_shape[0], image_shape[1])
        labels_colored = np.zeros_like(gt_image)
        for lab in label_colors:
            label_mask = labels == lab
            labels_colored[label_mask] = np.array((*label_colors[lab], 127))

        mask = scipy.misc.toimage(labels_colored, mode="RGBA")
        init_img = scipy.misc.toimage(image)
        init_img.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(init_img)

# Save the output images
def save_inference_samples(runs_dir, image_test, gt_test, sess, image_shape, logits, keep_prob, input_image, label_colors):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(sess, logits, keep_prob, input_image, image_test, gt_test, image_shape, label_colors)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num
        
def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))