import warnings
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', FutureWarning)

import numpy as np
from collections import defaultdict
import random, cPickle, sys, os, csv, json, operator, cv2, chainercv
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, average_precision_score
from PIL import Image, ImageDraw
from chainer.dataset import dataset_mixin
import chainer
import xml.etree.ElementTree as ET
import copy
import scipy.io as sio


# Colors for print in terminal ;)
c1 = "\033[40;1;31m"
c2 = "\033[40;1;34m"
c3 = "\033[40;1;32m"
c4 = "\033[40;1;36m"
c0 = "\033[0m"

class GetExample(dataset_mixin.DatasetMixin):
	def __init__(self, pairs, train=True, dataset='LIP', image_size=(299, 299)):
		self.pairs = pairs
		self.dataset = dataset
		self.train = train
		self.boxjitter = 0.1
		self.image_size = image_size		
		self.meanchannel = {
		'LIP': np.array([90.023, 96.370, 107.909], dtype=np.float32), 
		'WIDER': np.array([91.384, 97.159, 110.740], dtype=np.float32),
		'BAPD': np.array([100.506, 107.573, 114.595], dtype=np.float32)}	
		self.pca_lighting = {'eigen_value':np.array((0.2175, 0.0188, 0.0045), dtype=np.float32),
		'eigen_vector': np.array(((0.4009, -0.814,  0.4203),(0.7192, -0.0045, -0.6948),
			(-0.5675, -0.5808, -0.5836)), dtype=np.float32)}	
	def __len__(self):
		return len(self.pairs)

	def get_LIP(self, i, flip):		
		# LIP
		image_path = self.pairs[i][0]		
		# Read image (BGR)
		input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
		# Random crop
		if self.train:
			xmin, ymin, width, height = 0.0, 0.0, input_image.shape[1], input_image.shape[0]
			xmin = np.random.normal(loc=xmin, scale=self.boxjitter*width, size=1)[0]
			ymin = np.random.normal(loc=ymin, scale=self.boxjitter*height, size=1)[0]
			# Some annotations need to be handled with care
			xmin, ymin = max(xmin,0.0), max(ymin,0.0)
			if xmin >= input_image.shape[1]:
				xmin, width = 0.0, input_image.shape[1]
			if ymin >= input_image.shape[0]:
				ymin, height = 0.0, input_image.shape[0]			
			H_slice = slice(int(ymin), int(ymin + height))
			W_slice = slice(int(xmin), int(xmin + width))
			input_image = input_image[H_slice, W_slice, :]
		# Resize image
		fy = float(self.image_size[0]) / float(input_image.shape[0]) # Height
		fx = float(self.image_size[1]) / float(input_image.shape[1]) # Width
		input_image = cv2.resize(input_image, None, None, fx=fx, fy=fy,
			interpolation=cv2.INTER_LINEAR).astype(np.float32)
		# Mean-channel subtraction
		input_image -= self.meanchannel['LIP']
		# WxHx3 to 3xWxH
		input_image = np.transpose(input_image, [2,0,1])
		# Augmentation
		if flip:
			input_image = input_image[:,:,::-1]
		if self.train:
			input_image = chainercv.transforms.pca_lighting(
				input_image, 0.1, eigen_value=self.pca_lighting['eigen_value'],
				 eigen_vector=self.pca_lighting['eigen_vector'])
		# Read segmentation labels		
		label_path = self.pairs[i][1]
		# Read segmentation label (HxW)
		label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
		# Random crop
		if self.train:
			label = label[H_slice, W_slice]
		# Resize segmentation label
		fy = float(self.image_size[0]) / float(label.shape[0]) # Height
		fx = float(self.image_size[1]) / float(label.shape[1]) # Width
		# Because of labels we use INTER_NEAREST
		label = cv2.resize(label, None, None, fx=fx, fy=fy,
			interpolation=cv2.INTER_NEAREST).astype(np.int32)
		# Augmentation
		if flip:
			label = label[:,::-1]
			# Fix Left-Right labels
			label_flipped = copy.deepcopy(label)
			label_flipped[label == 14] = 15
			label_flipped[label == 15] = 14
			label_flipped[label == 16] = 17
			label_flipped[label == 17] = 16
			label_flipped[label == 18] = 19
			label_flipped[label == 19] = 18
			label = label_flipped

		return input_image, label

	def get_WIDER(self, i, flip):
		# WIDER
		image_path, annotations = self.pairs[i]		
		# Read image (BGR)
		input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
		# Crop bounding box
		xmin, ymin, width, height = annotations['bbox']
		# Random bbox jitter
		if self.train:
			xmin = np.random.normal(loc=xmin, scale=self.boxjitter*width, size=1)[0]
			ymin = np.random.normal(loc=ymin, scale=self.boxjitter*height, size=1)[0]
		# Some annotations need to be handled with care
		xmin, ymin = max(xmin,0.0), max(ymin,0.0)
		if xmin >= input_image.shape[1]:
			xmin, width = 0.0, input_image.shape[1]
		if ymin >= input_image.shape[0]:
			ymin, height = 0.0, input_image.shape[0]
		H_slice = slice(int(ymin), int(ymin + height))
		W_slice = slice(int(xmin), int(xmin + width))
		input_image = input_image[H_slice, W_slice, :]
		# Resize image
		fy = float(self.image_size[0]) / float(input_image.shape[0]) # Height
		fx = float(self.image_size[1]) / float(input_image.shape[1]) # Width
		input_image = cv2.resize(input_image, None, None, fx=fx, fy=fy,
			interpolation=cv2.INTER_LINEAR).astype(np.float32)
		# Mean-channel subtraction
		input_image -= self.meanchannel['WIDER']
		# WxHx3 to 3xWxH
		input_image = np.transpose(input_image, [2,0,1])
		# Augmentation
		if flip:
			input_image = input_image[:,:,::-1]
		if self.train:
			input_image = chainercv.transforms.pca_lighting(
				input_image, 0.1, eigen_value=self.pca_lighting['eigen_value'],
				 eigen_vector=self.pca_lighting['eigen_vector'])
		# Fix label annotations
		annotations['attribute'] = np.array(annotations['attribute'])
		label = np.zeros_like(annotations['attribute'], dtype=np.int32)
		label[annotations['attribute'] == 1] = 1
		label[annotations['attribute'] == 0] = -1
		label[annotations['attribute'] == -1] = 0
		return input_image, label

	def get_BAPD(self, i, flip):
		# Berkeley Attributes of People
		image_path, bbox, label = self.pairs[i]
		# Read image (BGR)
		input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
		# Crop bounding box		
		xmin, ymin, xmax, ymax = bbox.astype(np.float32)
		width = xmax-xmin
		height = ymax-ymin
		# Random bbox jitter
		if self.train:
			xmin = np.random.normal(loc=xmin, scale=self.boxjitter*width, size=1)[0]
			ymin = np.random.normal(loc=ymin, scale=self.boxjitter*height, size=1)[0]
		# Some annotations need to be handled with care
		xmin, ymin = max(xmin,0.0), max(ymin,0.0)
		if xmin >= input_image.shape[1]:
			xmin, width = 0.0, input_image.shape[1]
		if ymin >= input_image.shape[0]:
			ymin, height = 0.0, input_image.shape[0]			
		H_slice = slice(int(ymin), int(ymin + height))
		W_slice = slice(int(xmin), int(xmin + width))
		input_image = input_image[H_slice, W_slice, :]
		# Resize image
		fy = float(self.image_size[0]) / float(input_image.shape[0]) # Height
		fx = float(self.image_size[1]) / float(input_image.shape[1]) # Width
		input_image = cv2.resize(input_image, None, None, fx=fx, fy=fy,
			interpolation=cv2.INTER_LINEAR).astype(np.float32)
		# Mean-channel subtraction
		input_image -= self.meanchannel['BAPD']
		# WxHx3 to 3xWxH
		input_image = np.transpose(input_image, [2,0,1])
		# Augmentation
		if flip:
			input_image = input_image[:,:,::-1]
		if self.train:
			input_image = chainercv.transforms.pca_lighting(
				input_image, 0.1, eigen_value=self.pca_lighting['eigen_value'],
				 eigen_vector=self.pca_lighting['eigen_vector'])
		return input_image, label

	def get_example(self, i):
		# Augmentation
		if self.train:
		    flip = bool(random.randint(0,1))
		else:
		    flip = False
		
		if self.dataset == 'LIP':
			return self.get_LIP(i, flip)
		elif self.dataset == 'WIDER':
			return self.get_WIDER(i, flip)
		elif self.dataset == 'BAPD':
			return self.get_BAPD(i, flip)			

def PrepData(args):

	datasets = {}

	# WIDER
	dataset = {'format':'image,attribute_bbox','train':[],'test':[]}
	data_split = json.load(open('%s/WIDER/wider_attribute_trainval.json' % args.dataset_folder, 'r'))
	for i in data_split['images']:
		image_path = '%s/WIDER/Image/%s'%(args.dataset_folder, i['file_name'])
		instance = zip([image_path]*len(i['targets']), i['targets'])
		dataset['train'].extend(instance)
	data_split = json.load(open('%s/WIDER/wider_attribute_test.json' % args.dataset_folder, 'r'))
	for i in data_split['images']:
		image_path = '%s/WIDER/Image/%s'%(args.dataset_folder, i['file_name'])
		instance = zip([image_path]*len(i['targets']), i['targets'])
		dataset['test'].extend(instance)
	datasets.update({'WIDER':dataset})

	# Look into Person (LIP)
	data_split = open('%s/LIP/train_id.txt' % args.dataset_folder).readlines()
	train_list = [x.split()[-1] for x in data_split]
	data_split = open('%s/LIP/val_id.txt' % args.dataset_folder).readlines()
	validation_list = [x.split()[-1] for x in data_split]
	data_split = os.listdir('%s/LIP/images/testing_images' % args.dataset_folder)
	test_list = [x.split('.')[0] for x in data_split]
	dataset = {'format':'image,segmentation_labels','train':[],'validation':[],'test':[]}	
	for i in train_list:
		instance = ('%s/LIP/images/train_images/%s.jpg' % (args.dataset_folder,i),\
		 '%s/LIP/parsing/train_segmentations/%s.png' % (args.dataset_folder,i))
		dataset['train'].append(instance)
	for i in validation_list:
		instance = ('%s/LIP/images/val_images/%s.jpg' % (args.dataset_folder,i),\
		 '%s/LIP/parsing/val_segmentations/%s.png' % (args.dataset_folder,i))
		dataset['validation'].append(instance)
	for i in test_list:
		instance = ('%s/LIP/images/testing_images/%s.jpg' % (args.dataset_folder,i),\
		 None)
		dataset['test'].append(instance)
	datasets.update({'LIP':dataset})

	# Berkeley Attributes of People	(BAPD)
	dataset = {'format':'image,bbox,annotations','train':[],'test':[]}
	for split in ['train', 'test']:
		data_split = sio.loadmat('%s/BAPD/ground_truth/gt_attributes_%s.mat' % (args.dataset_folder, split))		
		images = [x[0][0] for x in data_split['images']]
		boxes = [x[0] for x in data_split['boxes']]
		attributes = [x[0] for x in data_split['attributes']]
		for i in range(len(images)):
			image_path = '%s/BAPD/Images/%s.jpg' % (args.dataset_folder, images[i])
			image_path = [image_path] * len(boxes[i])			
			bbox = list(boxes[i])			
			label = np.zeros_like(attributes[i], dtype=np.int32)
			label[attributes[i] == 1] = 1
			label[attributes[i] == 0] = -1
			label[attributes[i] == -1] = 0
			label = list(label)
			dataset[split].extend(zip(image_path, bbox, label))
	datasets.update({'BAPD':dataset})		

	return datasets

def ComputeMetrics(y_true, y_pred, perbatch=False): 

    estimation = np.concatenate(tuple(y_pred), axis=0).astype(np.float32)
    groundtruth = np.concatenate(tuple(y_true), axis=0).astype(np.float32)
    nb_attributes = estimation.shape[1]
    # Perbatch (which is WRONG!)
    if perbatch:
        K = estimation.shape[0]/128
        estimation = np.array_split(estimation, K)
        groundtruth = np.array_split(groundtruth, K)
    else:
        estimation = [estimation]
        groundtruth = [groundtruth]
    # Compute AP and Accuracy
    ap = np.zeros((nb_attributes,1))
    acc = np.zeros((nb_attributes,1))
    ap_counts = np.zeros((nb_attributes,1))
    acc_counts = np.zeros((nb_attributes,1))
    for ests, gts in zip(estimation, groundtruth):
        for dim in range(gts.shape[1]):
            tmp = gts[:,dim].reshape(gts.shape[0],1)
            fmt_gt = tmp[np.where(tmp!=-1)]
            fmt_gt[np.where(fmt_gt==0)] = -1
            fmt_est = ests[:,dim].reshape(ests.shape[0],1)
            fmt_est = fmt_est[np.where(tmp!=-1)]            
            # AP
            ap_score = average_precision_score(fmt_gt, fmt_est)            
            if not np.isnan(ap_score):
                ap[dim] += ap_score
                ap_counts[dim] += 1
            # Accuracy
            acc_score = accuracy_score(fmt_gt,
			 np.where(fmt_est >= 0.5, np.zeros_like(fmt_gt) + 1, np.zeros_like(fmt_gt) - 1))
			# NOTE: we can optimize threshold using validation set
            acc[dim] += acc_score
            acc_counts[dim] += 1

    return ap/ap_counts, acc/acc_counts

def Report(history, report_interval, iterk=None, T=None,  split='train'):
    if split == 'train':
    	k = -report_interval
    else:
    	k = 0
    for prefix in history.keys():
		if prefix in ['LIP','MSCOCO','PASCAL_SBD']:
			loss = np.asarray(history[prefix]['loss'][k:])
			mean_class_accuracy = np.asarray(history[prefix]['mean_class_accuracy'][k:])
			miou = np.asarray(history[prefix]['miou'][k:])
			pixel_accuracy = np.asarray(history[prefix]['pixel_accuracy'][k:])
			ks_domain = ['loss','mean_class_accuracy','miou','pixel_accuracy']
			cs_domain = [c1,c2,c3,c4]
			vs_domain = [loss, mean_class_accuracy, miou, pixel_accuracy]
		elif prefix in ['WIDER', 'BAPD']:
			loss = np.asarray(history[prefix]['loss'][k:])
			ap, accuracy = ComputeMetrics(
				history[prefix]['groundtruth'][k:],
				history[prefix]['prediction'][k:], perbatch=False)
			ks_domain = ['loss','ap','accuracy']
			cs_domain = [c1,c2,c3]
			vs_domain = [loss, ap, accuracy]


		# Report format
		report = []
		if split == 'train':
			report.append('%s %s after %.2f/%d hours/iters:' % 
				(split, prefix, T/3600.0, iterk))
		else:
			report.append('%s %s @ %dx%d:' % 
				(split, prefix, history[prefix]['image_size'][0], history[prefix]['image_size'][1]))
		
		# Loss and average precision
		for k_domain, c_domain, v_domain in zip(ks_domain,cs_domain,vs_domain):
			report.append('%s:%s%.4f%s'%(k_domain,c_domain,np.nanmean(v_domain),c0))
		print '  '.join(report)