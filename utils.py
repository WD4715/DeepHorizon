from skimage.io import imsave
from skimage import color
from skimage.draw import line, set_color, disk
from torchvision import transforms
import torch
from model import Model, NGDSAC, Loss
from model import NGDSAC
import cv2
import numpy as np

imagesize = 224
hypotheses = 16
inlierthreshold = 0.05
inlierbeta = 100.
inlieralpha=0.1
scorethreshold = 0.4
verbose = True
model = Model(net_capacity=4)
model.load_state_dict(torch.load("./model_weight/deep_horizon_pretrained.net"))
model.eval()
model.cuda()
ngdsac = NGDSAC(hypotheses, inlierthreshold, inlierbeta, inlieralpha, Loss(imagesize), 1)


def process_frame(image):
	'''
	Estimate horizon line for an image and return a visualization.

	image -- 3 dim numpy image tensor
	'''

	# determine image scaling factor
	image_scale = max(image.shape[0], image.shape[1])
	image_scale = imagesize / image_scale 

	# convert image to RGB
	if len(image.shape) < 3:
		image = color.gray2rgb(image)

	# store original image dimensions		
	src_h = int(image.shape[0] * image_scale)
	src_w = int(image.shape[1] * image_scale)

	# resize and to gray scale
	image = transforms.functional.to_pil_image(image)
	image = transforms.functional.resize(image, (src_h, src_w))
	image = transforms.functional.adjust_saturation(image, 0)
	image = transforms.functional.to_tensor(image)

	# make image square by zero padding
	padding_left = int((imagesize - image.size(2)) / 2)
	padding_right = imagesize - image.size(2) - padding_left
	padding_top = int((imagesize - image.size(1)) / 2)
	padding_bottom = imagesize - image.size(1) - padding_top

	padding = torch.nn.ZeroPad2d((padding_left, padding_right, padding_top, padding_bottom))
	image = padding(image)

	image_src = image.clone().unsqueeze(0)

	# normalize image (mean and variance), values estimated offline from HLW training set
	img_mask = image.sum(0) > 0
	image[:,img_mask] -= 0.45
	image[:,img_mask] /= 0.25
	image = image.unsqueeze(0).cuda()

	with torch.no_grad():
		#predict data points and neural guidance
		points, log_probs = model(image)

		# fit line with NG-DSAC, providing dummy ground truth labels
		ngdsac(points, log_probs, torch.zeros((1,2)), torch.zeros((1)), torch.ones((1)), torch.ones((1))) 

	def draw_line(data, lX1, lY1, lX2, lY2, clr):
		'''
		Draw a line with the given color and opacity.

		data -- image to draw to
		lX1 -- x value of line segment start point
		lY1 -- y value of line segment start point
		lX2 -- x value of line segment end point
		lY2 -- y value of line segment end point
		clr -- line color, triple of values
		'''

		rr, cc = line(lY1, lX1, lY2, lX2)
		set_color(data, (rr, cc), clr)

	def draw_models(labels, clr, data):
		'''
		Draw circles for a batch of images.
	
		labels -- line parameters, array shape (Nx2) where 
			N is the number of images in the batch
			2 is the number of line parameters (offset,  slope)
		data -- batch of images to draw to
		'''

		# number of image in batch
		n = labels.shape[0]

		for i in range (n):

			#line
			lY1 = int(labels[i, 0] * imagesize)
			lY2 = int(labels[i, 1] * imagesize + labels[i, 0] * imagesize)
			draw_line(data[i], 0, lY1, imagesize, lY2, clr)

		return data	

	def draw_wpoints(points, data, weights, clrmap):
		'''
		Draw 2D points for a batch of images.

		points -- 2D points, array shape (Nx2xM) where 
			N is the number of images in the batch
			2 is the number of point dimensions (x, y)
			M is the number of points
		data -- batch of images to draw to
		weights -- array shape (NxM), one weight per point, for visualization
		clrmap -- OpenCV color map for visualizing weights
			
		'''

		# create explicit color map
		color_map = np.arange(256).astype('u1')
		color_map = cv2.applyColorMap(color_map, clrmap)
		color_map = color_map[:,:,::-1] # BGR to RGB

		n = points.shape[0] # number of images
		m = points.shape[2] # number of points

		for i in range (0, n):

			s_idx = weights[i].sort(descending=False)[1] # draw low weight points first
			weights[i] = weights[i] / weights[i].max() # normalize weights for visualization

			for j in range(0, m):

				idx = int(s_idx[j])

				# convert weight to color
				clr_idx = float(min(1, weights[i,idx]))
				clr_idx = int(clr_idx * 255)
				clr = color_map[clr_idx, 0] / 255

				# draw point
				r = int(points[i, 0, idx] * imagesize)
				c = int(points[i, 1, idx] * imagesize)
				rr, cc = disk((r, c), 2)
				set_color(data[i], (rr, cc), clr)

		return data

	# normalized inlier score of the estimated line
	score = ngdsac.batch_inliers[0].sum() / points.shape[2]

	image_src = image_src.cpu().permute(0,2,3,1).numpy() #Torch to Numpy
	viz_probs = image_src.copy() * 0.2 # make a faint copy of the input image
	
	# draw estimated line
	if score > scorethreshold:
		image_src = draw_models(ngdsac.est_parameters, clr=(0,0,1), data=image_src)

	viz = [image_src]

	if verbose:	
		# create additional visualizations

		# draw faint estimated line 
		viz_score = viz_probs.copy()
		viz_probs = draw_models(ngdsac.est_parameters, clr=(0.3,0.3,0.3), data=viz_probs)
		viz_inliers = viz_probs.copy()

		# draw predicted points with neural guidance and soft inlier count, respectively
		viz_probs = draw_wpoints(points, viz_probs, weights=torch.exp(log_probs), clrmap=cv2.COLORMAP_PLASMA)
		viz_inliers = draw_wpoints(points, viz_inliers, weights=ngdsac.batch_inliers, clrmap=cv2.COLORMAP_WINTER)

		# create a explicit color map for visualize score of estimate line
		color_map = np.arange(256).astype('u1')
		color_map = cv2.applyColorMap(color_map, cv2.COLORMAP_HSV)	
		color_map = color_map[:,:,::-1]

		# map score to color
		score = int(score*100) #using only the first portion of HSV to get a nice (red, yellow, green) gradient
		clr = color_map[score, 0] / 255

		viz_score = draw_models(ngdsac.est_parameters, clr=clr, data=viz_score)

		viz = viz + [viz_probs, viz_inliers, viz_score]

	#undo zero padding of inputs
	if padding_left > 0:
		viz = [img[:,:,padding_left:,:] for img in viz]
	if padding_right > 0:
		viz = [img[:,:,:-padding_right,:] for img in viz]
	if padding_top > 0:
		viz = [img[:,padding_top:,:,:] for img in viz]
	if padding_bottom > 0:
		viz = [img[:,:-padding_bottom,:,:] for img in viz]		

	# convert to a single uchar image
	viz = np.concatenate(viz, axis=2)
	viz = viz * 255
	viz = viz.astype('u1')

	return viz[0]