import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class Model(nn.Module):
	'''
	Network predicting a set of points for an input image.

	'''

	def __init__(self, net_capacity):
		'''
		Constructor.

		net_capacity -- scaling factor applied to the number of channels in each layer
		'''
		super(Model, self).__init__()

		c = net_capacity

		strides = [1, 1, 2, 2, 2, 2, 2]

		self.output_dim = 2 # dimensionality of the output points

		# build network
		self.conv1 = nn.Conv2d(3, 8*c, 3, strides[0], 1)
		self.bn1 = nn.BatchNorm2d(8*c)	
		self.conv2 = nn.Conv2d(8*c, 16*c, 3, strides[1], 1)
		self.bn2 = nn.BatchNorm2d(16*c)
		self.conv3 = nn.Conv2d(16*c, 32*c, 3, strides[2], 1)
		self.bn3 = nn.BatchNorm2d(32*c)
		self.conv4 = nn.Conv2d(32*c, 64*c, 3, strides[3], 1)
		self.bn4 = nn.BatchNorm2d(64*c)

		self.conv5 = nn.Conv2d(64*c, 64*c, 3, strides[4], 1)
		self.bn5 = nn.BatchNorm2d(64*c)	
		self.conv6 = nn.Conv2d(64*c, 64*c, 3, strides[5], 1)
		self.bn6 = nn.BatchNorm2d(64*c)
		self.conv7 = nn.Conv2d(64*c, 64*c, 3, strides[6], 1)
		self.bn7 = nn.BatchNorm2d(64*c)

		self.conv8 = nn.Conv2d(64*c, 64*c, 3, 1, 1)
		self.bn8 = nn.BatchNorm2d(64*c)	
		self.conv9 = nn.Conv2d(64*c, 64*c, 3, 1, 1)
		self.bn9 = nn.BatchNorm2d(64*c)
		self.conv10 = nn.Conv2d(64*c, 64*c, 3, 1, 1)
		self.bn10 = nn.BatchNorm2d(64*c)		
				
		# output branch 1 for predicting points
		self.fc1 = nn.Conv2d(64*c, 128*c, 1, 1, 0)
		self.bn_fc1 = nn.BatchNorm2d(128*c)
		self.fc2 = nn.Conv2d(128*c, 128*c, 1, 1, 0)
		self.bn_fc2 = nn.BatchNorm2d(128*c)
		self.fc3 = nn.Conv2d(128*c, self.output_dim, 1, 1, 0)

		# output branch 2 for predicting neural guidance
		self.fc1_1 = nn.Conv2d(64*c, 128*c, 1, 1, 0)
		self.bn_fc1_1 = nn.BatchNorm2d(128*c)
		self.fc2_1 = nn.Conv2d(128*c, 128*c, 1, 1, 0)
		self.bn_fc2_1 = nn.BatchNorm2d(128*c)
		self.fc3_1 = nn.Conv2d(128*c, 1, 1, 1, 0)

	def forward(self, inputs):
		'''
		Forward pass.

		inputs -- 4D data tensor (BxCxHxW)
		'''

		batch_size = inputs.size(0)

		x = F.relu(self.bn1(self.conv1(inputs)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))

		res = x

		x = F.relu(self.bn5(self.conv5(res)))
		x = F.relu(self.bn6(self.conv6(x)))
		x = F.relu(self.bn7(self.conv7(x)))

		res = x

		x = F.relu(self.bn8(self.conv8(res)))
		x = F.relu(self.bn9(self.conv9(x)))
		x = F.relu(self.bn10(self.conv10(x)))

		res = res + x
		
		# === output branch 1, predict 2D points ====================
		x1 = F.relu(self.bn_fc1(self.fc1(res)))
		x1 = F.relu(self.bn_fc2(self.fc2(x1)))
		points = self.fc3(x1)
		points = torch.sigmoid(points) # normalize to 0,1

		# map local (patch-centric) point predictions to global image coordinates
		# i.e. distribute the points over the image
		patch_offset = 1 / points.size(2)
		patch_size = 3

		points = points * patch_size - patch_size / 2 + patch_offset / 2

		for col in range(0, points.size(3)):
			points[:,1,:,col] = points[:,1,:,col] + col * patch_offset
			
		for row in range(0, points.size(2)):
			points[:,0,row,:] = points[:,0,row,:] + row * patch_offset

		points = points.view(batch_size, 2, -1)
		
		# === output branch 2, predict neural guidance ============== 
		x2 = F.relu(self.bn_fc1_1(self.fc1_1(res.detach())))
		x2 = F.relu(self.bn_fc2_1(self.fc2_1(x2)))
		log_probs = self.fc3_1(x2)
		log_probs = log_probs.view(batch_size, -1)
		log_probs = F.logsigmoid(log_probs) # normalize output to 0,1

		# normalize probs to sum to 1
		normalizer = torch.logsumexp(log_probs, dim=1)
		normalizer = normalizer.unsqueeze(1).expand(-1, log_probs.size(1))
		norm_log_probs = log_probs - normalizer

		return points, norm_log_probs

class NGDSAC:
	'''
	Neural-Guided DSAC to robustly fit lines.
	'''

	def __init__(self, hyps, inlier_thresh, inlier_beta, inlier_alpha, loss_function, invalid_loss):
		'''
		Constructor.

		hyps -- number of line hypotheses sampled for each image
		inlier_thresh -- threshold used in the soft inlier count, its measured in relative image size
		inlier_beta -- scaling factor within the sigmoid of the soft inlier count
		inlier_alpha -- scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)
		loss_function -- function to compute the quality of estimated line parameters wrt ground truth
		invalid_loss -- punishment when sampling invalid hypothesis
		'''

		self.hyps = hyps
		self.inlier_thresh = inlier_thresh
		self.inlier_beta = inlier_beta
		self.inlier_alpha = inlier_alpha
		self.loss_function = loss_function
		self.invalid_loss = invalid_loss

	def __sample_hyp(self, x, y, p, pool):
		'''
		Calculate a line hypothesis (slope, intercept) from two random points.

		x -- vector of x values
		y -- vector of y values
		p -- sampling probabilities for selecting points
		pool -- indicator vector updated with which points have been selected
		'''

		# select points
		idx = torch.multinomial(p, 2, replacement = True)
		idx1 = int(idx[0])
		idx2 = int(idx[1])

		# set indicators which points have been selected
		pool[idx1] += 1
		pool[idx2] += 1
	
		# validity check, do not choose too close together
		if torch.abs(x[idx1] - x[idx2]) < 0.05:
			return 0, 0, False # no valid hypothesis found, indicated by False

		# calculate line parameters
		slope = (y[idx1] - y[idx2]) / (x[idx1] - x[idx2])
		intercept = y[idx1] - slope * x[idx1]

		return slope, intercept, True # True indicates a valid hypothesos

		

	def __soft_inlier_count(self, slope, intercept, x, y):
		'''
		Soft inlier count for a given line and a given set of points.

		slope -- slope of the line
		intercept -- intercept of the line
		x -- vector of x values
		y -- vector of y values
		'''

		# point line distances
		dists = torch.abs(slope * x - y + intercept)
		dists = dists / torch.sqrt(slope * slope + 1)

		# soft inliers
		dists = 1 - torch.sigmoid(self.inlier_beta * (dists - self.inlier_thresh)) 
		score = torch.sum(dists)

		return score, dists	

	def __call__(self, prediction, log_probs, labels,  xStart, xEnd, imh):
		'''
		Perform robust, differentiable line fitting according to NG-DSAC.

		Returns the expected loss and hypothesis distribution entropy.
		Expected loss can be used for backprob, entropy for monitoring / debugging.

		prediction -- predicted 2D points for a batch of images, array of shape (Bx2xN) where
			B is the number of images in the batch
			2 is the number of point dimensions (y, x)
			N is the number of predicted points
		log_probs -- log of selection probabilities, array of shape (BxN)
		labels -- ground truth labels for the batch, array of shape (Bx2) where
			2 is the number of parameters (intercept, slope)
		xStart -- x-values where each ground truth line starts (for calculating the loss), array of shape (B)
		xEnd -- x-values where each ground truth line ends (for calculating the loss), array of shape (B)
		imh -- relative height of the image (for calculating the loss), <= 1, array of shape (B)
		'''

		# faster on CPU because of many, small matrices
		prediction = prediction.cpu()
		batch_size = prediction.size(0)

		avg_exp_loss = 0 # expected loss
		avg_entropy = 0 # hypothesis distribution entropy 

		self.est_parameters = torch.zeros(batch_size, 2) # estimated lines (w/ max inliers)
		self.batch_inliers = torch.zeros(batch_size, prediction.size(2)) # (soft) inliers for estimated lines
		self.g_log_probs = torch.zeros(batch_size, prediction.size(2)) # gradient tensor for neural guidance

		for b in range(0, batch_size):

			hyp_losses = torch.zeros([self.hyps, 1]) # loss of each hypothesis
			hyp_scores = torch.zeros([self.hyps, 1]) # score of each hypothesis

			max_score = 0 	# score of best hypothesis

			y = prediction[b, 0] # all y-values of the prediction
			x = prediction[b, 1] # all x.values of the prediction

			p = torch.exp(log_probs[b]) # selection probabilities for points

			for h in range(0, self.hyps):	

				# === step 1: sample hypothesis ===========================
				slope, intercept, valid = self.__sample_hyp(x, y, p, self.g_log_probs[b])
				if not valid: 
					hyp_losses[h] = self.invalid_loss
					hyp_scores[h] = 0.0001
					continue # skip other steps for invalid hyps

				# === step 2: score hypothesis using soft inlier count ====
				score, inliers = self.__soft_inlier_count(slope, intercept, x, y)

				hyp = torch.zeros([2])
				hyp[1] = slope
				hyp[0] = intercept

				# === step 3: calculate loss of hypothesis ================
				loss = self.loss_function(hyp, labels[b],  xStart[b], xEnd[b], imh[b]) 

				# store results
				hyp_losses[h] = loss
				hyp_scores[h] = score

				# keep track of best hypothesis so far
				if score > max_score:
					max_score = score
					self.est_parameters[b] = hyp.detach()
					self.batch_inliers[b] = inliers.detach()

			# === step 4: calculate the expectation ===========================

			#softmax distribution from hypotheses scores			
			hyp_scores = F.softmax(self.inlier_alpha * hyp_scores, 0)

			# expectation of loss
			avg_exp_loss += torch.sum(hyp_losses * hyp_scores)

		return avg_exp_loss / batch_size
	
class Loss:
	'''
	Compares two lines by calculating the distance between their ends in the image.
	'''

	def __init__(self, image_size, cut_off = 0.25):
		'''
		Constructor.

		image_size -- size of the input images, used to normalize the loss
		cut_off -- soft clamping of loss after this value
		'''
		self.image_size = image_size
		self.cut_off = cut_off
	
	def __get_max_points(self, slope, intercept, xStart, xEnd):
		'''
		Calculates the 2D points where a line intersects with the image borders.

		slope -- slope of the line
		intercept -- intercept of the line
		'''
		pts = torch.zeros([2, 2])

		x0 = float(xStart)
		x1 = float(xEnd)
		y0 = intercept + x0 * slope
		y1 = intercept + x1 * slope
		
		pts[0, 0] = x0
		pts[0, 1] = y0
		pts[1, 0] = x1
		pts[1, 1] = y1

		return pts

	def __call__(self, est, gt,  xStart, xEnd, imh):
		'''
		Calculate the line loss.

		est -- estimated line, form: [intercept, slope]
		gt -- ground truth line, form: [intercept, slope]
		'''

		pts_est = self.__get_max_points(est[1], est[0], xStart, xEnd,)
		pts_gt = self.__get_max_points(gt[1], gt[0], xStart, xEnd,)

		# not clear which ends of the lines should be compared (there are ambigious cases), compute both and take min
		loss = pts_est - pts_gt
		loss = loss.norm(2, 1).max()

		loss = loss * self.image_size / float(imh)

		# soft clamping
		if loss < self.cut_off:
			return loss
		else:
			return torch.sqrt(self.cut_off * loss)