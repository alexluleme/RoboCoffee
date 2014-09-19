#!/usr/bin/env python
import sys
#import freenect
import cv2
import numpy as np
import visual_frame_convert
import time
import os
import glob
import shutil
import mlpy
import scipy.spatial.distance as dist


HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"


def getRGBHistograms(RGBimage):
	# compute histograms: 
	[histR, bin_edges] = np.histogram(RGBimage[:,:,0], bins=(range(-1,256, 16)))
	[histG, bin_edges] = np.histogram(RGBimage[:,:,1], bins=(range(-1,256, 16)))
	[histB, bin_edges] = np.histogram(RGBimage[:,:,2], bins=(range(-1,256, 16)))
	# normalize histograms:
	histR = histR.astype(float); histR = histR / np.sum(histR);
	histG = histG.astype(float); histG = histG / np.sum(histG);
	histB = histB.astype(float); histB = histB / np.sum(histB);
	return (histR, histG, histB)


def skin_detection(rgb):
	rgb = rgb.astype('float')
	A = rgb[:,:,0] + rgb[:,:,1] + rgb[:,:,2]
	normalizedR = rgb[:,:,0] / (rgb[:,:,0] + rgb[:,:,1] + rgb[:,:,2])
	normalizedG = rgb[:,:,1] / (rgb[:,:,0] + rgb[:,:,1] + rgb[:,:,2])
	normalizedB = rgb[:,:,2] / (rgb[:,:,0] + rgb[:,:,1] + rgb[:,:,2])

	#print "SKIN:", np.mean(normalizedR), np.mean(normalizedG), np.mean(normalizedB)
	return np.count_nonzero((normalizedR>=0.35) & (normalizedR <=0.55) & (normalizedG >=0.28) & (normalizedG <=0.35)) / float(rgb.shape[0]*rgb.shape[1])
	

def intersect_rectangles(r1, r2):
	x11 = r1[0]; y11 = r1[1]; x12 = r1[0]+r1[2]; y12 = r1[1]+r1[3];
	x21 = r2[0]; y21 = r2[1]; x22 = r2[0]+r2[2]; y22 = r2[1]+r2[3];
		
	X1 = max(x11, x21); X2 = min(x12, x22);
	Y1 = max(y11, y21); Y2 = min(y12, y22);

	W = X2 - X1
	H = Y2 - Y1
	if (H>0) and (W>0):
		E = W * H;
	else:
		E = 0.0;
	Eratio = 2.0*E / (r1[2]*r1[3] + r2[2]*r2[3])
	return Eratio


def resizeFrame(frame, targetWidth):	
	(Width, Height) = frame.shape[1], frame.shape[0]

	if targetWidth > 0: 							# Use FrameWidth = 0 for NO frame resizing
		ratio = float(Width) / targetWidth		
		newHeight = int(round(float(Height) / ratio))
		frameFinal = cv2.resize(frame, (targetWidth, newHeight))
	else:
		frameFinal = frame;

	return frameFinal




def getFaceFeatures(rgb, cascadeFrontal, cascadeProfile, storage, newWidth, minWidthRange, onlyFeatures):
	if not onlyFeatures:
		facesFrontal = []; facesProfile = []
		image = cv2.cv.fromarray(rgb)
		detectedFrontal = cv2.cv.HaarDetectObjects(image, cascadeFrontal, storage, 1.3, 2, cv2.cv.CV_HAAR_DO_CANNY_PRUNING, (newWidth/minWidthRange, newWidth/minWidthRange))

		for (x,y,w,h),n in detectedFrontal:
			facesFrontal.append((x,y,w,h))

		# remove overlaps:
		while (1):
			Found = False
			for i in range(len(facesFrontal)):
				for j in range(len(facesFrontal)):
					if i != j:
						interRatio = intersect_rectangles(facesFrontal[i], facesFrontal[j])
						if interRatio>0.3:
							Found = True;
							del facesFrontal[i]
							break;
				if Found:
					break;
			if not Found:	# not a single overlap has been detected -> exit loop
				break;

		# remove non-skin
		countFaces = 0; facesFinal = []
		for (x,y,w,h) in facesFrontal:
			countFaces+=1
			#print h, y+2*h, rgb.shape[0]
			if y+int(h*2) < rgb.shape[0]:
				y = y+int(h/3)				
				h = int(2*h)
			else:
				h = rgb.shape[0] - y - 1
				
			curFace = rgb[y:y+h, x+int(2*w/10):x+int(8*w/10), :]
			skinPercent = skin_detection(curFace) 
			if skinPercent > 0:
				curFace = resizeFrame(curFace, 100)
				curFace = curFace.astype(float);
				"""				
				Rnorm = curFace[:,:,0]/(curFace[:,:,1]+curFace[:,:,2])		
				Gnorm = curFace[:,:,1]/(curFace[:,:,0]+curFace[:,:,2])	
				Bnorm = curFace[:,:,2]/(curFace[:,:,1]+curFace[:,:,0])
				b1 = np.sort(Rnorm,axis=None)
				b2 = np.sort(Gnorm,axis=None)
				b3 = np.sort(Bnorm,axis=None)
				Add1 = []
				Add2 = []
				Add3 = []
				for xx in range(0,9):
					Add1.append(b1[-xx-1])
					Add2.append(b2[-xx-1])
					Add3.append(b3[-xx-1])
				"""
				maxR = curFace[:,:,0].max()
				maxG = curFace[:,:,1].max()
				maxB = curFace[:,:,2].max()
				Average =  np.mean([maxR,maxG,maxB])
				curFace[:,:,0] = 255*curFace[:,:,0] / Average
				curFace[:,:,1] = 255*curFace[:,:,1] / Average
				curFace[:,:,2] = 255*curFace[:,:,2] / Average
				[histR, histG, histB] = getRGBHistograms(curFace)
				Features = np.concatenate([histR, histG, histB])

				facesFinal.append((x,y,w,h,Features))
	else:
		curFace = np.copy(rgb)
		curFace = resizeFrame(curFace, 100)
		curFace = curFace.astype(float);
		maxR = curFace[:,:,0].max()
		maxG = curFace[:,:,1].max()
		maxB = curFace[:,:,2].max()
		Average =  np.mean([maxR,maxG,maxB])
		curFace[:,:,0] = 255*curFace[:,:,0] / Average
		curFace[:,:,1] = 255*curFace[:,:,1] / Average
		curFace[:,:,2] = 255*curFace[:,:,2] / Average
		[histR, histG, histB] = getRGBHistograms(curFace)
		Features = np.concatenate([histR, histG, histB])
		facesFinal = [(0,0,rgb.shape[1],rgb.shape[0],Features)]
	return (facesFinal)

def analyzeImageFolder(imagePath, modelName):	# for training the model
	HAAR_CASCADE_PATH_FRONTAL = "haarcascade_frontalface_default.xml"
	HAAR_CASCADE_PATH_PROFILE = "haarcascade_frontalface_default.xml"
	cascadeFrontal = cv2.cv.Load(HAAR_CASCADE_PATH_FRONTAL);
	cascadeProfile = cv2.cv.Load(HAAR_CASCADE_PATH_PROFILE);
	storage = cv2.cv.CreateMemStorage()

	D = glob.glob(imagePath + os.sep + "*.jpg")
	count = 0;
	for i, d in enumerate(D):		
		img = cv2.imread(d, cv2.CV_LOAD_IMAGE_COLOR)
		#print d
		rgb = cv2.cvtColor(img, cv2.cv.CV_BGR2RGB)	
		faces = getFaceFeatures(rgb, cascadeFrontal, cascadeProfile, storage, rgb.shape[1], 10, False)
		if len(faces)==1:
			count += 1
			#print count
			(x,y,w,h,F) = faces[0]
			#print x,y,w,h
			curFace = img[y:y+h, x:x+w, :]
			#cv2.imshow('image',curFace); 
			#cv2.waitKey(2)
			
			F = F.reshape(F.shape[0], 1)
			if count==1:
				FeatureMatrix = F;
			else:
				FeatureMatrix = np.concatenate((FeatureMatrix, F), axis=1)


	#pca = mlpy.PCA(method='cov') # pca (eigenvalue decomposition)
	#pca.learn(FeatureMatrix.T)
	#coeff = pca.coeff()
	#y_eig = pca.transform(FeatureMatrix.T, k=2)
	np.save(modelName, FeatureMatrix)

def analyzeImageFolderTest(imagePath, modelName, onlyFeatures):	# for extracting features and testing given a trained model
	HAAR_CASCADE_PATH_FRONTAL = "haarcascade_frontalface_default.xml"
	HAAR_CASCADE_PATH_PROFILE = "haarcascade_frontalface_default.xml"
	cascadeFrontal = cv2.cv.Load(HAAR_CASCADE_PATH_FRONTAL);
	cascadeProfile = cv2.cv.Load(HAAR_CASCADE_PATH_PROFILE);
	storage = cv2.cv.CreateMemStorage()
	FeatureMatrix = np.load(modelName+".npy")
	#print FeatureMatrix.T

	D = glob.glob(imagePath + os.sep + "*.jpg")
	count = 0;
	distances = []
	for i, d in enumerate(D):		
		img = cv2.imread(d, cv2.CV_LOAD_IMAGE_COLOR)
		rgb = cv2.cvtColor(img, cv2.cv.CV_BGR2RGB)	
		faces = getFaceFeatures(rgb, cascadeFrontal, cascadeProfile, storage, rgb.shape[1], 10, onlyFeatures)
		if len(faces)==1:
			count += 1
			(x,y,w,h,F) = faces[0]
			# Here calculates the similarity of each face
			Dists = dist.cdist(FeatureMatrix.T, F.reshape(F.shape[0], 1).T,'correlation')
			distances.append(Dists.mean())
		else:
			distances.append(np.inf)
	return distances


def fcount(path):

	count1 = 0
	for root, dirs, files in os.walk(path):
		count1 +=len(dirs)
	return count1


def evaluateScript(modelPaths, onlyFeatures):


	number_folders = fcount("Persons")

	CM = np.zeros((len(modelPaths), number_folders))
	
	for m,mname in enumerate(modelPaths):
		# m = the number of model, mname = model's name (i.e. Andrew)
		Distances = []
		Lengths = []	



	for i in range(number_folders):
		dirName = "Persons/Person_" + str(i)
		#print dirName
		Temp = analyzeImageFolderTest(dirName, mname, onlyFeatures)
		Distances.append(analyzeImageFolderTest(dirName, mname, onlyFeatures))
		Lengths.append(len(Temp))
		
	maxLength = max(Lengths)
	for i in range(maxLength):	# for each file
		idx = [k for k in range(len(Lengths)) if Lengths[k]>i]
		curDistances = [Distances[k][i] for k in idx]
		curWinner = idx[np.argmin(curDistances)]			
		CM[m][curWinner] += len(idx)
		
	FinalAnswer = np.argmax(CM,axis = 1)
	Confidence = max(max(CM)) / np.sum(CM)
	print CM,Confidence,FinalAnswer
	lala = "Persons/Person_" + str(FinalAnswer);
	lolo = glob.glob(lala + os.sep + "*.jpg")
	img = cv2.imread(lolo[0], cv2.CV_LOAD_IMAGE_COLOR)
	cv2.imshow('image',img)
	cv2.imwrite('winnerFace.jpg',img)
	#cv2.waitKey(10000)
	
	return FinalAnswer



def main(argv):
	counter = 0
	pathInput = "transfered_images"
	Model = "Model"
	analyzeImageFolder(pathInput, Model)

	FinalAnswer = evaluateScript([Model], True)
	RightAnswer = 1
	if FinalAnswer == RightAnswer:
		counter+=1


			
			

if __name__ == "__main__":
	main(sys.argv)
