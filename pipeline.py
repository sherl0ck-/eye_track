
from __future__ import print_function
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import time
import math
import sys
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import itertools

def cantorEncode(x1, x2):
	return 0.5 * (x1+x2) * (x1+x2+1) + x2

def cantorDecode(z):
	w = math.floor(0.5 * (math.sqrt(8*z+1)-1))
	t = (w*w+w)/2.0
	y = z - t
	x = w - y
	return (x,y)

#####################
####### Machine Learning Scripts
#####################

# Cross validation function for single classifier
def crossValidationSVR(classifier, X, y, k=10, verbose=0):
	# split data set into different sizes, and train the model
	size = len(X) / k
	epsilon = 27
	accuracies = []
	for j in range(k):
		kXtest = np.array(X[j * size : (j + 1) * size])
		kytest = np.array(y[j * size : (j + 1) * size])
		if j == 0:
			kXtrain = np.array(X[(j + 1) * size : len(X)])
			kytrain = np.array(y[(j + 1) * size : len(y)])
		elif j + 1 == k:
			kXtrain = np.array(X[0 : j * size])
			kytrain = np.array(y[0 : j * size])
		else:
			kXtrain = np.append(np.array(X[0 : j * size]), np.array(X[(j + 1) * size : len(X)]), axis = 0)
			kytrain = np.append(np.array(y[0 : j * size]), np.array(y[(j + 1) * size : len(y)]), axis = 0)

		# get data and fit it
		kytrain = [cantorEncode(sam[0], sam[1]) for sam in kytrain]
		dataFit = classifier.fit(kXtrain, kytrain)

		# test the predictions
		# and calculate the accuracy
		correct = 0
		prediction = dataFit.predict(kXtest)
		runningLength = 0
		for m in range(len(kytest)):
			yval = kytest[m]
			if (prediction[m] <= 0):
				continue

			runningLength += 1
			xval = cantorDecode(prediction[m])
			loss = max(0, np.abs(math.sqrt((yval[0] - xval[0])*(yval[0] - xval[0]) 
								+ (yval[1] - xval[1])*(yval[1] - xval[1]))) 
								- epsilon)
			# print(yval[0], yval[1], " - ", xval[0], str(xval[1]) + ":", loss)
			if loss == 0:
				correct += 1

		accuracy = float(correct)/runningLength
		accuracies.append(accuracy)
		if verbose > 0:
			print("k-fold (" + str(j) + "):",  str(accuracy), "(True: " + str(correct) + ", false: " + str(len(kytest) - correct) + ")")

	return sum(accuracies) / len(accuracies)

# Score for single classifier SVR
def scoreSVR(classifier, X, y, Xtest, ytest):
	epsilon = 27
	y = [cantorEncode(sam[0], sam[1]) for sam in y]
	dataFit = classifier.fit(X, y)
	correct = 0
	prediction = dataFit.predict(Xtest)
	runningLength = 0
	totalLoss = 0
	for m in range(len(ytest)):
		yval = ytest[m]
		if (prediction[m] <= 0):
			continue
		
		runningLength += 1
		xval = cantorDecode(prediction[m])
		loss = max(0, np.abs(math.sqrt((yval[0] - xval[0])*(yval[0] - xval[0]) 
							+ (yval[1] - xval[1])*(yval[1] - xval[1]))) 
							- epsilon)

		totalLoss += loss
		# print(yval[0], yval[1], " - ", xval[0], str(xval[1]) + ":", loss)
		if loss == 0:
			correct += 1
	return float(correct)/runningLength, float(totalLoss)/runningLength


# Cross validation function for single classifier
def crossValidationSVRmulti(classifierX, classifierY, X, y, k=10, verbose=0):
	# split data set into different sizes, and train the model
	size = len(X) / k
	epsilon = 27
	accuracies = []
	for j in range(k):
		kXtest = np.array(X[j * size : (j + 1) * size])
		kytest = np.array(y[j * size : (j + 1) * size])
		if j == 0:
			kXtrain = np.array(X[(j + 1) * size : len(X)])
			kytrain = np.array(y[(j + 1) * size : len(y)])
		elif j + 1 == k:
			kXtrain = np.array(X[0 : j * size])
			kytrain = np.array(y[0 : j * size])
		else:
			kXtrain = np.append(np.array(X[0 : j * size]), np.array(X[(j + 1) * size : len(X)]), axis = 0)
			kytrain = np.append(np.array(y[0 : j * size]), np.array(y[(j + 1) * size : len(y)]), axis = 0)


		# get data and fit it
		kytrainX = [sam[0] for sam in kytrain]
		kytrainY = [sam[1] for sam in kytrain]
		
		dataFitX = classifierX.fit(kXtrain, kytrainX)
		dataFitY = classifierY.fit(kXtrain, kytrainY)
		# test the predictions
		# and calculate the accuracy
		correct = 0
		predictionX = dataFitX.predict(kXtest)
		predictionY = dataFitY.predict(kXtest)
		for m in range(len(kytest)):
			yval = kytest[m]

			predX = predictionX[m]
			predY = predictionY[m]
			loss = max(0, np.abs(math.sqrt((yval[0] - predX)*(yval[0] - predX) 
								+ (yval[1] - predY)*(yval[1] - predY))) 
								- epsilon)
			# print(yval[0], yval[1], " - ", xval[0], str(xval[1]) + ":", loss)
			if loss == 0:
				correct += 1

		accuracy = float(correct)/len(kytest)
		accuracies.append(accuracy)
		if verbose > 0:
			print("k-fold (" + str(j) + "):",  str(accuracy), "(True: " + str(correct) + ", false: " + str(len(kytest) - correct) + ")")

	return sum(accuracies) / len(accuracies)


# Score for multi classifier SVR
def scoreSVRmulti(classifierX, classifierY, X, y, Xtest, ytest):
	epsilon = 27
	yX = [sam[0] for sam in y]
	yY = [sam[1] for sam in y]
	dataFitX = classifierX.fit(X, yX)
	dataFitY = classifierY.fit(X, yY)
	correct = 0
	predictionX = dataFitX.predict(Xtest)
	predictionY = dataFitY.predict(Xtest)
	loss = 0
	totalLoss = 0
	for m in range(len(ytest)):
		yval = ytest[m]
		
		predX = predictionX[m]
		predY = predictionY[m]
		loss = max(0, np.abs(math.sqrt((yval[0] - predX)*(yval[0] - predX) 
							+ (yval[1] - predY)*(yval[1] - predY))) 
							- epsilon)
		totalLoss += loss
		#print(yval[0], yval[1], " - ", xval[0], str(xval[1]) + ":", loss)
		if loss == 0:
			correct += 1
	return float(correct)/len(ytest), float(totalLoss)/len(ytest)

# Script for running Lasso
def scoreLasso(X, y, Xtest, ytest):
	y = [cantorEncode(sam[0], sam[1]) for sam in y]

	# Run cross validation
	lassoCV = linear_model.LassoCV(normalize=True, cv=10, max_iter=1000000).fit(X, y)
	lalpha = lassoCV.alpha_
	lassoBest = linear_model.Lasso(alpha=lalpha, normalize=True).fit(X, y)
	# Predict
	predY = lassoBest.predict(Xtest)
	correct = np.zeros(10)
	losses = np.zeros(10)
	for t in range(len(predY)):
		predYDecoded = cantorDecode(max(int(predY[t]), 0))
		# print("Predicted: ", predYDecoded, "True: ", ytest[t], "Difference: ", int(math.sqrt((ytest[t][0] - predYDecoded[0])**2 + (ytest[t][1] - predYDecoded[1])**2)))
		for k in range (1,11):
			loss = max(math.sqrt((ytest[t][0] - predYDecoded[0])**2 + (ytest[t][1] - predYDecoded[1])**2)-30*k, 0)
			losses[k-1] += loss
			if (loss==0):
				correct[k-1] +=1 


	testScore = (correct/len(Xtest))
	losses = (losses/len(Xtest))
	return testScore, losses

# Script for running Lasso ensemble
def scoreLassoMulti(X, y, Xtest, ytest):
	yX = [sam[0] for sam in y]
	yY = [sam[1] for sam in y]
	# Run cross validation
	lassoCVX = linear_model.LassoCV(normalize=True, cv=10, max_iter=100000).fit(X, yX)
	lassoCVY = linear_model.LassoCV(normalize=True, cv=10, max_iter=100000).fit(X, yY)
	lalphaX = lassoCVX.alpha_
	lalphaY = lassoCVY.alpha_
	lassoBestX = linear_model.Lasso(alpha=lalphaX, normalize=True).fit(X, yX)
	lassoBestY = linear_model.Lasso(alpha=lalphaY, normalize=True).fit(X, yY)

	# Predict
	predYX = lassoBestX.predict(Xtest)
	predYY = lassoBestY.predict(Xtest)
	correct = np.zeros(10)
	losses = np.zeros(10)
	for t in range(len(predYX)):
		predYDecoded = (predYX[t], predYY[t])
		#print("Predicted: ", predYDecoded, "True: ", ytest[t], "Difference: ", int(math.sqrt((ytest[t][0] - predYDecoded[0])**2 + (ytest[t][1] - predYDecoded[1])**2)))
		for k in range (1,11):
			loss = max(math.sqrt((ytest[t][0] - predYDecoded[0])**2 + (ytest[t][1] - predYDecoded[1])**2)-30*k, 0)
			losses[k-1] += loss
			if (loss==0):
				correct[k-1] +=1 

	testScore = (correct/len(Xtest))
	losses = (losses/len(Xtest))
	return testScore, losses


# call with 
dataPath = sys.argv[1]
calibrated = int(sys.argv[2])
combinations = int(sys.argv[3])
if len(sys.argv) < 4:
	print("Add more parameters")
	exit()

# features = [("pupil", [0,4]), ("haar", [4,12]), ("saliency", [12,2112])]
features = [("pupil", [0, 4]), ("haar", [4, 12]), ("saliency", [12, 522])]
X = []
Xtest = []
y = []
ytest = []

if dataPath:
	dataFile = open(dataPath, 'r')
else:
	dataFile = open("data-final-haar50000.txt", 'r')

dataRead = dataFile.read()
lines = dataRead.splitlines()
# np.random.shuffle(lines);

# person-dependent
if calibrated:
	print("################ Person Calibrated: ################")
	counter = 0
	for line in lines:
		splitty = line.split()
		sz = len(splitty)
		if (int(splitty[sz-2])>0 and int(splitty[sz-1]) > 0):
			sample = splitty[:sz-2]

			# test set
			if (counter % 4 == 0):
				Xtest.append(sample)
				ytest.append([int(splitty[sz-2]) + 20, int(splitty[sz-1]) + 20])
			
			#training set
			else: 
				X.append(sample)
				y.append([int(splitty[sz-2]) + 20, int(splitty[sz-1]) + 20])
			counter = counter+1

#person-independent
else:
	print("################ Uncalibrated: ################")
	counter = 0
	getReady = False
	for line in lines:
		splitty = line.split()
		sz = len(splitty)
		if (int(splitty[sz-1]) == -9 and counter>=len(lines)*0.7):
			getReady = True;

		if int(splitty[sz-2])>0 and int(splitty[sz-1]) > 0:
			sample = splitty[:sz-2]

			if (counter < len(lines)*0.7 or not getReady):
				X.append(sample)
				y.append([int(splitty[sz-2]) + 20, int(splitty[sz-1]) + 20])

			elif (counter > len(lines)*0.7 and getReady):
				Xtest.append(sample)
				ytest.append([int(splitty[sz-2]) + 20, int(splitty[sz-1]) + 20])
		
		counter = counter + 1 

scaler = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(X)
Xscaled = scaler.transform(X)
XtestScaled = scaler.transform(Xtest)
futureSwitches = map(list, itertools.product([0, 1], repeat=len(features)))[1:]

# Run for cantor encoded classifier
if combinations:
	for combinations in futureSwitches:
		combX = np.empty([len(Xscaled),0])
		combXtest = np.empty([len(XtestScaled), 0])

		print("Running ", end="")
		for i in range(len(combinations)):
			if combinations[i]:
				combX = np.concatenate((combX, Xscaled[:, features[i][1][0]:features[i][1][1]]), axis = 1)
				combXtest = np.concatenate((combXtest, XtestScaled[:, features[i][1][0]:features[i][1][1]]), axis = 1)
				print(features[i][0] + " ", end="")
		print(combX.shape, combXtest.shape)

		# Lasso runs
		lassoScore, lassoLoss = scoreLasso(combX, y, combXtest, ytest)
		print("[LassoCantor] Score:", lassoScore)
		print("[LassoCantor] Average loss:", lassoLoss)
		lassoScoreMulti, lassoLossMulti = scoreLassoMulti(combX, y, combXtest, ytest)
		print("[LassoCantorEnsemble] Score:", lassoScoreMulti)
		print("[LassoCantorEnsemble] Average Loss:", lassoLossMulti)

		# SVR runs
		classifier = SVR(kernel='rbf', C=2100000)
		svrCvScore = crossValidationSVR(classifier, combX, y, verbose=0)
		svrScore, svrLoss = scoreSVR(classifier, combX, y, combXtest, ytest)
		print("[SVRCantor] Cross Validation:", svrCvScore, "Score:", svrScore, "Loss:", svrLoss)


		classifierX = SVR(kernel='rbf', C=1700)
		classifierY = SVR(kernel='rbf', C=650)
		svrCvScore = crossValidationSVRmulti(classifierX, classifierY, combX, y, verbose=0)
		svrScore, svrLoss = scoreSVRmulti(classifierX, classifierY, combX, y, combXtest, ytest)
		print("[SVREnsemble] Cross Validation:", svrCvScore, "Score:", svrScore, "Loss:", svrLoss)
		print("--------------------")
else:
	combX = Xscaled
	combXtest = XtestScaled

	print("Running all")
	print(combX.shape, combXtest.shape)

	# Lasso runs
	lassoScore, lassoLoss = scoreLasso(combX, y, combXtest, ytest)
	print("[LassoCantor] Score:", lassoScore)
	print("[LassoCantor] Average loss:", lassoLoss)
	lassoScoreMulti, lassoLossMulti = scoreLassoMulti(combX, y, combXtest, ytest)
	print("[LassoCantorEnsemble] Score:", lassoScoreMulti)
	print("[LassoCantorEnsemble] Average Loss:", lassoLossMulti)

	# SVR runs
	classifier = SVR(kernel='rbf', C=2100000)
	svrCvScore = crossValidationSVR(classifier, combX, y, verbose=0)
	svrScore, svrLoss = scoreSVR(classifier, combX, y, combXtest, ytest)
	print("[SVRCantor] Cross Validation:", svrCvScore, "Score:", svrScore, "Loss:", svrLoss)


	classifierX = SVR(kernel='rbf', C=1700)
	classifierY = SVR(kernel='rbf', C=650)
	svrCvScore = crossValidationSVRmulti(classifierX, classifierY, combX, y, verbose=0)
	svrScore, svrLoss = scoreSVRmulti(classifierX, classifierY, combX, y, combXtest, ytest)
	print("[SVREnsemble] Cross Validation:", svrCvScore, "Score:", svrScore, "Loss:", svrLoss)
	print("--------------------")


#person-independent
# counter = 0
# getReady = False
# for line in lines:
# 	splitty = line.split()
# 	if (int(splitty[sz-1]) == -9):
# 		getReady = True;

# 	if int(splitty[sz-2])>0 and int(splitty[sz-1]) > 0):
# 		sample = splitty[:sz-2]

# 		if (counter < 30000 and !getReady):
# 			X.append(sample)
# 			y.append([int(splitty[sz-2]) + 20, int(splitty[sz-1]) + 20])

# 		else if (counter > 30000 and getReady):
# 			Xtest.append(sample)
# 			ytest.append([int(splitty[sz-2]) + 20, int(splitty[sz-1]) + 20])
#	getReady=False
# Normalize



# ---------------
# --- Encoded Run
# ---------------
# max_cv = -100
# maxC = 10000
# bestCEncoded = 1
# bestCVEncoded = 0
# for i in range(1,100000,1000): #range(1000000, 1139040, 25000):
# 	start = time.time()

# 	# Use 0-epsilon to customize it later
# 	classifier = SVR(kernel='rbf', C=i, epsilon=0)

# 	# split data set into different sizes, and train the model
# 	k = 10
# 	size = len(Xscaled) / k
# 	epsilon = 20
# 	accuracies = []
# 	for j in range(k):
# 		kXtest = np.array(Xscaled[j * size : (j + 1) * size])
# 		kytest = np.array(y[j * size : (j + 1) * size])
# 		if j == 0:
# 			kXtrain = np.array(Xscaled[(j + 1) * size : len(Xscaled)])
# 			kytrain = np.array(y[(j + 1) * size : len(y)])
# 		elif j + 1 == k:
# 			kXtrain = np.array(Xscaled[0 : j * size])
# 			kytrain = np.array(y[0 : j * size])
# 		else:
# 			kXtrain = np.append(np.array(Xscaled[0 : j * size]), np.array(Xscaled[(j + 1) * size : len(Xscaled)]), axis = 0)
# 			kytrain = np.append(np.array(y[0 : j * size]), np.array(y[(j + 1) * size : len(y)]), axis = 0)

# 		# get data and fit it
# 		kytrain = [cantorEncode(sam[0], sam[1]) for sam in kytrain]
# 		dataFit = classifier.fit(kXtrain, kytrain)

# 		# test the predictions
# 		# and calculate the accuracy
# 		correct = 0
# 		prediction = dataFit.predict(kXtest)
# 		for m in range(len(kytest)):
# 			yval = kytest[m]
# 			xval = cantorDecode(prediction[m])
# 			loss = max(0, np.abs(math.sqrt((yval[0] - xval[0])*(yval[0] - xval[0]) 
# 								+ (yval[1] - xval[1])*(yval[1] - xval[1]))) 
# 								- epsilon)
# 			if loss == 0:
# 				correct += 1

# 		accuracy = float(correct)/len(kytest)
# 		accuracies.append(accuracy)
# 		if False:
# 			print "k-fold (" + str(j) + "):",  str(accuracy), "(True: " + str(correct) + ", false: " + str(len(kytest) - correct) + ")"

# 	if True:
# 		print "Average:", sum(accuracies) / len(accuracies)

# 	if sum(accuracies) / len(accuracies) > bestCVEncoded:
# 		bestCEncoded = i
# 		bestCVEncoded = sum(accuracies) / len(accuracies)
# 	print "Best C is " + str(bestCEncoded) + " with " + str(bestCVEncoded) + " on " + str(i)

	# svr_rbf.fit(Xscaled, y)

	# temp_cv = cross_val_score(svr_rbf, Xscaled, y, cv=10).mean()
	# print(temp_cv)

	#Multiple C
	# if temp_cv>max_cv:
	# 	maxC = i
	# 	max_cv = temp_cv

	# print "It took me: ", int(time.time()-start), "s for training stuff, C= ", i

# ---------------
# --- Two Models Run
# ---------------
# max_cv = -100
# maxC = 10000
# bestC = 1;
# bestCV = 0;
# for i in range(1): #range(1000000, 1139040, 25000):
# 	start = time.time()

# 	# Use 0-epsilon to customize it later
# 	Xclassifier = SVR(kernel='rbf', C=1125000, epsilon=20)
# 	Yclassifier = SVR(kernel='rbf', C=1125000, epsilon=20)

# 	# split data set into different sizes, and train the model
# 	k = 10
# 	size = len(Xscaled) / k
# 	epsilon = 20;
# 	accuracies = []
# 	for j in range(k):
# 		kXtest = np.array(Xscaled[j * size : (j + 1) * size])
# 		kytestFirst = np.array(y[j * size : (j + 1) * size])[:,0]
# 		kytestSecond = np.array(y[j * size : (j + 1) * size])[:,1]
# 		if j == 0:
# 			kXtrain = np.array(Xscaled[(j + 1) * size : len(Xscaled)])
# 			kytrainFirst = np.array(y[(j + 1) * size : len(y)])[:,0]
# 			kytrainSecond = np.array(y[(j + 1) * size : len(y)])[:,1]
# 		elif j + 1 == k:
# 			kXtrain = np.array(Xscaled[0 : j * size])
# 			kytrainFirst = np.array(y[0 : j * size])[:,0]
# 			kytrainSecond = np.array(y[0 : j * size])[:,1]
# 		else:
# 			kXtrain = np.append(np.array(Xscaled[0 : j * size]), np.array(Xscaled[(j + 1) * size : len(Xscaled)]), axis = 0)
# 			kytrainFirst = np.append(np.array(y[0 : j * size]), np.array(y[(j + 1) * size : len(y)]), axis = 0)[:,0]
# 			kytrainSecond = np.append(np.array(y[0 : j * size]), np.array(y[(j + 1) * size : len(y)]), axis = 0)[:,1]

# 		# get data and fit it
# 		dataFitFirst = Xclassifier.fit(kXtrain, kytrainFirst)
# 		dataFitSecond = Yclassifier.fit(kXtrain, kytrainSecond)

# 		# test the predictions
# 		# and calculate the accuracy
# 		correct = 0
# 		predictionFirst = dataFitFirst.predict(kXtest)
# 		predictionSecond = dataFitSecond.predict(kXtest)
# 		for m in range(len(kytestFirst)):
# 			lossFirst = max(0, np.abs(predictionFirst[m] - kytestFirst[m]) - epsilon)
# 			lossSecond = max(0, np.abs(predictionSecond[m] - kytestSecond[m]) - epsilon)
# 			if lossFirst == 0 and lossSecond == 0:
# 				correct += 1

# 		accuracy = float(correct)/len(kytestFirst)
# 		accuracies.append(accuracy)
# 		if True:
# 			print "k-fold (" + str(j) + "):",  str(accuracy), "(True: " + str(correct) + ", false: " + str(len(kytestFirst) - correct) + ")"

# 	if True:
# 		print "Average:", sum(accuracies) / len(accuracies)
	
# 	if bestCV < sum(accuracies) / len(accuracies):
# 		bestC = i
# 		bestCV = sum(accuracies) / len(accuracies)
# 	print "Best C is " + str(bestC) + " with " + str(bestCV) + " on " + str(i)

# print "Encoded " + str(bestCEncoded) + " (" + str(bestCVEncoded) + ") and double " + str(bestC) + " (" + str(bestCV) + ") "


#Uncomment to run for multiple C
# svr_rbf = SVR(kernel='rbf', C=maxC, epsilon=cantorEncode(29, 29))
# svr_rbf.fit(Xscaled, y)

