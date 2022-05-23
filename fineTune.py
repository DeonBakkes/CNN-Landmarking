'''
 Elementary Blocks Network to landmark anatomical images
 Copyright (C) 2018  Le Van Linh (van-linh.le@u-bordeaux.fr)
 Version: 1.0
 Created on: March, 2018

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see http://www.gnu.org/licenses/.
'''




try:
	import cPickle as pickle
except ImportError:
	import pickle
import os
import lasagne
import sys
import glob
from PIL import Image
import numpy as np
from lasagne import layers
from nolearn.lasagne import NeuralNet, TrainSplit
from matplotlib import pyplot
from readCSV import loaddata, load2d
from utils import AdjustVariable, plot_sample, draw_loss_2, write_file
from lasagne.layers import DenseLayer
import theano
from sklearn.externals import joblib
import Tkinter, tkFileDialog
import tkSimpleDialog


root = Tkinter.Tk()
root.withdraw()
dirname = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select your directory with TESTING and TRAINING folders for CNN landmarking')

if not os.path.exists(os.path.join(dirname,"fineTuning")):
    os.mkdir(os.path.join(dirname,"fineTuning"))

path_FineTuning = os.path.join(dirname, "fineTuning")

diagnostic_feature_name = tkSimpleDialog.askstring(title="Diagnostic feature",
                                  prompt="Diagnostic morphology feature name?:")

path_Training = os.path.join(dirname, "Training")
path_Testing = os.path.join(dirname, "Testing")
output_Training = os.path.join(path_Training, "scaled")
output_Testing = os.path.join(path_Testing, "scaled")


scaled_landmarks_csv = os.path.join(output_Training, diagnostic_feature_name + ".csv")
testing_images_csv = os.path.join(output_Testing, diagnostic_feature_name + ".csv")

with open(scaled_landmarks_csv, 'r') as csv:
     first_line = csv.readline()
     your_data = csv.readlines()

ncol = first_line.count(',')

print ncol, "x & y co-ordinates"


FMODEL = os.path.join(dirname, diagnostic_feature_name + '.pickle')

os.chdir(output_Training)

img = Image.open(glob.glob("*.JPG")[0])
width, height = img.size
print "Height", height, "Width", width
images_height = height
images_width = width

height_txt = open('C:\\CNNlandmarking\\images_height.txt', "w")
height_txt.write(str(images_height))
height_txt.close()

width_txt = open('C:\\CNNlandmarking\\images_width.txt', "w")
width_txt.write(str(images_width))
width_txt.close()

epochs = tkSimpleDialog.askinteger(title="Epochs",
                                  prompt="How many finetuning epochs must be run?:")

'''
    Build the layers that have the same ordered with the trained model
'''

def build_model():
	net = {}
	net['input'] = lasagne.layers.InputLayer((None,1,images_height,images_width))

	net['conv1'] = lasagne.layers.Conv2DLayer(net['input'], 32, (3,3),pad=1)
	net['pool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'],pool_size=(2,2))
	net['drop2'] = lasagne.layers.DropoutLayer(net['pool1'],p=0.4)
	net['conv2'] = lasagne.layers.Conv2DLayer(net['drop2'], 64, (2,2),pad=1)
	net['pool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'],pool_size=(2,2))
	net['drop3'] = lasagne.layers.DropoutLayer(net['pool2'],p=0.5)
	net['conv3'] = lasagne.layers.Conv2DLayer(net['drop3'], 128, (2,2),pad=1)
	net['pool3'] = lasagne.layers.MaxPool2DLayer(net['conv3'],pool_size=(2,2))
	net['drop4'] = lasagne.layers.DropoutLayer(net['pool3'],p=0.7)
	net['hidden4'] = lasagne.layers.DenseLayer(net['drop4'],num_units=1000)
	net['drop5'] = lasagne.layers.DropoutLayer(net['hidden4'],p=0.7)
	net['hidden5'] = lasagne.layers.DenseLayer(net['drop5'],num_units=1000)
	net['output'] = lasagne.layers.DenseLayer(net['hidden5'],num_units=ncol,nonlinearity=None)
	return net

'''
    Load the trained model and copy the parameter values into the corresponding layers ( function build_model).
    Then, change the output of the last layer to fine-tune the trained model

    Parameters:
        - model_file: trained model file
'''
def set_weights(model_file):
	#print model_file

	with open(model_file, 'rb') as ff:
	   model = pickle.load(ff)

	#model = joblib.load(model_file)
	print('Set the weights...')
	#newnet = model
	print(model)
	all_param = lasagne.layers.get_all_param_values(model.layers)
	net = build_model()
	lasagne.layers.set_all_param_values(net['output'],all_param,trainable=True)
	output_layer = lasagne.layers.DenseLayer(net['hidden5'],num_units = ncol, nonlinearity=None)

	return output_layer

'''
    Build the fine_tuning model after load the trained model and change the output
    Parameters:
        - nlayers: list of layers after copy the values from trained model
'''
def build_fine_tuning_model(nlayers):
	net3 = NeuralNet(
	layers=nlayers,

		# learning parameters
		update= lasagne.updates.nesterov_momentum,


		update_learning_rate = 0.01,
		update_momentum = 0.9,
		regression=True,




		max_epochs = epochs, # maximum iteration
		train_split = TrainSplit(eval_size=0.4),
		verbose=1,
	)
	return net3

'''
def build_model2(modelfile):
	with open(modelfile) as f:
		model = pickle.load(f)
	print('Set the weights...')
	print(model)
	all_param = lasagne.layers.get_all_param_values(model.layers)
	net = build_model()
	lasagne.layers.set_all_param_values(net['output'],all_param,trainable=True)
	newlayers = lasagne.layers.DenseLayer(net['hidden5'],num_units = 16, nonlinearity=None)
	#model.layers = newlayers
	print(model)
	return model
'''

if __name__ == '__main__':

    # Load data
	FTRAINF = scaled_landmarks_csv
	FTESTF = testing_images_csv
	saveloss = os.path.join(path_FineTuning, "fineTuning_loss.jpg")
	savetest = os.path.join(path_FineTuning, "fineTuning_test.jpg")
	savemodel = os.path.join(path_FineTuning, diagnostic_feature_name + '.pickle')
	X1,y1 = load2d(FTRAINF,test=False)

	#=================================================================
	# Load the parameters into list of layer, create a new network and train
	#newlayers = set_weights(FMODEL)
	newlayers = set_weights(FMODEL)
	net2 = build_fine_tuning_model(newlayers)
	net2.fit(X1,y1)

    # Save the fine-tuning model
	sys.setrecursionlimit(150000)
	with open(os.path.join(path_FineTuning, diagnostic_feature_name + '.pickle'),'wb') as f:
		pickle.dump(net2,f,-1)


	# draw the loss
	draw_loss_2(net2, saveloss)

	# test the fine-tuning network and draw the results
	X, _ = load2d(FTESTF,test=True)
	y_pred = net2.predict(X)

	fig = pyplot.figure(figsize=(4, 4))
	fig.subplots_adjust(
    		left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
	for i in range(16):
    		ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    		plot_sample(X[i], y_pred[i], ax)
	fig.savefig(savetest,dpi=90)
	pyplot.show()


print 'Finish!'

