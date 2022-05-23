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
import sys
import numpy as np
from matplotlib import pyplot
from readCSV import loaddata, load2d
from utils import AdjustVariable, plot_sample, draw_loss_2, write_file
from pandas.io.parsers import read_csv
import theano
from sklearn.externals import joblib
import Tkinter, tkFileDialog
import tkSimpleDialog

root = Tkinter.Tk()
root.withdraw()
dirname = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select your directory with TESTING and TRAINING folders for CNN landmarking')

FTEST = os.path.join(dirname, "testing/scaled")

if not os.path.exists(os.path.join(dirname,"Save")):
    os.mkdir(os.path.join(dirname,"Save"))
if not os.path.exists(os.path.join(dirname,"Save/images")):
    os.mkdir(os.path.join(dirname,"Save/images"))
fineTuning_dir = os.path.join(dirname, "fineTuning")

def loadCSV(fname = None):
    df = read_csv(os.path.expanduser(fname))
    df = df.dropna()
    imagePaths = df['Image']
    return imagePaths

def extract_fileNames(imagePaths):
    paths = imagePaths.values
    alist=[]

    for i in range(len(paths)):

        head, tail = os.path.split(str(imagePaths[i]))
        alist.append(tail)

    print(len(alist)), "images to landmark"
    return alist

FSAVEFOLDER = os.path.join(dirname,"Save")

FSAVEIMAGES = os.path.join(FSAVEFOLDER, "images")


diagnostic_feature_name = tkSimpleDialog.askstring(title="Diagnostic feature",
                                  prompt="Diagnostic morphology feature name?:")

DATA=[diagnostic_feature_name]


pickle_file_from_training = os.path.join(fineTuning_dir, diagnostic_feature_name + '.pickle')


for i in DATA:
    fmodelf = pickle_file_from_training

    ftestf = os.path.join(FTEST, i + '.csv')
    flandmarks = diagnostic_feature_name + '.txt'

    net = None
    sys.setrecursionlimit(100000)

    with open(fmodelf, 'rb') as f:
		net = pickle.load(f)


    X, _ = load2d(ftestf,test=True)
    y_pred = net.predict(X)


    paths = loadCSV(ftestf)
    fileNames = extract_fileNames(paths)

    for i in range(len(y_pred)):
        predi = y_pred[i]

        saveImg = os.path.join(FSAVEIMAGES, fileNames[i])
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
        plot_sample(X[i],predi,ax)
        fig.savefig(saveImg)
        pyplot.close(fig)


    #print net

    for j in range(len(y_pred)):
        predi = y_pred[j]

        name = os.path.splitext(fileNames[j])[0]
        flandmarks = os.path.join(FSAVEFOLDER, name + '_' + diagnostic_feature_name + '.txt')
        write_file(flandmarks, predi)


    print('Finish!')
'''
# plot the landmarks on the images
fig = pyplot.figure(figsize=(4, 4))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i], y_pred[i], ax)

pyplot.show()
'''
