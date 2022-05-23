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


import os
import cv2
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

#FTRAIN = r"L:\Deon\CNN landmarking first try\Ornithodoros train.csv"
#FTRAIN_FIX = r'C:\Try in new folder\Training\scaled'
#FTEST_FIX = r'C:\Try in new folder\testing\scaled'

#mamke thingy to get width independandty from path and image...
#images_height = 193
#images_width = 258
height_txt = open('C:\\CNNlandmarking\\images_height.txt', "r")
width_txt = open('C:\\CNNlandmarking\\images_width.txt', "r")

images_height = int(height_txt.readline())
images_width = int(width_txt.readline())

#print "Height", images_height, "Width", images_width

def readImage(arrayImgs):
    X = arrayImgs.values
    for i in range(len(X)):
        #print arrayImgs[i], "readimage"
        image = cv2.imread(X[i],cv2.IMREAD_GRAYSCALE)
        image = image.reshape(-1)
        image = image/255.
        image = image.astype(np.float32)
        X[i] = image
    X = np.vstack(X)
    return X

#scale the target to [-1,1]
def scaleTarget(target):
    print('Normalize target...')
    #print target
    #print target.shape
    #evencol = (target[:,::2] - 128)/128    # LE 192,256
    #ddcol = (target[:,1::2] - 96)/96
    value1 = images_width/2 #width
    value2 = images_height/2  #height
    evencol = (target[:,::2] - value1)/value1
    oddcol = (target[:,1::2] - value2)/value2
    #print ""
    #print "evencol", len(evencol), evencol[0], evencol.shape[0], evencol.shape[1]
    #print ""
    #print "oddcol", len(oddcol), oddcol[0], oddcol.shape[0], oddcol.shape[1]
    #print ""
    rs = np.empty((evencol.shape[0],evencol.shape[1] + oddcol.shape[1]))
    rs[:,::2] = evencol
    rs[:,1::2] = oddcol
    '''
    for i in range(len(target)):
            targeti = target[i]
            for j in range(len(targeti)):
                    if j % 2 == 0:
                            targeti[j] = (targeti[j] - value1)/value1
                    else:
                            targeti[j] = (targeti[j] - value2)/value2
            target[i] = targeti
    '''
    return rs
# try make value1 and value 2 in multi-line string above
def loaddata(fname = None,test=False):
    if fname == None:
        fname = FTEST_FIX if test else FTRAIN_FIX
    df = read_csv(os.path.expanduser(fname))
    #print df
    #df = read_csv(os.path.expanduser(FTRAIN))
    print fname, "loaddata"
    df = df.dropna()
    imagePath = df['Image']
    #print imagePath[0]
    #print imagePath[1]
    X = readImage(imagePath)
    if not test:
        #y = df[df.columns[:-1]].values # why did LE have this and Philip another?
        y = df[df.columns[1:]].values
        y = y.astype(np.float32)
        #print y[0]
        #print y[1]
        y = scaleTarget(y)
        #print X[1]
        #print y[1]
        X,y = shuffle(X,y,random_state=42)
        y = y.astype(np.float32)
        #print X
        #print 'break'
        #print y
    else:
        y = None
    return X,y





# test loaddata method
#X,y = loaddata()
#print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
#    X.shape, X.min(), X.max()))
#print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
#    y.shape, y.min(), y.max()))

# reshape (convert) the data from 49152 to 192x256 (h x w)
def load2d(fname=None,test=False):
    print fname, "load2d"
    X,y = loaddata(fname, test=test)
    #X = X.reshape(-1,1,192,256)
    X = X.reshape(-1,1, images_height, images_width)
    if not test:
        print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(X.shape, X.min(), X.max()))
        print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(y.shape, y.min(), y.max()))
    return X,y

