import glob
from PIL import Image
from pandas.io.parsers import read_csv
import os
import tkSimpleDialog
import Tkinter, tkFileDialog
from cnnmodel import train


root = Tkinter.Tk()
root.withdraw()
SAVE_FIX = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select your directory with TESTING and TRAINING folders for CNN landmarking')

FTRAIN_FIX = os.path.join(SAVE_FIX, 'training\scaled')
FTEST_FIX = os.path.join(SAVE_FIX, 'testing\scaled')


#############  Get jpg dimensions  #############################################
os.chdir(FTRAIN_FIX)

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

diagnostic_feature_name = tkSimpleDialog.askstring(title="Diagnostic feature",
                                  prompt="Diagnostic morphology feature name?:")

epochs = tkSimpleDialog.askinteger(title="Epochs",
                                  prompt="How many training epochs must be run?:")

DATA=[diagnostic_feature_name]


for i in DATA:
	ftrain = os.path.join(FTRAIN_FIX, i + ".csv")
	ftest = os.path.join(FTEST_FIX, i + ".csv")
	savemodel = os.path.join(SAVE_FIX, i + ".pickle")
	saveloss = os.path.join(SAVE_FIX,  i + "_loss.jpg")
	savetest = os.path.join(SAVE_FIX, i + "_test.jpg")

	df = read_csv(ftrain)
	df = df.dropna()
	landmarks = (df.shape[1]-1)

	train(landmarks, ftrain,ftest,epochs,savemodel,saveloss,savetest)


print("Finish!!")
