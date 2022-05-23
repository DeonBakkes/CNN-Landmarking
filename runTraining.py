import glob
from PIL import Image
from pandas.io.parsers import read_csv
import os
import tkSimpleDialog
import Tkinter, tkFileDialog
from cnnmodel import train
#import gc
#import multiprocessing as mp
#scripts_dir = os.getcwd()
#print scripts_dir
#gc.disable()
#gc_thresh=gc.get_threshold()
#print gc_thresh

root = Tkinter.Tk()
root.withdraw()
SAVE_FIX = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select your directory with TESTING and TRAINING folders for CNN landmarking')
#SAVE_FIX = r'C:\RGB split try'
FTRAIN_FIX = os.path.join(SAVE_FIX, 'training\scaled')
FTEST_FIX = os.path.join(SAVE_FIX, 'testing\scaled')

#SAVE_FIX_txt = open('C:\\CNNlandmarking\\SAVE_FIX.txt', "w")
#SAVE_FIX_txt.write(str(SAVE_FIX))
#SAVE_FIX_txt.close()

#############  Get jpg dimensions  #############################################
os.chdir(FTRAIN_FIX)
#glob.glob("*.JPG")
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
################################################################################
#import readCSV


#parent_conn, child_conn = mp.Pipe(duplex=False)
#p = mp.Process(target=readCSV.main, args=(child_conn,))
#p.start()
#parent_conn.send(images_height)
#parent_conn.send(images_width)#

#FTEST_FIX = tkFileDialog.askdirectory(parent=root,initialdir=SAVE_FIX,title='Please select the TESTING directory with scaled images')
#r'C:\Try in new folder\Training\scaled'
#DATA=['v10']
#DATA=['spiracle']

diagnostic_feature_name = tkSimpleDialog.askstring(title="Diagnostic feature",
                                  prompt="Diagnostic morphology feature name?:")
#diagnostic_feature_name = r'female spiracle'
#DATA=['spiracle']
#FTRAIN_FIX = r"C:\Deon\Test data\train"
#FTEST_FIX = r"C:\Deon\Test data\test"
#SAVE_FIX = r"C:\Deon\Test data\results"

#diagnostic_feature_name_txt = open('C:\\CNNlandmarking\\diagnostic_feature_name.txt', "w")
#diagnostic_feature_name_txt.write(str(diagnostic_feature_name))
#diagnostic_feature_name_txt.close()


epochs = tkSimpleDialog.askinteger(title="Epochs",
                                  prompt="How many training epochs must be run?:")
#epochs = 2000
#epochs_txt = open('C:\\CNNlandmarking\\epochs.txt', "w")
#epochs_txt.write(str(epochs))
#epochs_txt.close()
#epochs_txt = open('C:\\CNNlandmarking\\epochs.txt', "r")
#epochs = int(epochs_txt.readline())

#diagnostic_feature_name_txt = open('C:\\CNNlandmarking\\diagnostic_feature_name.txt', "w")
#diagnostic_feature_name_txt.write(str(diagnostic_feature_name))
#diagnostic_feature_name_txt.close()
#diagnostic_feature_name_txt = open('C:\\CNNlandmarking\\diagnostic_feature_name.txt', "r")
#diagnostic_feature_name = str(diagnostic_feature_name_txt.readline())

#SAVE_FIX_txt = open('C:\\CNNlandmarking\\SAVE_FIX.txt', "w")
#SAVE_FIX_txt.write(str(SAVE_FIX))
#SAVE_FIX_txt.close()
#SAVE_FIX_txt = open('C:\\CNNlandmarking\\SAVE_FIX.txt', "r")
#SAVE_FIX = str(SAVE_FIX_txt.readline())



DATA=[diagnostic_feature_name]

#epochs = 5000
for i in DATA:
	ftrain = os.path.join(FTRAIN_FIX, i + ".csv")
	ftest = os.path.join(FTEST_FIX, i + ".csv")
	savemodel = os.path.join(SAVE_FIX, i + ".pickle")
	saveloss = os.path.join(SAVE_FIX,  i + "_loss.jpg")
	savetest = os.path.join(SAVE_FIX, i + "_test.jpg")

	df = read_csv(ftrain)
	df = df.dropna()
	landmarks = (df.shape[1]-1)

	#print(ftrain)
	#print(ftest)
	#print(savemodel)
	#print(saveloss)
	#print(savetest)
	train(landmarks, ftrain,ftest,epochs,savemodel,saveloss,savetest)


print("Finish!!")
