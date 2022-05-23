from PIL import Image
import os, sys
import glob
#####
import Tkinter, tkFileDialog
import tkSimpleDialog
import os
import csv
import fnmatch

root = Tkinter.Tk()
root.withdraw()
dirname = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select your directory with TESTING and TRAINING folders for CNN landmarking')
#######

path_Training = os.path.join(dirname, "Training")
path_Testing = os.path.join(dirname, "Testing")
#path_Training = r"C:\Deon\Spiracles FULL RES\Training"
#path_Testing = r"C:\Deon\Spiracles FULL RES\testing"
#original_csv = os.path.join(path_Training, "spiracle.csv")
original_csv = tkFileDialog.askopenfilename(parent=root,initialdir=path_Training,title='Please select csv file with training landmark data')

if not os.path.exists(os.path.join(path_Testing,"scaled")):
    os.mkdir(os.path.join(path_Testing,"scaled"))
if not os.path.exists(os.path.join(path_Training,"scaled")):
    os.mkdir(os.path.join(path_Training,"scaled"))


file1 = open(original_csv,"r")
output_Training = os.path.join(path_Training, "scaled")
output_Testing = os.path.join(path_Testing, "scaled")

diagnostic_feature_name = tkSimpleDialog.askstring(title="Diagnostic feature",
                                  prompt="Diagnostic morphology feature name?:")

scaled_csv = os.path.join(output_Training, diagnostic_feature_name + ".csv")
file2 = open(scaled_csv,"w")


scale_percent = tkSimpleDialog.askinteger(title="Scale images",
                                  prompt="Scale percent?:")
#scale_percent = 10


if not os.path.exists(output_Training):
    os.makedirs(output_Training)
if not os.path.exists(output_Testing):
    os.makedirs(output_Testing)

################################################################################
def resize_jpg():
        os.chdir(path_Training)
        for file in glob.glob("*.JPG"):
            print file
            img = Image.open(file)
            width, height = img.size
            new_width = int(width * scale_percent / 100)
            new_height = int(height * scale_percent / 100)
            imResize = img.resize((new_width,new_height), Image.ANTIALIAS)
            imResize.save(os.path.join(output_Training, file), 'JPEG', quality=90)


        os.chdir(path_Testing)
        for file in glob.glob("*.JPG"):
            print file
            img = Image.open(file)
            width, height = img.size
            new_width = int(width * scale_percent / 100)
            new_height = int(height * scale_percent / 100)
            imResize = img.resize((new_width,new_height), Image.ANTIALIAS)
            imResize.save(os.path.join(output_Testing, file), 'JPEG', quality=90)

################################################################################
def resize_landmarks(file1, file2):
    abc = file1.readlines()
    for line in range (0, len(abc)):
        newline = ""
        if line == 0:
            file2.write(abc[line])
        else:
            head, tail = os.path.split(abc[line].split(',')[0])

            newline = os.path.join(head, "scaled", tail)
            for item in range(1, len(abc[line].split(','))):
                value = float(abc[line].split(',')[item])

                newline = newline + "," + str(int(value * scale_percent / 100))

                if item == len(abc[line].split(','))-1:
                    newline = newline + "\n"
                    file2.write(newline)

    file1.close()
    file2.close()
    print ""
    print "Done with landmarks"


resize_jpg()
resize_landmarks(file1, file2)

path_Testing_scaled = os.path.join(path_Testing, "scaled")
testing_scaled_csv = os.path.join(path_Testing_scaled, diagnostic_feature_name + ".csv")
#path = 'C:/Deon/Spiracles FULL RES/testing/scaled'

with open(testing_scaled_csv, 'wb') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(['ImageId', 'Image'])
  for root, dirs, files in os.walk(path_Testing_scaled):
    for filename in files:
        if not fnmatch.fnmatch(filename, '*.csv'):
            writer.writerow([filename, os.path.join(root,filename)])


#for root, dirs, files in os.walk(path_Testing_scaled):
#    for filename in files:
#        if not fnmatch.fnmatch(filename, '*.csv'):


#with open(testing_scaled_csv, 'wb') as csvfile:
#  writer = csv.writer(csvfile)
#  writer.writerow(['ImageId', 'Image'])
#  for root, dirs, images in os.walk(path_Testing_scaled):
#    for filename in images:
#        writer.writerow([filename, os.path.join(root,filename)])

#f = open(testing_scaled_csv, "r+")
#lines = f.readlines()
#lines=lines[:-1]
#f = open(testing_scaled_csv, "w+")
#f.writelines(lines)
