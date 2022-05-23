from PIL import Image
import os
import Tkinter, tkFileDialog
import fnmatch
import tkSimpleDialog
import csv

root = Tkinter.Tk()
root.withdraw()
folder = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select your directory with TRAINING images for data augmentation')


diagnostic_feature_name = tkSimpleDialog.askstring(title="Diagnostic feature",
                                  prompt="Diagnostic morphology feature name?:")


for file in os.listdir(folder):
    if fnmatch.fnmatch(file, '*.jpg'):
        img = Image.open(os.path.join(folder, file))
        data = img.getdata()

        r = [(d[0],0,0) for d in data]
        g = [(0,d[1],0) for d in data]
        b = [(0,0,d[2]) for d in data]

        img.putdata(r)
        img.save(os.path.join(folder, 'red' + file))
        img.putdata(g)
        img.save(os.path.join(folder, 'green' + file))
        img.putdata(b)
        img.save(os.path.join(folder, 'blue' + file))
        print file

input_csv = os.path.join(folder, diagnostic_feature_name + ".csv")
output_csv_red = os.path.join(folder, diagnostic_feature_name + "red.csv")
output_csv_blue = os.path.join(folder, diagnostic_feature_name + "blue.csv")
output_csv_green = os.path.join(folder, diagnostic_feature_name + "green.csv")

#red
with open(output_csv_red ,'wb') as outFile:
    fileWriter = csv.writer(outFile)
    with open(input_csv,'r') as inFile:
        fileReader = csv.reader(inFile)
        next(fileReader)
        for row in fileReader:
            fileWriter.writerow(row)

text = open(output_csv_red, "r")
text = ''.join([i for i in text]) \
    .replace("raining/", "raining/red")
x = open(output_csv_red,"w")
x.writelines(text)
x.close()

#blue
with open(output_csv_blue ,'wb') as outFile:
    fileWriter = csv.writer(outFile)
    with open(input_csv,'r') as inFile:
        fileReader = csv.reader(inFile)
        next(fileReader)
        for row in fileReader:
            fileWriter.writerow(row)

text = open(output_csv_blue, "r")
text = ''.join([i for i in text]) \
    .replace("raining/", "raining/blue")
x = open(output_csv_blue,"w")
x.writelines(text)
x.close()

#green
with open(output_csv_green ,'wb') as outFile:
    fileWriter = csv.writer(outFile)
    with open(input_csv,'r') as inFile:
        fileReader = csv.reader(inFile)
        next(fileReader)
        for row in fileReader:
            fileWriter.writerow(row)

text = open(output_csv_green, "r")
text = ''.join([i for i in text]) \
    .replace("raining/", "raining/green")
x = open(output_csv_green,"w")
x.writelines(text)
x.close()

#combine csv files

with open(input_csv,'ab') as outFile:
    fileWriter = csv.writer(outFile)
    with open(output_csv_red,'r') as inFile:
        fileReader = csv.reader(inFile)
        for row in fileReader:
            fileWriter.writerow(row)

with open(input_csv,'ab') as outFile:
    fileWriter = csv.writer(outFile)
    with open(output_csv_blue,'r') as inFile:
        fileReader = csv.reader(inFile)
        for row in fileReader:
            fileWriter.writerow(row)

with open(input_csv,'ab') as outFile:
    fileWriter = csv.writer(outFile)
    with open(output_csv_green,'r') as inFile:
        fileReader = csv.reader(inFile)
        for row in fileReader:
            fileWriter.writerow(row)

#delete extra files
os.remove(output_csv_green)
os.remove(output_csv_red)
os.remove(output_csv_blue)


print 'Finish!'