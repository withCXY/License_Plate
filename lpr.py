from flask import Flask, request, jsonify
import numpy as np
import cv2 
from flask import render_template
import os
import torch.nn as nn

import Find_Chars
import Find_Plate


# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0) #(B,G,R)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLUE = (255.0, 165.0,0)
SCALAR_ORANGE = (0, 165.0, 255.0)

showSteps = False

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if not os.path.exists('./static/uploads/'):
        os.makedirs('./static/uploads/')
    file = request.files['image']
    filepath = './static/uploads/' + file.filename
    file.save(filepath)
    
    image = cv2.imread(filepath)
    license_plate = recognize_license_plate(image, file.filename)
    original_image_path = filepath
    generated_image_path = './static/uploads/new_' + file.filename
    # imgPlate_path = './static/uploads/imgPlate_' + file.filename
    # imgThresh_path = './static/uploads/imgThresh_' + file.filename
    return jsonify({'result': license_plate, 'original_image_path': original_image_path, 'generated_image_path': generated_image_path}) # , 'imgPlate_path': imgPlate_path, 'imgThresh_path': imgThresh_path})

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 4000)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(4000, 1000)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(1000, 43)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)

    p0 = [int(i) for i in p2fRectPoints[0]]
    p1 = [int(i) for i in p2fRectPoints[1]]
    p2 = [int(i) for i in p2fRectPoints[2]]
    p3 = [int(i) for i in p2fRectPoints[3]]

    cv2.line(imgOriginalScene, tuple(p0), tuple(p1), SCALAR_YELLOW, 2)
    cv2.line(imgOriginalScene, tuple(p1), tuple(p2), SCALAR_YELLOW, 2)
    cv2.line(imgOriginalScene, tuple(p2), tuple(p3), SCALAR_YELLOW, 2)
    cv2.line(imgOriginalScene, tuple(p3), tuple(p0), SCALAR_YELLOW, 2)

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0 
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0 
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # choose a plain jane font
    fltFontScale = float(plateHeight) / 30.0                    # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))           # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # call getTextSize

            # unpack roatated rect into center point, width and height, and angle
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # write the chars in below the plate
    else:                                                                                       # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # write the chars in above the plate

    textSizeWidth, textSizeHeight = textSize                # unpack text size width and height

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          # based on the text area center, width, and height

    
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_BLUE, intFontThickness)


def recognize_license_plate(img, filename):

    imgOriginalScene  = img            # open image

    if imgOriginalScene is None:                            # if image was not read successfully
        print("\nerror: image not read from file \n\n")  # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit program
    # end if

    listOfPossiblePlates = Find_Plate.detectPlatesInScene(imgOriginalScene)           # detect plates

    listOfPossiblePlates = Find_Chars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

    if len(listOfPossiblePlates) == 0:                          # if no plates were found
        print("\nno license plates were detected\n")  # inform user no plates were found
    else:                                                       # else
                # if we get in here list of possible plates has at leat one plate

                # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]

        cv2.imwrite("./static/uploads/imgPlate_" + filename, licPlate.imgPlate)           # show crop of plate and threshold of plate
        cv2.imwrite("./static/uploads/imgThresh_" + filename, licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     # if no chars were found in the plate
            print("\nno characters were detected\n\n")  # show message
            return 'no chars'                                         
        
        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)
        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)
        cv2.imwrite("./static/uploads/new_" + filename, imgOriginalScene)
        
    return licPlate.strChars



if __name__ == '__main__':
    app.run(debug=True, port=8080)