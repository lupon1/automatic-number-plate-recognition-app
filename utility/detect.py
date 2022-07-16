# Functions for ANPR APP
# Francesco Esposito - July 2022

import cv2
import torch
from os import path
from time import time

# Variables
workingPath = path.dirname(__file__)
plateModelPath = path.join(workingPath, '..', 'models', 'licensePlates_v7.4.pt')
ocrModelPath = path.join(workingPath, '..', 'models', 'ocr_v3.3.pt')

classesIndex = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z']

# Load models
plate = torch.hub.load('ultralytics/yolov5', 'custom', path=plateModelPath)
ocr = torch.hub.load('ultralytics/yolov5', 'custom', path=ocrModelPath)
plate.conf = 0.70
ocr.conf = 0.35
plate.names = ['plate']


# Main functions
def detect_plate(img):
    '''
    A function that founds license plates in a image and returns the tensor
    of the nearest one

    Args:
        img (opencv image): an image of a vehicle

    Returns:
        tensors of the nearest detected license plate
    '''
    # Inference to find a license plate in the image/frame
    results = plate(img, size=640)
    # Keep the biggest license plate
    maxRes = 0  # Max resolution
    tensors = results.xyxy[0]
    for tens in tensors:
        resolution = (int(tens[2]) - int(tens[0])) * (int(tens[3]) - int(tens[1]))
        if resolution >= maxRes:
            maxRes = resolution
            result = tens
    # If a plate is present in the image, returns the biggest one
    if maxRes > 0:
        return result

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def detect_ocr(img, tensor):
    '''
    A function to detect characters in a image of license plate

    Args:
        img (opencv image): The image of a vehicle
        tensor (pytorch tensor): the tensor of the detected license plate in the image
    Returns:
        tensors of the characters detected in the license plate
    '''

    def image_crop(img, tensor):
        '''
        A function to crop a opencv image

        Args:
            img (opencv image): the image of a vehicle in opencv format
            tensor (pytorch tensor): the tensor of the detected license plate in the image

        Returns:
            opencv image: the cropped image of the license plate
        '''

        # Find the 2 points from the tensor to crop the image
        xmin = int(tensor[0].item())
        ymin = int(tensor[1].item())
        xmax = int(tensor[2].item())
        ymax = int(tensor[3].item())
        croppedImage = img[ymin:ymax, xmin:xmax]
        return croppedImage

    def image_resize(img, size=416, inter=cv2.INTER_LINEAR):
        '''
        A function to resize the license plate image

        Args:
            img (opencv image): the detected image of the license plate
            size (int, optional): The size of the biggest side of output image. Defaults to 416.
            inter (int or cv.INTER_, optional): Interpolation methods.
                                                Can be: cv.INTER_NEAREST,
                                                        cv.INTER_LINEAR,
                                                        cv.INTER_CUBIC,
                                                        cv.INTER_AREA,
                                                        cv.INTER_LANCZOS4
                                                Defaults to cv2.INTER_LINEAR.

        Returns:
            opencv image: the resized image of the license plate
        '''
        # Detect the biggest side
        w, h = img.shape[1], img.shape[0]
        if w > h:
            maxSide = w
        else:
            maxSide = h
        prop = size / maxSide  # Calculate the proportions
        newSize = (int(w * prop), int(h * prop))
        resizedImage = cv2.resize(img, newSize, interpolation=inter)
        return resizedImage

    def image_preprocess(img):
        '''
        A function to apply some filters at the license plate image before pass to the ocr model

        Args:
            img (opencv image): the resized image of the license plate

        Returns:
            opencv image: the processed license plate image
        '''

        # Convert the image to black and white
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply blur to the image
        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
        # Apply Threshold or binarization to the image
        imgThresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 39, 9)
        return imgThresh

    # Apply some process to the image and pass it to the OCR algorithm
    imgCrop = image_crop(img, tensor)
    imgResized = image_resize(imgCrop)
    imgProcessed = image_preprocess(imgResized)
    results = ocr(imgProcessed, size=416)
    if len(results) > 0:
        return results.xyxy[0]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_number_plate(tensores, skewThres=8, debug=False):
    '''
    A function that returns the string of correct order number plate

    Args:
        tensores (pytorch tensor): the tensor of the inference runs on processed image
        skewThres (int, optional): the threshold to identify when a license plate is tilted. Defaults to 8.
        debug (bool, optional): Debug option to print some info. Defaults to False.
    '''

    def get_plate(uppers):
        '''
        A function to identify the correct order of the inferenced characters
        in a horizontal license plate

        Args:
            uppers (list): a list of detected chars tensors sorted by uppers

        Returns:
            str: the string of the number plate in the correct order
        '''
        uppers = sorted(uppers, key=lambda tup: tup[2])
        plate = ''.join([classesIndex[e[-2]] for e in uppers])
        return plate

    def get_skew_plate(uppers):
        '''
        A function to identify the correct order of the inferenced characters
        in a skewed license plate

        Args:
            uppers (list): a list of detected chars tensors sorted by uppers

        Returns:
            str: the string of the number plate in the correct order
        '''
        plateList = [[classesIndex[uppers[0][-2]]]]
        plateIndex = 0
        for i in range(1, len(uppers)):
            # Checks if the character is at right or left of previus character
            if uppers[i][0] > uppers[i-1][0]:  # Down left
                if skew > 0:
                    plateIndex += 1
                    plateList.append([])
                plateList[plateIndex].append(classesIndex[uppers[i][-2]])
            else:  # Top right
                if skew < 0:
                    plateIndex += 1
                    plateList.append([])
                plateList[plateIndex].append(classesIndex[uppers[i][-2]])
        # Join chars of license plate
        plate = ''
        for lst in plateList:
            if skew > 0:
                lst.reverse()
            plate += ''.join(lst)
        return plate

    # Start get_number_plate()
    # To delete doble detections
    distThres = 25
    tensores = tensores.numpy().tolist()
    newTens = tensores.copy()
    for i in range(len(tensores)):
        x1, y1 = float(tensores[i][0]), float(tensores[i][1])
        for n in range(i+1, len(tensores)):
            x2, y2 = float(tensores[n][0]), float(tensores[n][1])
            distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
            if distance < distThres:
                if float(tensores[i][-2]) > float(tensores[n][-2]):
                    newTens.remove(tensores[n])
                else:
                    newTens.remove(tensores[i])

    # To calculate skew factor
    if len(newTens) > 0:
        uppers = [(float(t[0]), float(t[1]), float(t[0])+float(t[1])*3, float(t[-2]), int(t[-1]), i) for i, t in enumerate(newTens)]
        uppers = sorted(uppers, key=lambda tup: tup[1])
        # To calculate skew
        skew = 0
        if len(uppers) > 1:
            p1 = int(uppers[0][0]), int(uppers[0][1])
            p2 = int(uppers[1][0]), int(uppers[1][1])
            if p1[0] < p2[0]:
                skew = p1[-1] - p2[-1]
            else:
                skew = p2[-1] - p1[-1]

        # Debug section
        if debug:
            print()
            print('DEBUG')
            print('Uppers', [classesIndex[u[-2]] for u in uppers])
            print('Skew:', skew)
            print('Characters:', len(tensores), '>>', len(newTens))
            print()

        # To calculate confidence mean
        conf = [t[-3] for t in uppers]
        OCRconfMean = int((sum(conf) / len(conf))*100)

        # To obtain the license plate number depending on the skew
        if -abs(skewThres) < skew < skewThres:
            return get_plate(uppers), OCRconfMean
        else:
            return get_skew_plate(uppers), OCRconfMean
    else:
        return 'NO OCR DETECTIONS', 0


def plate_from_photo(img, debug=False):
    '''
    A function that runs inference on a image to found a license plate and
    returns his OCR detection with some stats

    Args:
        img (opencv image): the image to inference
        debug (bool, optional): Debug option to print some info. Defaults to False.

    Returns:
        _type_: _description_
        opencv image: the image with detected plate and his number if any is detected
        str: the text of the detected license plate
        int: the mean of all characters inference confidence
        float: the time in ms of all inference process
    '''
    startTime = time()
    tensor = detect_plate(img)
    if tensor is not None:
        ocrTensors = detect_ocr(img, tensor)
        if ocrTensors is not None:
            numberPlate, OCRconfMean = get_number_plate(ocrTensors, debug=debug)
            # Draw the detected license plate and his number on the original image
            xMin, yMin, xMax, yMax = int(tensor[0]), int(tensor[1]), int(tensor[2]), int(tensor[3])
            charSize = (img.shape[1]/100)*0.12
            lineSize = int(round(charSize*2, 0))
            outputImg = cv2.rectangle(img, (xMin, yMin), (xMax, yMax), (255, 164, 101), lineSize)
            cv2.putText(outputImg,
                        numberPlate,
                        (xMin, yMin-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        charSize,
                        (255, 164, 101),
                        lineSize)
            endTime = int((time() - startTime) * 1000)
            return outputImg, numberPlate, OCRconfMean, endTime
        else:
            endTime = int((time() - startTime) * 1000)
            return img, 'NO DETECTIONS', 0, endTime
    else:
        endTime = int((time() - startTime) * 1000)
        return img, 'NO DETECTIONS', 0, endTime


# For test purpouses
if __name__ == '__main__':
    imgPath = path.join(workingPath, 'testImages', 'wt1.jpg')  # Image path for test
    img = cv2.imread(imgPath)  # Open the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB space
    finalImg, license, OCRconfMean, infTime = plate_from_photo(img, debug=False)  # Detection and prediction
    print(f'License plate: {license}, OCR Confidence: {OCRconfMean}%')  # Print results
    finalImg = cv2.cvtColor(finalImg, cv2.COLOR_RGB2BGR)  # Convert to BGR space
    finalImg = cv2.resize(finalImg, (640, 480))
    cv2.imshow(license, finalImg)  # Show image in a window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
