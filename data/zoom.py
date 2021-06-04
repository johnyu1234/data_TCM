
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys


def zero(number):
    if(number <0):
        number=0
    return number 
           
path='X41/'
#images = glob.glob(path+'/*.jpg')
images=glob.glob(path+'/*.jpg')
print(images)
#img = cv2.imread('C:/Users/朱嘉瑩/Desktop/中藥圖片/A/VID_20200713_1358001.jpg')
#print(img)
#-------------------------

k=0
for fname in os.listdir(path): 

    if (fname !='desktop.ini'):
        k=k+1 
        image=cv2.imread(path+fname)  
        #image = cv2.imread('C:/Users/Jeff Liou/Desktop/image_class_python/10015.jpg')
    
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    
        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        # blur and threshold the image
        blurred = cv2.blur(gradient, (9, 9))
        (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # perform a series of erosions and dilations
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)
        (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    
        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
    
        # draw a bounding box arounded the detected barcode and display the image
        #不畫出來
    
        
        '''
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        cv2.imshow("Image", image)
        cv2.imwrite("contoursImage2.jpg", image) 
        cv2.waitKey(0)
        '''
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = zero(min(Xs))
        x2 = zero(max(Xs))
        y1 = zero(min(Ys))
        y2 = zero(max(Ys))
        hight = y2 - y1
        width = x2 - x1
        if hight >width :
            width=hight
        else:
            hight=width
        #需要先建立new資料夾
        cropImg = image[y1:y1+hight, x1:x1+width]
        #cv2.imshow( "cropImg" , cropImg)
        newpath='new/'+str(k)
        cv2.imwrite( newpath+".jpg" , cropImg)
        
        #cv2.waitKey()

