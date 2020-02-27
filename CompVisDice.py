#SAN16602715 Jacqueline Sangster
#imports
import numpy as np
import cv2

#Load the testing image(s) in
filename = 'C:\\Users\\Jacqui\\Documents\\Work\\Assignments\\Fourth Year\\Project\\Code\\dice\\train\\d8\\d8_color000.jpg'
img = cv2.imread(filename)
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#threshold the thing - use this to eliminate the background. but also the number...
ret, thresh = cv2.threshold(grayimg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#extract the corners/edges
dst = cv2.cornerHarris(grayimg, 5, 3, 0.01)
edges = cv2.Canny(img,140,150)
#do something to make the lines constant
kernel = np.ones((11,11), np.uint8)
#edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#get a fill of the thing up to the line or something

#segmentation by texture
gKernel180 = cv2.getGaborKernel((21,21), 9.0, 180, 10.0, 0.5, 0, ktype=cv2.CV_32F)
textureImg180 = cv2.filter2D(grayimg, cv2.CV_8UC3, gKernel180)

#cv2.drawContours(img, contours, -1, (0,0,255), cv2.FILLED)
#cv2.imshow('what is happening', img)
#dst = cv2.dilate(dst,None)

#need to combine these features to get a shape of the face.
#Then do ocr to find the number on each visible sides to infer? 

#make sure that the edges image is in bgr so it can take the dist thing.
#edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) #change edge detection to something that will just do the right thng
#thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR) #flip this first
#comb = edges + thresh
#combine the edges and corners
#comb[dst>0.01*dst.max()]=[0,0,255]

#flip di
#edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
threshInv = cv2.bitwise_not(thresh)
h, w = thresh.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(thresh, mask, (0,0), 0) 
obj = thresh | threshInv

maskEdge = cv2.copyMakeBorder(edges, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
cv2.floodFill(obj, maskEdge, (0,0), 255) # to eliminate the shadow look at mask and determine how it can fill the region up to the edge

#conntected objs?
#cont = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
#cv2.imshow('1', cont)
#uselesscv2.imshow('2', obj)
#cv2.imshow('5', textureImg180)
##cv2.imshow('', comp)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()




