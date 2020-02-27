filename = 'C:\\Users\\Jacqui\\Documents\\Work\\Assignments\\Fourth Year\\Project\\Code\\dice\\train\\d8\\d8_color000.jpg'
img = cv2.imread(filename)
grayimg = cv2.cvtcolor(img, cv2.COLOR_BGR2GRAY)

#threshold locally
ret, thresh = cv2.threshold(grayimg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

corners = cv2.cornerHarris(grayimg, 5, 3, 0.01)
edges = cv2.Canny(img,140,150)

gKernel = cv2.getGaborKernel((21,21), 9.0, 180, 10.0, 0.5, 0, ktype=cv2.CV_32F)
textureSeg = cv2.filter2D(grayimg, cv2.CV_8UC3, gKernel)

#Ok now to do the classifier.
#Very tempting to just chuck it in an svm.
#Or a decision tree but I don't know which is better or easier to code. 
