# Brendan Hanschke
# EENG 350 - Section A
# February 3, 2025

# Objective:
# take picture with camera and detect the green shape with a mask
# Clean up mask with transformations
# Display contours of shape using mask

from time import sleep
import numpy as np
import cv2


filename = input("Filename: ") + ".jpg"

# Upper and lower limit for detect of green shape (HSV). Between 80 degrees and 160 degrees
uppergreen = np.array([80,255,255]) # 100% saturation and value
lowergreen = np.array([40,51,51]) # 20% saturation and value

#initialize the camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
sleep(1)

#Take picture
ret, image = camera.read()
if not ret:
    print("Couldn't capture image from camera")
else:
    # save image in a file
    print("Saving image " + filename)
    try:
        cv2.imwrite(filename, image)
    except:
        print("Couldn't save " + filename)
        pass

# Convert the image to a shape and create HSV reference image
shapes = cv2.imread(filename)
y,x,c = shapes.shape
shapesHSV = cv2.cvtColor(shapes, cv2.COLOR_RGB2HSV)

# Mask creation and transformation to detect and clean up the green shape detection
mask = cv2.inRange(shapesHSV, lowergreen, uppergreen)
kernel = np.ones((5,5), np.uint8) # Variable size for the kernel. 5 seems the best for now
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
result = cv2.bitwise_and(shapes, shapes, mask=opening)

# Display the mask to check if it is detecting
# cv2.imshow("Result",result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Creation of the contour, which uses the mask to create an outline of the shape
contour_visualize = shapes.copy()
contours,_ = cv2.findContours(opening,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contour_visualize, contours, -1, (0,0,255), 1)

# Display the contour on the original image to check
cv2.imshow("Contours", contour_visualize)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Displays a rectangle around the contour, and calculates the center and area
shapesBox = shapes.copy()
shapeCenters = np.empty((0,2))
shapeAreas = np.empty((0))

for index,cnt in enumerate(contours_green):
    contour_area = cv2.contourArea(cnt)
    if contour_area > 300: # Variable Area for contour
        x, y, w, h = cv2.boundingRect(cnt)
        center = int(x+w/2), int(y+h/2)
        shapeAreas = np.append(shapeAreas, contour_area)
        shapeCenters = np.vstack((shapeCenters, center))
        cv2.rectangle(shapesBox, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(shapesBox, 'Green', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(shapesBox, '+', center, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
#print((shapeCenters, shapeAreas))

# Display the contour rectangle for final product
cv2.imshow("Big Contours",shapesBox)
cv2.waitKey(0)
cv2.destroyAllWindows()
