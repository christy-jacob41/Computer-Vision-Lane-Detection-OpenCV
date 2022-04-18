import cv2
import numpy as np
import sys

if(len(sys.argv) != 3) :
    print(sys.argv[0], "takes 2 arguments. Not ", len(sys.argv)-1)
    sys.exit()

name_input = sys.argv[1]
name_output = sys.argv[2]

image_input = cv2.imread(name_input, cv2.IMREAD_UNCHANGED);
if(image_input is None) :
    print(sys.argv[0], "Failed to read image from ", name_input)
    sys.exit()
cv2.imshow('original image', image_input);

rank = len(image_input.shape)
if(rank == 2) :
    gray_image = image_input
elif(rank == 3) :
    gray_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
else :
    print(sys.argv[0], "Can't handle unusual image ", name_input)
    sys.exit()

cv2.imshow('gray image', gray_image);
rows, cols = gray_image.shape
image_output = np.zeros([rows, cols], dtype=np.uint8)

# this is slow but we are not concerned with speed here
for i in range(0, rows) :
    for j in range(0, cols) :
        image_output[i,j] = gray_image[i,j]

cv2.imshow('output image', image_output);
cv2.imwrite(name_output, image_output);

# wait for key to exit

cv2.waitKey(0)
cv2.destroyAllWindows()

