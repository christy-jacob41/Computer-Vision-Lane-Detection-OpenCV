import fractions
from turtle import width
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import argparse
import sys
from os import listdir
from os.path import isfile, join

def detect_lane(frame) :
    result = 0
    return result

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
    [(0, int(height*.6)), (0, height), (width, height), (width, int(height*.8)), (int(width *.7), int(height*.3)), (int(width *.4), int(height*.4))]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

###########################################################################

def runon_image(path) :
    frame = cv2.imread(path)
    height = frame.shape[0]
    width = frame.shape[1] 
    if height > 800:
        frame = cv2.resize(frame, (int(width/4), int(height/4)))

    # Change the codes here
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frameYellow = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #yellow mask
    ly = np.array([0, 90, 150], dtype = "uint8")
    uy = np.array([30, 255, 255], dtype="uint8")
    
    #white mask
    lw = np.array([180, 180, 180], dtype = "uint8")
    uw = np.array([255, 255, 255], dtype="uint8")

    #red mask
    lr = np.array([180, 180, 180], dtype = "uint8")
    ur = np.array([255, 255, 255], dtype="uint8")

    my = cv2.inRange(frameYellow, ly, uy)
    mw = cv2.inRange(frameGray, lw, uw)
    my = cv2.morphologyEx(my, cv2.MORPH_OPEN, kernel=np.ones((3,3),dtype=np.uint8))
    my = cv2.morphologyEx(my, cv2.MORPH_CLOSE, kernel=np.ones((9,9),dtype=np.uint8))

    myw = cv2.bitwise_or(my, mw)

    FrameW = cv2.bitwise_and(frame, frame, mask = mw)
    FrameY = cv2.bitwise_and(frame, frame, mask = my)
    FrameYW = cv2.bitwise_and(frame, frame, mask = myw)

    FrameG = cv2.cvtColor(FrameYW, cv2.COLOR_BGR2GRAY)
    (thresh, FrameBW) = cv2.threshold(FrameG, 50, 255, cv2.THRESH_BINARY)

    mc = cv2.inRange(frame, 0, 255)
    
    # contours, hierarchy = cv2.findContours(FrameYW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     hull = cv2.convexHull(cnt)
    #     cv2.drawContours(mc,[hull],0,(255), 5)
    # mc = cv2.morphologyEx(mc, cv2.MORPH_ERODE, kernel=np.ones((5,5),dtype=np.uint8))
    FrameC = cv2.bitwise_and(frame, frame, mask = mc)

    FrameBW = cv2.GaussianBlur(FrameBW,(9,9),0)
    FrameBW = cv2.GaussianBlur(FrameBW,(9,9),0)
    FrameBW = cv2.GaussianBlur(FrameBW,(9,9),0)


    can = cv2.Canny(FrameBW, 60, 150)
    can = region_of_interest(can)

    rho = 2
    theta = np.pi/180
    thresh = 80
    min_len = 15
    max_gap = 95
    lines = cv2.HoughLinesP(can, rho, theta, thresh, np.array([]), minLineLength=min_len, maxLineGap=max_gap)
    linesFiltered = []

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            a = float((y2-y1)/(x2-x1))
            if not np.isnan(a) or np.isinf(a) or (a == 0):
                if (a > -50) and (a < -0.3):
                    add = True
                    for lineTemp in linesFiltered:
                        xT1,yT1,xT2,yT2=lineTemp.reshape(4)
                        aT = float((yT2-yT1)/(xT2-xT1))

                        if abs(a-aT) < 0.25 or abs(x1-xT1)<15:
                            add = False
                            
                    if add:
                       linesFiltered.append(line)
                if (a > 0.3) and (a < 50):
                    add = True
                    for lineTemp in linesFiltered:
                        xT1,yT1,xT2,yT2=lineTemp.reshape(4)
                        aT = float((yT2-yT1)/(xT2-xT1))

                        if abs(a-aT) < 0.25 or abs(x1-xT1)<15:
                            add = False
                            
                    if add:
                        linesFiltered.append(line)
    
    lines = np.array(linesFiltered)
    #averaged_lines = average_slope_intercept(frame, lines)
    # for lin in averaged_lines:
    #     print(averaged_lines)
    
    num_clusters =0

    if len(lines)>0:
        lines = lines.reshape(lines.shape[0],4)
        scaler = StandardScaler()
        scaler.fit(lines)
        lines = scaler.fit_transform(lines)

        db = DBSCAN(eps=0.5, min_samples=3).fit(lines) #applying DBSCAN Algorithm on our normalized lines
        labels = db.labels_

        lines = scaler.inverse_transform(lines) #getting back our original values

        num_clusters = np.max(labels) + 1

        print(len(lines), "clusters detected")

        

    line_image = display_lines(frame, linesFiltered)
    
    # detections_in_frame = detect_lane(frame)

    #detections_in_frame = num_clusters
    detections_in_frame = len(lines)

    frame = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Change the codes above
    cv2.imshow("one image", frame)
    cv2.waitKey(0)
    return detections_in_frame

def runon_folder(path) :
    files = None
    if(path[-1] != "/"):
        path = path + "/"
        files = [join(path,f) for f in listdir(path) if isfile(join(path,f))]
    all_detections = 0
    for f in files:
        print(f)
        f_detections = runon_image(f)
        all_detections += f_detections
    return all_detections

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', help="requires path")
    args = parser.parse_args()
    folder = args.folder
    if folder is None :
        print("Folder path must be given \n Example: python proj1.py -folder images")
        sys.exit()

    if folder is not None :
        all_detections = runon_folder(folder)
        print("total of ", all_detections, " detections")

    cv2.destroyAllWindows()



