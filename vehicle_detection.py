#!/usr/bin/env python

'''
face detection using haar cascades
USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
from firebase import firebase
# local modules
from video import create_capture
from common import clock, draw_str


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "C:/opencv/build/install/etc/haarcascades/cars.xml")
    #nested_fn  = args.get('--nested-cascade', "C:/opencv/build/install/etc/haarcascades/p1.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    #nested = cv2.CascadeClassifier(nested_fn)
    count=0
    i=1
    cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')
    while 1:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        cars = cascade.detectMultiScale(gray, 1.1, 1)
        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        if cascade.empty():
           for (x,y,w,h) in cars:
                count=count+1
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)  
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                count=draw_rects(vis_roi, subrects, (255, 0, 0))
        dt = clock() - t

        #for (i, (x, y, w, h)) in enumerate(rects):
        #   cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #   cv2.putText(vis, "Car #{}".format(i + 1), (x, y - 10),
        #cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('car', vis)

        if cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
    from firebase import firebase
    from firebase.firebase import FirebaseApplication
    #from firebase.firebase import FirebasePut
    count=len(cars)
    print(count)
    data={'cars' : count}
    firebase = firebase.FirebaseApplication('https://sd14cs013.firebaseio.com/')
    firebase.put('https://sd14cs013.firebaseio.com/','count',data)
