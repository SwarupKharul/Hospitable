from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request
import numpy as np
import math
from django.conf import settings
import pytesseract
from PIL import Image
from PIL import ImageFilter

face_detection_videocam = cv2.CascadeClassifier(os.path.join(
			settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
hand_detection_videocam = cv2.CascadeClassifier(os.path.join(
			settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_hand_detection.xml'))

# define range of skin color in HSV
lower_skin = np.array([0,20,70], dtype=np.uint8)
upper_skin = np.array([20,255,255], dtype=np.uint8)
	
class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
		self.video.release()
	
	def get_frame(self):
		success, image = self.video.read()
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces_detected = face_detection_videocam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
		hand_detected = hand_detection_videocam.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,minSize=(10,10))
		for (x, y, w, h) in faces_detected:
			cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
		for (x, y, w, h) in hand_detected:
			cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)

		
		frame_flip = cv2.flip(image,1)
		kernel = np.ones((3,3),np.uint8)

		#define region of interest
		roi=frame_flip[100:300, 100:300]
           
		cv2.rectangle(frame_flip,(100,100),(300,300),(0,255,0),0)    
		hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
     #extract skin colur imagw  
		mask = cv2.inRange(hsv, lower_skin, upper_skin)
         
    #extrapolate the hand to fill dark spots within
		mask = cv2.dilate(mask,kernel,iterations = 4)
        
    #blur the image
		mask = cv2.GaussianBlur(mask,(5,5),100) 
         
    #find contours
		contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
		try:
    	#find contour of max area(hand)
			cnt = max(contours, key = lambda x: cv2.contourArea(x))

        #approx the contour a little
			epsilon = 0.0005*cv2.arcLength(cnt,True)
			approx= cv2.approxPolyDP(cnt,epsilon,True) 
            
        #make convex hull around hand
			hull = cv2.convexHull(cnt)
            
        #define area of hull and area of hand
			areahull = cv2.contourArea(hull)
			areacnt = cv2.contourArea(cnt)
        
        #find the percentage of area not covered by hand in convex hull
			arearatio=((areahull-areacnt)/areacnt)*100
        
        #find the defects in convex hull with respect to hand
			hull = cv2.convexHull(approx, returnPoints=False)
			defects = cv2.convexityDefects(approx, hull)
            
        # l = no. of defects
			l=0
    
        #code for finding no. of defects due to fingers
			for i in range(defects.shape[0]):
				s,e,f,d = defects[i,0]
				start = tuple(approx[s][0])
				end = tuple(approx[e][0])
				far = tuple(approx[f][0])
				pt= (100,180)
                
                
                # find length of all sides of triangle
				a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
				b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
				c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
				s = (a+b+c)/2
				ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
                
                #distance between point and convex hull
				d=(2*ar)/a
                
                # apply cosine rule here
				angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                
                # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
				if angle <= 90 and d>30:
					l += 1
					cv2.circle(roi, far, 3, [255,0,0], -1)
                
                #draw lines around hand
				cv2.line(roi,start, end, [0,255,0], 2)	
				
			l+=1
			
			#print corresponding gestures which are in their ranges
			font = cv2.FONT_HERSHEY_SIMPLEX
			if faces_detected==():
				cv2.putText(frame_flip,'We are Freshers',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
			elif l==1:
					cv2.putText(frame_flip,'Best of Luck',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)		
			elif l==2:
				cv2.putText(frame_flip,'Cool',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
	
			elif l==3:
				cv2.putText(frame_flip,'Thank You',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
				
			elif l==4:
				cv2.putText(frame_flip,'No Problem',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
			
			elif not hand_detected==():
				cv2.putText(frame_flip,'Hello',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

			

		except:
			pass
			
		ret, jpeg = cv2.imencode('.jpg', frame_flip)
		return jpeg.tobytes()

	def render_features(self):

		success, cap = self.video.read()
		gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
		frame_flip = cv2.flip(cap,1)

		font_scale=1.5
		font=cv2.FONT_HERSHEY_PLAIN
		ret, frame=cap.read()
		
		imgH,imgW,=frame.shape
		x1,y1,w1,h1=0,0,imgH,imgW
		imgchar=pytesseract.image_to_string(frame)
		imgboxes=pytesseract.image_to_boxes(frame)
		for boxes in imgboxes.splitlines():
					boxes=boxes.split(' ')
					x,y,w,h=int(boxes[1]),int(boxes[2]),int(boxes[3]),int(boxes[4])
					cv2.rectangle(frame,(x,imgH-y),(w,imgH-h),(0,0,255),3)
		cv2.putText(frame,imgchar,(x1+int(w1/50),y1+int(h1/50)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
		font=cv2.FONT_HERHEY_SIMPLEX

		cv2.imshow('Text detection',frame)


		ret, jpeg = cv2.imencode('.jpg', frame_flip)
		return jpeg.tobytes()