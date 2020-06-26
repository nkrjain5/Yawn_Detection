import cv2
import dlib
import numpy
import math
import logging
import sys


PREDICTOR_PATH=r"shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
MOUTH_POINTS = list(range(48, 61))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))

SCALING_FACTOR=0.4
YAWN_FACTOR=0.5
EYE_FACTOR=40


def euclidian_dist(x,y):
	return math.sqrt(((x[0]-y[0])**2)+((x[1]-y[1])**2))

def nothing(x
	) :pass



def draw_landmarks(landmarks,image):
	for idx,pt in enumerate(landmarks):
		cv2.putText(image,str(idx),(pt[0],pt[1]),fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.3,color=(255, 255, 255))
		cv2.circle(image,(pt[0],pt[1]),3,[255,255,21],-1)
	return image



def lip_distance(landmarks):
	try:
		MAR=euclidian_dist(landmarks[61],landmarks[67])+euclidian_dist(landmarks[62],landmarks[66])+euclidian_dist(landmarks[65],landmarks[63])
		MAR=MAR/(3*euclidian_dist(landmarks[60],landmarks[64]))
		return MAR
	except: 
		return 0



def eyes_distance(landmarks):
	EAR_L=euclidian_dist(landmarks[37],landmarks[41])+euclidian_dist(landmarks[38],landmarks[40])
	EAR_L=EAR_L/(2*euclidian_dist(landmarks[36],landmarks[39]))

	EAR_R=euclidian_dist(landmarks[43],landmarks[47])+euclidian_dist(landmarks[44],landmarks[46])
	EAR_R=EAR_R/(2*euclidian_dist(landmarks[42],landmarks[45]))

	return EAR_L,EAR_R



def main():

	logging.basicConfig(filename="Yawn_detect.log",format='%(asctime)s %(message)s',filemode='w')
	logger=logging.getLogger()
	logger.setLevel(logging.DEBUG) 
	cap=cv2.VideoCapture(0)
	cv2.namedWindow('Live',cv2.WINDOW_NORMAL)
	# mask = numpy.zeros((480,640), dtype=numpy.float64)

	while cv2.waitKey(1)!=27:
		ret,image=cap.read()
		if not ret: break
		image=cv2.flip(image,1)
		gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		gray=cv2.resize(gray,None,fx=SCALING_FACTOR,fy=SCALING_FACTOR,interpolation = cv2.INTER_AREA)
		rect=detector(gray,1)
		if len(rect)!=1:
			print('No/Too many face(s) ')
			continue

		landmarks=[[p.x, p.y] for p in predictor(gray, rect[0]).parts()]
		land_marked_img=draw_landmarks(landmarks,gray)


		a,b=eyes_distance(landmarks)
		c=lip_distance(landmarks)

		logger.info("left_eye: "+str(a))
		logger.info("right_eye: "+str(b))
		logger.info("mouth: "+str(c))

		if c >= YAWN_FACTOR and a<0.14 and b <0.1:
			cv2.putText(image,"YAWNING!",(100,100),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(255, 0, 0))
			print('yawn: ',c)
			logger.info("YAWN@ "+str(c))

		if sys.argv[1]==str(1):
			cv2.putText(image,str(a),(50,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 0))
			cv2.putText(image,str(b),(350,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 0))
			cv2.putText(image,str(c),(200,400),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 0))

		if sys.argv[1]==str(1):
			cv2.imshow('Live',land_marked_img)
		cv2.imshow('Live',image)
	cap.release()
	cv2.destroyAllWindows()


if __name__ =='__main__':
	main()