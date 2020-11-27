#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import math
import os
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import message_filters


class angle_calculator:

    def __init__(self):

        rospy.init_node('angle_calculation', anonymous=True)
        self.camera1_sub = message_filters.Subscriber("image_topic1",Image)
        self.camera2_sub = message_filters.Subscriber("image_topic2",Image)

        self.angle2_pub = rospy.Publisher("angle2_value",Float64,queue_size=1)
        self.angle3_pub = rospy.Publisher("angle3_value",Float64,queue_size=1)
        self.angle4_pub = rospy.Publisher("angle4_value",Float64,queue_size=1)

        self.spherex_pub = rospy.Publisher("spherex",Float64,queue_size=1)
        self.spherey_pub = rospy.Publisher("spherey",Float64,queue_size=1)
        self.spherez_pub = rospy.Publisher("spherez",Float64,queue_size=1)

        self.time_sync = message_filters.TimeSynchronizer([self.camera1_sub,self.camera2_sub],1)
        self.time_sync.registerCallback(self.callback)

        filepath = os.path.join(sys.path[0],"sphere_template.png")

        self.sphere_img = cv2.imread(filepath,0)

        self.bridge = CvBridge()

        self.lastBluePositionCam1 = [0,2.5]
        self.lastGreenPositionCam1 = [0,6]
        self.lastRedPositionCam1 = [0,9]

        self.lastBluePositionCam2 = [0,2.5]
        self.lastGreenPositionCam2 = [0,6]
        self.lastRedPositionCam2 = [0,9]


    def pixel2meter(self,image,camera):
    	circle0Pos = self.findYellowCentre(image,camera)
    	circle1Pos = self.findBlueCentre(image,camera)
    	dist0 = np.sum((circle0Pos-circle1Pos)**2)
    	metres0 = 2.5/np.sqrt(dist0)

    	return metres0

    def findCentre(self, image, lower, upper, colour, camera):
       mask = cv2.inRange(image, lower, upper)
       kernel = np.ones((5, 5), np.uint8)
       mask = cv2.dilate(mask, kernel, iterations=3)
       cv2.imwrite(colour+"_C1.png", mask)
       M = cv2.moments(mask)

       if(M['m00']==0):
           if colour == "redJoint":
               if camera == 1:
                   cx = self.lastRedPositionCam1[0]
                   cy = self.lastRedPositionCam1[1]
               else:
                   cx = self.lastRedPositionCam2[0]
                   cy = self.lastRedPositionCam2[1]
           elif colour == "greenJoint":
               if camera == 1:
                   cx = self.lastGreenPositionCam1[0]
                   cy = self.lastGreenPositionCam1[1]
               else:
                   cx = self.lastGreenPositionCam2[0]
                   cy = self.lastGreenPositionCam2[1]
           elif colour == "blueJoint":
               if camera == 1:
                   cx = self.lastBluePositionCam1[0]
                   cy = self.lastBluePositionCam1[1]
               else:
                   cx = self.lastBluePosition[0]
                   cy = self.lastBluePosition[1]
           else:
               cx = 0
               cy = 0
       else:
           cx = int(M['m10']/M['m00'])
           cy = int(M['m01']/M['m00'])

       if colour == "redJoint":
           if camera == 1:
               self.lastRedPositionCam1 = np.array([cx, cy])
           else:
               self.lastRedPositionCam2 = np.array([cx, cy])
       elif colour == "greenJoint":
           if camera == 1:
               self.lastGreenPositionCam1 = np.array([cx, cy])
           else:
               self.lastGreenPositionCam2 = np.array([cx, cy])
       elif colour == "blueJoint":
           if camera == 1:
               self.lastBluePositionCam1 = np.array([cx, cy])
           else:
               self.lastBluePositionCam2 = np.array([cx, cy])

       return np.array([cx, cy])

    def findYellowCentre(self, image, camera):
       return self.findCentre(image, (0, 80, 80), (30, 255, 255), "yellowJoint",camera)

    def findBlueCentre(self, image, camera):
       return self.findCentre(image, (90, 0, 0), (255, 70, 70), "blueJoint", camera)

    def findGreenCentre(self, image, camera):
       return self.findCentre(image, (0, 60, 0), (50, 255, 50), "greenJoint", camera)

    def findRedCentre(self, image, camera):
       return self.findCentre(image, (0, 0, 40), (30, 30, 255), "redJoint", camera)

    def getAngle3D(self,coord1, coord2):
    	coord1_u = coord1/np.linalg.norm(coord1)
    	coord2_u = coord2/np.linalg.norm(coord2)


    	dot_prod = np.dot(coord1_u,coord2_u)
    	angle = np.arccos(dot_prod)

    	return angle

    def getDifference(self,centre2, centre1):
    	x_diff = centre1[0] - centre2[0]
    	y_diff = centre1[1] - centre2[1]

    	return [x_diff,y_diff]

    def getVector(self,c1,c2):
    	return [c2[0]-c1[0],c2[1]-c1[1],c2[2]-c1[2]]

    def getVectorLength(self,vector):

    	return math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)


    def getAngles(self,cam1_image,cam2_image,conversion):

        cam1_yellow = conversion * self.findYellowCentre(cam1_image, 1)
        cam1_blue = conversion * self.findBlueCentre(cam1_image, 1)
        cam1_green = conversion * self.findGreenCentre(cam1_image, 1)
        cam1_red = conversion * self.findRedCentre(cam1_image, 1)
    	# print("Camera 1 centres:")
    	# print(cam1_yellow)
    	# print(cam1_blue)
    	# print(cam1_green)
    	# print(cam1_red)
        cam2_yellow = conversion*self.findYellowCentre(cam2_image, 2)
        cam2_blue = conversion*self.findBlueCentre(cam2_image, 2)
        cam2_green = conversion*self.findGreenCentre(cam2_image, 2)
        cam2_red = conversion*self.findRedCentre(cam2_image, 2)


    	# print("Camera 2 centres:")
    	# print(cam2_yellow)
    	# print(cam2_blue)
    	# print(cam2_green)
    	# print(cam2_red)

    	#distance in yz plane from yellow centre
        [yb_y,yb_z1] = self.getDifference(cam1_yellow,cam1_blue)
        [yg_y,yg_z1] = self.getDifference(cam1_yellow,cam1_green)
        [yr_y,yr_z1] = self.getDifference(cam1_yellow,cam1_red)

    	# print("yz plane:")
    	# print("blue : x=" + str(yb_x) + " z=" + str(yb_z1))
    	# print("green : x=" + str(yg_x) + " z=" + str(yg_z1))
    	# print("red : x=" + str(yr_x) + " z=" + str(yr_z1))



    	#get distance in xz plane from yellow centre
        [yb_x,yb_z2] = self.getDifference(cam2_yellow,cam2_blue)
        [yg_x,yg_z2] = self.getDifference(cam2_yellow,cam2_green)
        [yr_x,yr_z2] = self.getDifference(cam2_yellow,cam2_red)

    	# print("xz plane:")
    	# print("blue : x=" + str(yb_y) + " z=" + str(yb_z2))
    	# print("green : x=" + str(yg_y) + " z=" + str(yg_z2))
    	# print("red : x=" + str(yr_y) + " z=" + str(yr_z2))

        yb_z = -(yb_z1 + yb_z2) / 2
        yg_z = -(yg_z1 + yg_z2) / 2
        yr_z = -(yr_z1 + yr_z2) / 2

        yellow_xyz = [0,0,0]
        blue_xyz = [yb_x,yb_y,yb_z]
        green_xyz = [yg_x,yg_y,yg_z]
        red_xyz = [yr_x,yr_y,yr_z]

        yellow2blue = self.getVector(yellow_xyz,blue_xyz)

        drift2 = self.getAngle3D([1,0,0],yellow2blue) - math.pi/2
        drift3 = math.pi/2 - self.getAngle3D([0,1,0],yellow2blue)
        drift4 = self.getAngle3D([0,0,1],yellow2blue)
        # print("drift 4 : " + str(drift4))

        blue2green = self.getVector(blue_xyz,green_xyz)


        anglex = self.getAngle3D([1,0,0],blue2green)
        angley = self.getAngle3D([0,1,0],blue2green)
        anglez = self.getAngle3D([0,0,1],blue2green)

        angle2 = ((angley-drift2) - math.pi/2)
        angle3 = math.pi/2 - (anglex+drift3)

        green2red = self.getVector(green_xyz,red_xyz)

        blue_y = [3.5*math.cos(anglex),0,3.5*math.sin(anglex)]


        angle4 = self.getAngle3D(blue2green,green2red) - drift4



        # angle4 = self.getAngle3D(green2red,blue_y)

        #self.getAngle3D(green2red,blue_y)
        # if(red_xyz[2]>green_xyz[2]+3.0*math.sin(angle4)):
        #     angle4 = -angle4

        # if(red_xyz[2])




        # angle4 = self.getAngle3D([0,0,1],green2red) - self.getAngle3D([0,0,1],blue2green)

        # angle4 = (self.getAngle3D([0,1,0],green2red) - math.pi/2) #- angle2 - drift4



        # print("End-effector coordinates : " + str(red_xyz))

        # print("xyz coordinates")
        # print("yellow : " + str(yellow_xyz))
        # print("blue : " + str(blue_xyz))
        # print("green : " + str(green_xyz))
        # print("red : " + str(red_xyz))

    	# print("yellow2blue length = " + str(getVectorLength(yellow2blue)))
    	# print("blue2green length = " + str(getVectorLength(blue2green)))
    	# print("green2red length = " + str(getVectorLength(green2red)))
    	#
    	# print("drift2 : " + str(drift2))
    	# print("drift3 : " + str(drift3))
    	# print("drift4 : " + str(drift4))
    	#
    	# print("angle from x : " + str(anglex))
    	# print("angle from y : " + str(angley))
    	# print("angle from z : " + str(anglez))
    	#
    	# print("angle2 : " + str(angle2))
    	# print("angle3 : " + str(angle3))
    	# print("angle4 : " + str(angle4))

        return [0,angle2,angle3,angle4]

    def find_sphere_centre(self,img):
        choice = 'cv2.TM_CCOEFF'
        method = eval(choice)
        template = self.sphere_img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        w, h = template.shape[::-1]

        res = cv2.matchTemplate(gray, template, method)
        min_cal, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img, top_left, bottom_right, 0, 2)

        # plt.subplot(121), plt.imshow(res, cmap='gray')
        # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122), plt.imshow(img, cmap='gray')
        # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        # plt.suptitle(choice)
        # plt.show()

        centre_x = int((bottom_right[0] - top_left[0]) / 2) + top_left[0]
        centre_y = int((bottom_right[1] - top_left[1]) / 2) + top_left[1]
        centre = np.array([centre_x, centre_y])
        return centre

    def callback(self, camera1_data, camera2_data):

        cam1_image = self.bridge.imgmsg_to_cv2(camera1_data, "bgr8")
        cam2_image = self.bridge.imgmsg_to_cv2(camera2_data, "bgr8")

        cam1_sphere_centre = self.find_sphere_centre(cam1_image)
        cam2_sphere_centre = self.find_sphere_centre(cam2_image)

        a = self.pixel2meter(cam1_image,1)
        b = self.pixel2meter(cam2_image,2)
        # x = b * cam2_sphere_centre[0]
        # y = a * cam1_sphere_centre[0]
        # z = a * cam1_sphere_centre[1]
        # combinedSphereCentre = np.array([x, y, z])

        # print(combinedSphereCentre)

        yellowJointCentre1 = self.findYellowCentre(cam1_image,1)
        yellowJointCentre2 = self.findYellowCentre(cam2_image,2)

        [s_y,s_z1] = self.getDifference(yellowJointCentre1,cam1_sphere_centre)
        [s_x,s_z2] = self.getDifference(yellowJointCentre2,cam2_sphere_centre)

        x = s_x * b
        y = s_y * b * 5/6
        z = -(a*(s_z1+s_z2)/2) + 0.25



        joint_angles = self.getAngles(cam1_image,cam2_image,a)

        angle1 = Float64()
        angle1.data = 0.0
        angle2 = Float64()
        angle2.data = joint_angles[1]
        angle3 = Float64()
        angle3.data = joint_angles[2]
        angle4 = Float64()
        angle4.data = joint_angles[3]

        sphere_x = Float64()
        sphere_x.data = x
        sphere_y = Float64()
        sphere_y.data = y
        sphere_z = Float64()
        sphere_z.data = z

        try:
            self.angle2_pub.publish(angle2)
            self.angle3_pub.publish(angle3)
            self.angle4_pub.publish(angle4)

            self.spherex_pub.publish(sphere_x)
            self.spherey_pub.publish(sphere_y)
            self.spherez_pub.publish(sphere_z)
        except CvBridgeError as e:
            print(e)

        return

def main(args):

    ac = angle_calculator()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
