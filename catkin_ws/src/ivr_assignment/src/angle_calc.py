#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import math
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

        self.time_sync = message_filters.TimeSynchronizer([self.camera1_sub,self.camera2_sub],10)
        self.time_sync.registerCallback(self.callback)

        self.sphere_img = cv2.imread("sphere_template.png",0)

        self.bridge = CvBridge()



    def pixels_to_metres(self, image):
       blue_centre = self.findBlueCentre(image)
       green_centre = self.findGreenCentre(image)
       dist = np.sum((blue_centre - green_centre) ** 2)
       return 3 / np.sqrt(dist)

    def findCentre(self, image, lower, upper, colour):
       mask = cv2.inRange(image, lower, upper)
       kernel = np.ones((5, 5), np.uint8)
       mask = cv2.dilate(mask, kernel, iterations=3)
       cv2.imwrite(colour+"_C1.png", mask)
       M = cv2.moments(mask)
       cx = int(M['m10']/M['m00']) #Possibly adjust for when joints are obscured from camera view?
       cy = int(M['m01']/M['m00'])
       return np.array([cx, cy])

    def findYellowCentre(self, image):
       return self.findCentre(image, (0, 80, 80), (30, 255, 255), "yellowJoint")

    def findBlueCentre(self, image):
       return self.findCentre(image, (90, 0, 0), (255, 70, 70), "blueJoint")

    def findGreenCentre(self, image):
       return self.findCentre(image, (0, 60, 0), (50, 255, 50), "greenJoint")

    def findRedCentre(self, image):
       return self.findCentre(image, (0, 0, 40), (30, 30, 255), "redJoint")

    def find_joint_angles(self, image):
       a = self.pixels_to_metres(image)
       yellow_centre = a * self.findYellowCentre(image)
       blue_centre = a * self.findBlueCentre(image)
       green_centre = a * self.findGreenCentre(image)
       red_centre = a * self.findRedCentre(image)


       angle_one = np.arctan2(yellow_centre[0] - blue_centre[0], yellow_centre[1] - blue_centre[1])

       angle_two = np.arctan2(blue_centre[0] - green_centre[0], blue_centre[1] - green_centre[1]) - angle_one

       angle_three = np.arctan2(green_centre[0] - red_centre[0], green_centre[1] - red_centre[1]) - angle_two - angle_one


       return np.array([angle_one, angle_two, angle_three])

    def find_sphere_centre(self,img):
        choice = 'cv2.TM_CCOEFF'
        method = eval(choice)
        template = self.sphere_img
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(img, template, method)
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

        cam1_angles = self.find_joint_angles(self.bridge.imgmsg_to_cv2(camera1_data, "bgr8"))  # needs to be UMat
        cam2_angles = self.find_joint_angles(self.bridge.imgmsg_to_cv2(camera2_data, "bgr8"))  # needs to be UMat

        cam1_sphere_centre = self.find_sphere_centre(self.bridge.imgmsg_to_cv2(camera1_data, "bgr8"))
        cam2_sphere_centre = self.find_sphere_centre(self.bridge.imgmsg_to_cv2(camera2_data, "bgr8"))

        a = self.pixels_to_metres(self.bridge.imgmsg_to_cv2(camera1_data, "bgr8"))
        b = self.pixels_to_metres(self.bridge.imgmsg_to_cv2(camera2_data, "bgr8"))
        x = b * cam2_sphere_centre[0]
        y = a * cam1_sphere_centre[0]
        z = a * cam1_sphere_centre[1]
        combinedSphereCentre = np.array([x, y, z])

        print(combinedSphereCentre)



        angle2 = Float64()
        angle2.data = cam1_angles[1]
        angle3 = Float64()
        angle3.data = cam2_angles[1]

        print(angle2)
        print(angle3)

        try:
            self.angle2_pub.publish(angle2)
            self.angle3_pub.publish(angle3)
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
