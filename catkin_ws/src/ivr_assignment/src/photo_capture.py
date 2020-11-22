#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import message_filters

class photo_capture:

    def __init__(self):

        rospy.init_node('photo_capture', anonymous=True)

        self.camera1_sub = message_filters.Subscriber("/camera2/robot/image_raw", Image)
        self.camera2_sub = message_filters.Subscriber("/camera1/robot/image_raw", Image)

        self.time_sync = message_filters.TimeSynchronizer([self.camera1_sub,self.camera2_sub],10)
        self.time_sync.registerCallback(self.callback)

        self.bridge = CvBridge()

    def callback(self,img1,img2):

        try:
          self.cv_image1 = self.bridge.imgmsg_to_cv2(img1, "bgr8")
          self.cv_image2 = self.bridge.imgmsg_to_cv2(img2, "bgr8")
        except CvBridgeError as e:
          print(e)

        cv2.imwrite("camera1_view.png", self.cv_image1)
        cv2.imwrite("camera2_view.png", self.cv_image2)


def main(args):
    pc = photo_capture()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)
