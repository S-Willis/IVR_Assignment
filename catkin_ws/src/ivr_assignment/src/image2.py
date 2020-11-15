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


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera2 to a topic named image_topic2
    self.image_pub2 = rospy.Publisher("image_topic2", Image, queue_size=1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)
    # initialize the bridge between openCV and ROS
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
    cv2.imwrite(colour+"_C2.png", mask)
    M = cv2.moments(mask)
    cx = int(M['m10']/M['m00'])
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
    print(np.array([angle_one, angle_two, angle_three]))
    return np.array([angle_one, angle_two, angle_three])

  # Recieve data, process it, and publish
  def callback2(self,data):
    # Recieve the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    self.find_joint_angles(self.cv_image2)

    # Uncomment if you want to save the image
    cv2.imwrite('image_copy_img2.png', self.cv_image)
    #im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)

    # Publish the results
    try:
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
    except CvBridgeError as e:
      print(e)

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
