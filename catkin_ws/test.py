import cv2
import numpy as np
import math

def findCentre(image, lower, upper, color, camera):
		mask = cv2.inRange(image, lower, upper)
		cv2.imwrite(color+"_"+str(camera)+".png", mask)
		M = cv2.moments(mask)
		cx = int(M['m10'] / M['m00'])
		cy = int(M['m01'] / M['m00'])
		return np.array([cx, cy])


def findYellowCentre(image, camera):
		return findCentre(image, (0, 80, 80), (30, 255, 255), "yellow", camera)

def findBlueCentre(image, camera):
		return findCentre(image, (90, 0, 0), (255, 70, 70), "blue", camera)

def findGreenCentre(image, camera):
		return findCentre(image, (0, 60, 0), (50, 255, 50), "green", camera)

def findRedCentre(image, camera):
		return findCentre(image, (0, 0, 40), (30, 30, 255), "red", camera)

def pixel2meter(image):
		circle0Pos = findYellowCentre(image,1)
		circle1Pos = findBlueCentre(image, 1)
		circle2Pos = findGreenCentre(image, 1)
		# find the distance between two circles
		dist0 = np.sum((circle0Pos-circle1Pos)**2)
		dist1 = np.sum((circle1Pos-circle2Pos) ** 2)

		metres0 = 2.5/np.sqrt(dist0)
		# print(metres0)
		metres1 = 3.5/np.sqrt(dist1)
		# print(metres1)

		# return 3.5 / np.sqrt(dist1)
		return metres0


# Calculate the relevant joint angles from the image
# def detect_joint_angles(image, camera):
# 		a = pixel2meter(image)
# 		# Obtain the centre of each coloured blob
# 		center = a * findYellowCentre(image)
# 		circle1Pos = a * findBlueCentre(image)
# 		circle2Pos = a * findGreenCentre(image)
# 		circle3Pos = a * findRedCentre(image)
#
# 		if camera == 1:
# 				center[0] = center[0] * -1
# 				circle1Pos[0] = circle1Pos[0] * -1
# 				circle2Pos[0] = circle2Pos[0] * -1
# 				circle3Pos[0] = circle3Pos[0] * -1
#
# 		# Solve using trigonometry
# 		ja1 = np.arctan2(center[0] - circle1Pos[0], center[1] - circle1Pos[1])
# 		ja2 = np.arctan2(circle1Pos[0] - circle2Pos[0], circle1Pos[1] - circle2Pos[1]) - ja1
# 		ja3 = np.arctan2(circle2Pos[0] - circle3Pos[0], circle2Pos[1] - circle3Pos[1]) - ja2 - ja1
# 		return np.array([ja1, ja2, ja3])

def find_joint_angles(image, camera):
		a = pixel2meter(image)

		# yellow_centre = a * findYellowCentre(image)
		# blue_centre = a * findBlueCentre(image)
		# green_centre = a * findGreenCentre(image)
		# red_centre = a * findRedCentre(image)
		yellow_centre = findYellowCentre(image)
		blue_centre = findBlueCentre(image)
		green_centre = findGreenCentre(image)
		red_centre = findRedCentre(image)

		angle_one = np.arctan2(yellow_centre[0]-blue_centre[0], yellow_centre[1]-blue_centre[1])
		angle_two = np.arctan2(blue_centre[0]-green_centre[0], blue_centre[1]-green_centre[1]) - angle_one
		angle_three = np.arctan2(green_centre[0]-red_centre[0], green_centre[1]-red_centre[1]) - angle_two - angle_one

		# print(yellow_centre)
		# print(blue_centre)
		# print(green_centre)
		# print(red_centre)

		pic_yellow_centre = cv2.circle(image, (yellow_centre[0], yellow_centre[1]), radius=1, color=(0, 0, 255), thickness=-1)
		pic_blue_centre = cv2.circle(pic_yellow_centre, (blue_centre[0], blue_centre[1]), radius=1, color=(0, 0, 255), thickness=-1)
		pic_green_centre = cv2.circle(pic_blue_centre, (green_centre[0], green_centre[1]), radius=1, color=(0, 0, 255), thickness=-1)
		pic_red_centre = cv2.circle(pic_green_centre, (red_centre[0], red_centre[1]), radius=1, color=(0, 0, 255), thickness=-1)
		cv2.imwrite('angle_check_2.png', pic_red_centre)

		return np.array([angle_one, angle_two, angle_three])


# def find_sphere_centre(self, img):
# 		choice = 'cv2.TM_CCOEFF'
# 		method = eval(choice)
# 		template = self.sphere_img
# 		w, h = template.shape[::-1]
#
# 		res = cv2.matchTemplate(img, template, method)
# 		min_cal, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# 		if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
# 				top_left = min_loc
# 		else:
# 				top_left = max_loc
# 		bottom_right = (top_left[0] + w, top_left[1] + h)
#
# 		cv2.rectangle(img, top_left, bottom_right, 0, 2)
#
# 		# plt.subplot(121), plt.imshow(res, cmap='gray')
# 		# plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
# 		# plt.subplot(122), plt.imshow(img, cmap='gray')
# 		# plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
# 		# plt.suptitle(choice)
# 		# plt.show()
#
# 		centre_x = int((bottom_right[0] - top_left[0]) / 2) + top_left[0]
# 		centre_y = int((bottom_right[1] - top_left[1]) / 2) + top_left[1]
# 		centre = np.array([centre_x, centre_y])
# 		return centre

def getAngle2D(coord1, coord2):
	num = coord1[0]*coord2[0] + coord1[1]*coord2[1]

	denom = math.sqrt(coord1[0]**2 + coord1[1]**2)*math.sqrt(coord2[0]**2 + coord2[1]**2)

	angle = math.acos(num/denom)

	return angle

def getAngle3D(coord1, coord2):

	num = coord1[0]*coord2[0] + coord1[1]*coord2[1] + coord1[2]*coord2[2]
	denom = math.sqrt((coord1[0]**2 + coord1[1]**2 + coord1[2]**2))*math.sqrt((coord2[0]**2 + coord2[1]**2 + coord2[2]**2))

	angle = math.acos(num/denom)

	return angle

def getDifference(centre1, centre2):
	x_diff = centre1[0] - centre2[0]
	y_diff = centre1[1] - centre2[1]

	return [x_diff,y_diff]

def getVector(c1,c2):
	return [c2[0]-c1[0],c2[1]-c1[1],c2[2]-c1[2]]

def getAngles(cam1_image,cam2_image,conversion):

	cam1_yellow = conversion * findYellowCentre(cam1_image, 1)
	cam1_blue = conversion * findBlueCentre(cam1_image, 1)
	cam1_green = conversion * findGreenCentre(cam1_image, 1)
	cam1_red = conversion * findRedCentre(cam1_image, 1)

	cam2_yellow = conversion*findYellowCentre(cam2_image, 2)
	cam2_blue = conversion*findBlueCentre(cam2_image, 2)
	cam2_green = conversion*findGreenCentre(cam2_image, 2)
	cam2_red = conversion*findRedCentre(cam2_image, 2)

	#distance in xz plane from yellow centre

	[yb_y,yb_z1] = getDifference(cam1_yellow,cam1_blue)
	[yg_y,yg_z1] = getDifference(cam1_yellow,cam1_green)
	[yr_y,yr_z1] = getDifference(cam1_yellow,cam1_red)



	#get distance in yz plane from yellow centre

	[yb_x,yb_z2] = getDifference(cam2_yellow,cam2_blue)
	[yg_x,yg_z2] = getDifference(cam2_yellow,cam2_green)
	[yr_x,yr_z2] = getDifference(cam2_yellow,cam2_red)

	# print(yb_z1)
	# print(yb_z2)
	# print(yg_z1)
	# print(yg_z2)
	# print(yr_z1)
	# print(yr_z2)
	yb_z = (yb_z1 + yb_z2) / 2
	yg_z = (yg_z1 + yg_z2) / 2
	yr_z = (yr_z1 + yr_z2) / 2

	yellow_xyz = [0,0,0]
	blue_xyz = [yb_x,yb_y,yb_z]
	green_xyz = [yg_x,yg_y,yg_z]
	red_xyz = [yr_x,yr_y,yr_z]

	print(yellow_xyz)
	print(blue_xyz)

	z_vector = [0,0,1]

	blue2green = getVector(blue_xyz,green_xyz)
	# print(blue2green)

	yellow2blue = getVector(yellow_xyz,blue_xyz)
	print(yellow2blue)

	drift3 = (math.pi/2) - np.arctan2(yellow2blue[2],yellow2blue[0])
	drift2 = (math.pi/2) - np.arctan2(yellow2blue[2],yellow2blue[1])
	print(drift2)
	print(drift3)


	#
	# bg_xz = [blue2green[0],0,blue2green[2]]
	# bg_yz = [0,blue2green[1],blue2green[2]]

	# angle2 = (getAngle3D(z_vector,bg_xz) - drift2)
	# angle3 = -getAngle3D(z_vector,bg_yz) - drift3

	angle2 = (math.pi/2) - np.arctan2(blue2green[2],blue2green[1]) - drift2
	angle3 = -((math.pi/2) - np.arctan2(blue2green[2],blue2green[0])) - drift3


	print(angle2)
	print(angle3)

	# bxz = [yb_x,yb_z]
	# gxz = [yg_x,yg_z]
	#
	# byz = [yb_y,yb_z]
	# gyz = [yg_y,yg_z]
	#
	# print(bxz)
	# print(gxz)
	#
	# print()
	# print(byz)
	# print(gyz)
	#
	#
	#
	# drift = getAngle2D([z_vector[1],z_vector[2]],byz)
	# print(drift)
	#
	# angle2 = getAngle2D(byz ,gyz)
	# print(angle2)
	#
	# angle3 = getAngle2D(bxz , gxz)
	# print(angle3)
	#
	#
	# joint2 = ((math.pi/2) - angle2) - drift
	# joint3 = ((math.pi/2) - angle3) - drift
	#
	# print(joint2)
	# print(joint3)


	return


def main():

		# print("Angle 2 and 3:")
		cam2_image = cv2.imread("testPhotos/angle2anf3/camera1_angle2move1_angle3move-0.5.png")
		cam1_image = cv2.imread("testPhotos/angle2anf3/camera2_angle2move1_angle3move-0.5.png")

		# cam2_image = cv2.imread("testPhotos/angle2/camera1_angle2move1_angle3move0.png")
		# cam1_image = cv2.imread("testPhotos/angle2/camera2_angle2move1_angle3move0.png")

		# cam2_image = cv2.imread("testPhotos/angle3/camera1_angle2move0_angle3move-0.5.png")
		# cam1_image = cv2.imread("testPhotos/angle3/camera2_angle2move0_angle3move-0.5.png")

		a = pixel2meter(cam2_image)#get straight pic

		getAngles(cam1_image,cam2_image,a)

		# print(a)

		cam1_yellow = a * findYellowCentre(cam1_image, 1)
		cam1_blue = a * findBlueCentre(cam1_image, 1)
		cam1_green = a * findGreenCentre(cam1_image, 1)
		cam1_red = a * findRedCentre(cam1_image, 1)


		cam2_yellow = a*findYellowCentre(cam2_image, 2)
		cam2_blue = a*findBlueCentre(cam2_image, 2)
		cam2_green = a*findGreenCentre(cam2_image, 2)
		cam2_red = a*findRedCentre(cam2_image, 2)

		joint1 = np.arctan2(cam1_yellow[0]-cam1_blue[0], cam1_yellow[1]-cam1_blue[1])
		joint2 = np.arctan2(cam1_blue[0]-cam1_green[0], cam1_blue[1]-cam1_green[1]) - joint1
		joint3 = np.arctan2(cam2_blue[0]-cam2_green[0], cam2_blue[1]-cam2_green[1]) - joint1
		joint4 = np.arctan2(cam1_green[0]-cam1_red[0], cam1_green[1]-cam1_red[1]) - joint3 - joint2 - joint1

		# print("Joints:")
		# print(joint1)
		# print(joint2)
		# print(joint3)
		# print(joint4)

		alteredjoint1 = 0
		alteredjoint2 = joint2 + joint1
		alteredjoint3 = joint3 + joint1
		alteredjoint4 = joint4 + joint1 + joint1 + joint1

		# print("Altered Joints:")
		# print(alteredjoint1)
		# print(alteredjoint2)
		# print(alteredjoint3)
		# print(alteredjoint4)





if __name__ == "__main__":
		main()
