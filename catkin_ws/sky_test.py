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
		dist0 = np.sum((circle0Pos-circle1Pos)**2)
		metres0 = 2.5/np.sqrt(dist0)

		return metres0


def getAngle2D(coord1, coord2):
	num = coord1[0]*coord2[0] + coord1[1]*coord2[1]

	denom = math.sqrt(coord1[0]**2 + coord1[1]**2)*math.sqrt(coord2[0]**2 + coord2[1]**2)

	angle = math.acos(num/denom)

	return angle

def getAngle3D(coord1, coord2):
	coord1_u = coord1/np.linalg.norm(coord1)
	coord2_u = coord2/np.linalg.norm(coord2)


	dot_prod = np.dot(coord1_u,coord2_u)
	angle = np.arccos(dot_prod)

	return angle

def getDifference(centre2, centre1):
	x_diff = centre1[0] - centre2[0]
	y_diff = centre1[1] - centre2[1]

	return [x_diff,y_diff]

def getVector(c1,c2):
	return [c2[0]-c1[0],c2[1]-c1[1],c2[2]-c1[2]]

def getVectorLength(vector):

	return math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

def getAngles(cam1_image,cam2_image,conversion1):

	cam1_yellow = conversion * findYellowCentre(cam1_image, 1)
	cam1_blue = conversion * findBlueCentre(cam1_image, 1)
	cam1_green = conversion * findGreenCentre(cam1_image, 1)
	cam1_red = conversion * findRedCentre(cam1_image, 1)
	# print("Camera 1 centres:")
	# print(cam1_yellow)
	# print(cam1_blue)
	# print(cam1_green)
	# print(cam1_red)

	cam2_yellow = conversion*findYellowCentre(cam2_image, 2)
	cam2_blue = conversion*findBlueCentre(cam2_image, 2)
	cam2_green = conversion*findGreenCentre(cam2_image, 2)
	cam2_red = conversion*findRedCentre(cam2_image, 2)


	# print("Camera 2 centres:")
	# print(cam2_yellow)
	# print(cam2_blue)
	# print(cam2_green)
	# print(cam2_red)

	#distance in yz plane from yellow centre
	[yb_x,yb_z1] = getDifference(cam1_yellow,cam1_blue)
	[yg_x,yg_z1] = getDifference(cam1_yellow,cam1_green)
	[yr_x,yr_z1] = getDifference(cam1_yellow,cam1_red)

	# print("yz plane:")
	# print("blue : x=" + str(yb_x) + " z=" + str(yb_z1))
	# print("green : x=" + str(yg_x) + " z=" + str(yg_z1))
	# print("red : x=" + str(yr_x) + " z=" + str(yr_z1))



	#get distance in xz plane from yellow centre
	[yb_y,yb_z2] = getDifference(cam2_yellow,cam2_blue)
	[yg_y,yg_z2] = getDifference(cam2_yellow,cam2_green)
	[yr_y,yr_z2] = getDifference(cam2_yellow,cam2_red)

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

	# print("xyz coordinates")
	# print("yellow : " + str(yellow_xyz))
	# print("blue : " + str(blue_xyz))
	# print("green : " + str(green_xyz))
	# print("red : " + str(red_xyz))

	yellow2blue = getVector(yellow_xyz,blue_xyz)

	drift2 = getAngle3D([1,0,0],yellow2blue) - math.pi/2
	drift3 = math.pi/2 - getAngle3D([0,1,0],yellow2blue)
	drift4 = getAngle3D([0,0,1],yellow2blue)



	blue2green = getVector(blue_xyz,green_xyz)

	anglex = getAngle3D([1,0,0],blue2green)
	angley = getAngle3D([0,1,0],blue2green)
	anglez = getAngle3D([0,0,1],blue2green)



	angle2 = (angley-drift2) - math.pi/2
	angle3 = math.pi/2 - (anglex-drift3)

	green2red = getVector(green_xyz,red_xyz)
	angle4 = getAngle3D(green2red,blue2green) - drift4

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


def main():

		# print("Angle 2 and 3:")
		cam1_image = cv2.imread("testPhotos/angle2anf3/camera1_angle2move1_angle3move-0.5.png")
		cam2_image = cv2.imread("testPhotos/angle2anf3/camera2_angle2move1_angle3move-0.5.png")

		# cam1_image = cv2.imread("testPhotos/angle2/camera1_angle2move1_angle3move0.png")
		# cam2_image = cv2.imread("testPhotos/angle2/camera2_angle2move1_angle3move0.png")

		# cam1_image = cv2.imread("testPhotos/angle3/camera1_angle2move0_angle3move-0.5.png")
		# cam2_image = cv2.imread("testPhotos/angle3/camera2_angle2move0_angle3move-0.5.png")

		a = pixel2meter(cam2_image)#get straight pic


		getAngles(cam1_image,cam2_image,a)







if __name__ == "__main__":
		main()
