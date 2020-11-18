import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('image_copy_img1.png', 1)
mask = cv2.inRange(image, (0, 47, 97), (12, 78, 148))
kernel = np.ones((5, 5), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=3)
cv2.imwrite('b&w_image.png', mask)

M = cv2.moments(mask)
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
print(np.array([cx, cy]))

shape1 = mask[0:800,
              0:cx].copy()

shape2 = mask[0:800,
              cx:800].copy()

cv2.imwrite('shape1.png', shape1)
cv2.imwrite('shape2.png', shape2)

shape1_M = cv2.moments(shape1)
shape2_M = cv2.moments(shape2)

shape1_cx = int(shape1_M['m10'] / shape1_M['m00'])
shape1_cy = int(shape1_M['m01'] / shape1_M['m00'])
shape2_cx = int(shape2_M['m10'] / shape2_M['m00'])
shape2_cy = int(shape2_M['m01'] / shape2_M['m00'])

shape1_cropped = shape1[shape1_cy - 30:shape1_cy + 30,
                        shape1_cx - 30:shape1_cx + 30].copy()
shape2_cropped = shape2[shape2_cy - 30:shape2_cy + 30,
                        shape2_cx - 30:shape2_cx + 30].copy()

cv2.imwrite('shape1_cropped.png', shape1_cropped)
cv2.imwrite('shape2_cropped.png', shape2_cropped)

choice = 'cv2.TM_CCOEFF'
method = eval(choice)

template = cv2.imread('sphere_template.PNG', 0)
img = cv2.imread('image_copy_img1.png', 0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img, template, method)
min_cal, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
  top_left = min_loc
else:
  top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img, top_left, bottom_right, 0, 2)

plt.subplot(121), plt.imshow(res, cmap='gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img, cmap='gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle(choice)
plt.show()

centre_x = int((bottom_right[0] - top_left[0])/2) + top_left[0]
centre_y = int((bottom_right[1] - top_left[1])/2) + top_left[1]
print(centre_x, centre_y)
sphere_midpoint = (centre_x, centre_y)

originalImage = cv2.imread('image_copy_img1.png', 1)
cropped = originalImage[centre_y-50:centre_y+50, centre_x-50:centre_x+50].copy()
cv2.imwrite('final.png', cropped)



