import numpy as np
import pandas as pd
import cv2 as cv
import math

# Matching keypoints indices in the output of PoseNet
# 0. Left shoulder to right shoulder (5-6)
# 1. Left shoulder to left elbow (5-7)
# 2. Right shoulder to right elbow (6-8)
# 3. Left elbow to left wrist (7-9)
# 4. Right elbow to right wrist (8-10)
# 5. Left hip to right hip (11-12)
# 6. Left shoulder to left hip (5-11)
# 7. Right shoulder to right hip (6-12)
# 8. Left hip to left knee (11-13)
# 9. Right hip to right knee (12-14)
# 10. Left knee to left ankle (13-15)
# 11.  Right knee to right ankle (14-16)
parts_to_compare = [(5,6),(5,7),(6,8),(7,9),(8,10),(11,12),(5,11),(6,12),(11,13),(12,14),(13,15),(14,16)]

# function to get angle btw two points
def angle_length(p1, p2):

  '''
  Input:
    p1 - coordinates of point 1. List
    p2 - coordinates of point 2. List
  Output:
    Tuple containing the angle value between the line formed by two input points 
    and the x-axis as the first element and the length of this line as the second
    element
  '''

  angle = math.atan2(- int(p2[0]) + int(p1[0]), int(p2[1]) - int(p1[1])) * 180.0 / np.pi
  length = math.hypot(int(p2[1]) - int(p1[1]), - int(p2[0]) + int(p1[0]))
  
  return round(angle), round(length)


# MAIN
# json data of template
data_1 = pd.read_json('front_swings/front1/sevenKeyFrames/backswing.json')
keyPoints_1 = data_1['keyPoints']

# json data of target
data_2 = pd.read_json('front_swings/front2/sevenKeyFrames/backswing.json')
keyPoints_2 = data_2['keyPoints']

# template
arr_1 = []
for keyPoint in keyPoints_1:
    arr_1.append(np.array([keyPoint['position']['x'],keyPoint['position']['y']]))

# target
arr_2 = []
for keyPoint in keyPoints_2:
    arr_2.append(np.array([keyPoint['position']['x'],keyPoint['position']['y']]))

template_values = []
for part in parts_to_compare:
    template_values.append(angle_length(arr_1[part[0]][:2], arr_1[part[1]][:2]))
print(template_values)

target_values = []
for part in parts_to_compare:
  target_values.append(angle_length(arr_2[part[0]][:2], arr_2[part[1]][:2]))
print(target_values)

def matching(template_kp, target_kp, angle_deviation=30, size_deviation=1):

  '''Input:
      1. template_kp - list of tuples (for the template image) containng angles 
      between particular body parts and x-axis as first elements and its sizes 
      (distances between corresponding points as second elements)
      2. target_kp - same for the target image
      3. angle_deviation - acceptable angle difference between corresponding 
      body parts in the images
      4. size_deviation - acceptable proportions difference between the images
    Output:
      List of body parts which are deviated
  '''

  devs = []

  # set an anchor size for proportions calculations - distance between shoulders
  templ_anchor = template_kp[0][1]
  targ_anchor = target_kp[0][1]

  # for each body part that we calculated angle and size for
  for i in range(len(template_kp)):

    angles = (template_kp[i][0], target_kp[i][0])
    diff_angle = max(angles) - min(angles)

    templ_size = (template_kp[i][1],templ_anchor)
    templ_size = abs(min(templ_size) / max(templ_size))

    tar_size = (target_kp[i][1], targ_anchor)
    tar_size = abs(min(tar_size) / max(tar_size))

    if diff_angle > angle_deviation:
      devs.append(i)
      print("{0} has different angle".format(i))

    elif max(tar_size,templ_size) - min(tar_size,templ_size) > size_deviation:
      devs.append(i)
      print("{0} has different size".format(i))

  return devs

deviations = matching(template_values, target_values)
print(deviations)

def draw_deviations(img, keypoints, pairs, deviations):

  for i, pair in enumerate(pairs):

    if i in deviations:
      color = (0,0,255)
    else:
      color = (0,255,0)
      
    cv.line(img, (keypoints[pair[0]][1], keypoints[pair[0]][0]), (keypoints[pair[1]][1], keypoints[pair[1]][0]), color=color, lineType=cv.LINE_AA, thickness=1)

blank_image = np.zeros(shape=[257, 257, 3], dtype=np.uint8)
draw_deviations(blank_image, arr_2, parts_to_compare, deviations) # might be wrong
cv.imshow("target image", blank_image)

target_pose = np.zeros_like(blank_image)

# print('testing arr_2: ', arr_2)
# print('testing arr_2[5:, 0]: ', arr_2[5:])
# print('testing arr_2[5:, 1]: ', arr_2[5:][1])

arr_2_xVals = []
arr_2_yVals = []
for i in range(5, len(arr_2)):
  arr_2_xVals.append(arr_2[i][0])
  arr_2_yVals.append(arr_2[i][1])

# set the new dimensions of the image to reduce the size
buffer = 5 # size of the area around the pose
top_left_y = min(arr_2_xVals) - buffer
top_left_x = min(arr_2_yVals) - buffer
buttom_right_y = max(arr_2_xVals) + buffer
buttom_right_x = max(arr_2_yVals) + buffer

## stuck here because i don't have heatmap of target show...

#cv.waitKey(0)