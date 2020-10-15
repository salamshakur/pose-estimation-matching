import numpy as np
import pandas as pd
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
data_1 = pd.read_json('backswing_5.json')
keyPoints_1 = data_1['keyPoints']

# json data of target
data_2 = pd.read_json('backswing_6.json')
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