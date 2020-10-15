import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import cv2 as cv
import math

model_path = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
template_path = "3.PNG"
target_path = "9.PNG"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1] #257
width = input_details[0]['shape'][2]  #257

template_image_src = cv.imread(template_path)
template_image = cv.resize(template_image_src, (width, height))
cv.imshow("template", template_image)

target_image_src = cv.imread(target_path)
target_image = cv.resize(target_image_src, (width, height))
cv.imshow("target", target_image)

template_input = np.expand_dims(template_image.copy(), axis=0)
target_input = np.expand_dims(target_image.copy(), axis=0)

floating_model = input_details[0]['dtype'] == np.float32

if floating_model:
  template_input = (np.float32(template_input) - 127.5) / 127.5
  target_input = (np.float32(target_input) - 127.5) / 127.5

interpreter.set_tensor(input_details[0]['index'], template_input)
interpreter.invoke()
template_output_data = interpreter.get_tensor(output_details[0]['index'])
template_offset_data = interpreter.get_tensor(output_details[1]['index'])
template_heatmaps = np.squeeze(template_output_data)
template_offsets = np.squeeze(template_offset_data)
print("template_heatmaps' shape:", template_heatmaps.shape)
print("template_offsets' shape:", template_offsets.shape)

interpreter.set_tensor(input_details[0]['index'], target_input)
interpreter.invoke()
target_output_data = interpreter.get_tensor(output_details[0]['index'])
target_offset_data = interpreter.get_tensor(output_details[1]['index'])
target_heatmaps = np.squeeze(target_output_data)
target_offsets = np.squeeze(target_offset_data)

def parse_output(heatmap_data,offset_data, threshold):

  '''
  Input:
    heatmap_data - heatmaps for an image. Three dimension array
    offset_data - offset vectors for an image. Three dimension array
    threshold - probability threshold for the keypoints. Scalar value
  Output:
    array with coordinates of the keypoints and flags for those that have
    low probability
  '''

  joint_num = heatmap_data.shape[-1]
  pose_kps = np.zeros((joint_num,3), np.uint32)

  for i in range(heatmap_data.shape[-1]):

      joint_heatmap = heatmap_data[...,i]
      max_val_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
      remap_pos = np.array(max_val_pos/8*257,dtype=np.int32)
      pose_kps[i,0] = int(remap_pos[0] + offset_data[max_val_pos[0],max_val_pos[1],i])
      pose_kps[i,1] = int(remap_pos[1] + offset_data[max_val_pos[0],max_val_pos[1],i+joint_num])
      max_prob = np.max(joint_heatmap)

      if max_prob > threshold:
        if pose_kps[i,0] < 257 and pose_kps[i,1] < 257:
          pose_kps[i,2] = 1

  return pose_kps

def draw_kps(show_img,kps, ratio=None):
    for i in range(5,kps.shape[0]):
      if kps[i,2]:
        if isinstance(ratio, tuple):
          cv.circle(show_img,(int(round(kps[i,1]*ratio[1])),int(round(kps[i,0]*ratio[0]))),2,(0,255,255),round(int(1*ratio[1])))
          continue
        cv.circle(show_img,(kps[i,1],kps[i,0]),2,(0,255,255),-1)
    return show_img

template_show = np.squeeze((template_input.copy()*127.5+127.5)/255.0)
template_show = np.array(template_show*255,np.uint8)
template_kps = parse_output(template_heatmaps,template_offsets,0.3)
cv.imshow("kps drawing template", draw_kps(template_show.copy(), template_kps))

target_show = np.squeeze((target_input.copy()*127.5+127.5)/255.0)
target_show = np.array(target_show*255,np.uint8)
target_kps = parse_output(target_heatmaps,target_offsets,0.3)
cv.imshow("kps drawing target", draw_kps(target_show.copy(), target_kps))

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

print(template_kps)
template_values = []
for part in parts_to_compare:
  # print(template_kps[part[0]])
  template_values.append(angle_length(template_kps[part[0]][:2], template_kps[part[1]][:2]))
print(template_values)

target_values = []
for part in parts_to_compare:
  target_values.append(angle_length(target_kps[part[0]][:2], target_kps[part[1]][:2]))
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

draw_deviations(target_show, target_kps, parts_to_compare, deviations)
cv.imshow("show target", target_show)

# connect some of the points 
def join_point(img, kps):

  body_parts = [(5,6),(5,7),(6,8),(7,9),(8,10),(11,12),(5,11),
                      (6,12),(11,13),(12,14),(13,15),(14,16)]

  for part in body_parts:
    cv.line(img, (kps[part[0]][1], kps[part[0]][0]), (kps[part[1]][1], kps[part[1]][0]), 
            color=(255,255,255), lineType=cv.LINE_AA, thickness=3)

template_pose = np.zeros_like(template_show)
join_point(template_pose, template_kps[:, :2])
cv.imshow("template pose", template_pose)

target_pose = np.zeros_like(target_show)
join_point(target_pose, target_kps[:, :2])

# set the new dimensions of the image to reduce the size
buffer = 5 # size of the area around the pose
top_left_y = min(target_kps[5:, 0]) - buffer
top_left_x = min(target_kps[5:, 1]) - buffer
buttom_right_y = max(target_kps[5:, 0]) + buffer
buttom_right_x = max(target_kps[5:, 1]) + buffer

target_pose = target_pose[top_left_y:buttom_right_y, top_left_x:buttom_right_x]
cv.imshow("target pose", target_pose)

template_pose = cv.cvtColor(template_pose, cv.COLOR_BGR2GRAY)
target_pose = cv.cvtColor(target_pose, cv.COLOR_BGR2GRAY)

# the greater the threshold the more exact the pose has to match
threshold = 0.4 # .4 seems to be a perfect threshold

w, h = target_pose.shape[::-1]
res = cv.matchTemplate(target_pose,template_pose, cv.TM_CCOEFF_NORMED)
score = res.max()

print("score:", score)

if score >= threshold:
  print("Match")
else:
  print("Don't match")

def draw_grid(img, grid_size=9, heatmap=None, part=1):

    color = (0,255,255)

    small_size = min(img.shape[0], img.shape[1])
    cell_size = small_size // grid_size
    res = int(small_size % grid_size)

    x = res // 2
    y = res // 2

    while x < img.shape[1]:
      cv.line(img, (x, 0), (x, img.shape[0]), color=color, lineType=cv.LINE_AA, thickness=1)
      x += cell_size

    while y < img.shape[0]:
      cv.line(img, (0, y), (img.shape[1], y), color=color, lineType=cv.LINE_AA, thickness=1)
      y += cell_size

    center_x = res//2
    center_y = res//2 + cell_size//2

    cv.putText(image,str(round(heatmap[0,0,part],1)), (center_x,center_y), cv.FONT_HERSHEY_SIMPLEX, 0.3, color)

    for row_idx, row in enumerate(heatmap[...,part]):

      for col_idx, column in enumerate(row):
        cv.putText(image,str(round(heatmap[col_idx,row_idx,part],1)), (center_x,center_y), cv.FONT_HERSHEY_SIMPLEX, 0.3, color)
        center_y += cell_size
        
      center_x += cell_size
      center_y = res//2 + cell_size//2

# image = cv.imread('1.PNG')
# image = cv.resize(image, (257, 257))
# draw_grid(image, 9, template_heatmaps)
# cv.imshow("image", image)
# cv.imwrite('1.PNG', image)

# cv.waitKey(0)