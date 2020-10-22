# imported libraries
import numpy as np
import pandas as pd
import cv2 as cv
import math

# for testing, lets try out one reference (backswing)
f_address = pd.read_json('7 key frames/front/address.json')
f_backswing = pd.read_json('7 key frames/front/backswing.json')
f_takeaway = pd.read_json('7 key frames/front/takeaway.json')
f_backswing = pd.read_json('7 key frames/front/backswing.json')
f_downswing = pd.read_json('7 key frames/front/downswing.json')
f_impact = pd.read_json('7 key frames/front/impact.json')
f_follow = pd.read_json('7 key frames/front/follow.json')

s_address = pd.read_json('7 key frames/side/address.json')
s_takeaway = pd.read_json('7 key frames/side/takeaway.json')
s_backswing = pd.read_json('7 key frames/side/backswing.json')
s_downswing = pd.read_json('7 key frames/side/downswing.json')
s_impact = pd.read_json('7 key frames/side/impact.json')
s_follow = pd.read_json('7 key frames/side/follow.json')


# for testing, lets try out a target video
target = pd.read_json('target/front1.json')

# array of body joint pairs
pairs = [(5,6),(5,7),(6,8),(7,9),(8,10),(11,12),(5,11),(6,12),(11,13),(12,14),(13,15),(14,16)]

# flag to decide whether to use frontal or side references ## NOTE: REMEMBER TO USE THIS!
isFront = True

# creates an array of frames where each frame holds an array of keypoint values
frames = []
for frame in target['keyPoints']:
    joints = []
    for i in range(0,17):
        joints.append(np.array([ frame[i]['position']['x'] , frame[i]['position']['y'] ]))
    frames.append(joints)

# creates an array of the front backswing reference keypoints, indexed properly
f_backswing_joints = []
for i in range(0,17):
    f_backswing_joints.append(np.array([ f_backswing['keyPoints'][i]['position']['x'] , f_backswing['keyPoints'][i]['position']['y'] ]))

# function to get angles and lengths
def angle_length(p1, p2):
  angle = math.atan2(- int(p2[0]) + int(p1[0]), int(p2[1]) - int(p1[1])) * 180.0 / np.pi
  length = math.hypot(int(p2[1]) - int(p1[1]), - int(p2[0]) + int(p1[0]))
  return round(angle), round(length)

# creates an array of every frame's angle and length values
frames_values = []
for i in range(len(frames)):
    pair_values = []
    for pair in pairs:
        pair_values.append(angle_length(frames[i][pair[0]], frames[i][pair[1]]))
    frames_values.append(pair_values)

# creates an array of the front backswing reference angle and length values 
f_backswing_values = []
for pair in pairs:
    f_backswing_values.append(angle_length(f_backswing_joints[pair[0]], f_backswing_joints[pair[1]]))

# function to perform matching
def matching(template_kp, target_kp, angle_deviation=30, size_deviation=1):
  devs = []

  # set an anchor size for proportions calculations - distance between shoulders
  templ_anchor = template_kp[0][1]
  targ_anchor = target_kp[0][1]

  # for each body part that we calculated angle and size for
  for i in range(len(template_kp)):

    angles = (template_kp[i][0], target_kp[i][0])
    diff_angle = max(angles) - min(angles)

    templ_size = (template_kp[i][1],templ_anchor)

    ### NOTE: I don't believe there should ever be a 0 here...however I am adding this due to the issue noted below ###
    try:
        templ_size = abs(min(templ_size) / max(templ_size))
    except ZeroDivisionError as error:
        templ_size = 0

    tar_size = (target_kp[i][1], targ_anchor)

    ### NOTE: There shouldn't be a 0 I believe, so why is this happening? Would this affect the final outcome? ###
    try:
        tar_size = abs(min(tar_size) / max(tar_size))
    except ZeroDivisionError as error:
        tar_size = 0

    if diff_angle > angle_deviation:
      devs.append(i)
      #print("{0} has different angle".format(i))

    elif max(tar_size,templ_size) - min(tar_size,templ_size) > size_deviation:
      devs.append(i)
      #print("{0} has different size".format(i))

  return devs


# match every frame to the reference
smallest_length = 10000  # number of angles never exceeds this amount
matches = []
for i in range(len(frames_values)):
    current_match = matching(f_backswing_values, frames_values[i])

    if(len(current_match) < smallest_length):
        smallest_length = len(current_match)

    matches.append(current_match)

# find the best matches aka potentials
for i, match in enumerate(matches):
    if(len(match) == smallest_length):
        print('frame #', i , ': ', match)
    i = i + 1