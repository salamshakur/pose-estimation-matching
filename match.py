# imported libraries
import numpy as np
import pandas as pd
import math

# arrays of referenced key frames
references = []

# target video to extract key frames from
target = pd.read_json('target/front1.json')

# references (both frontal and side)
f_address = pd.read_json('7 key frames/front/address.json')
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

# flag to decide whether to use frontal or side references
isFront = True

# use appropriate references (0 - 6 indexed)
if (isFront == True):
    references.append(f_address)
    references.append(f_takeaway)
    references.append(f_backswing)
    references.append(f_downswing)
    references.append(f_impact)
    references.append(f_follow)
else: 
    references.append(s_address)
    references.append(s_takeaway)
    references.append(s_backswing)
    references.append(s_downswing)
    references.append(s_impact)
    references.append(s_follow)

# array of body joint pairs (1st index: corresponding posenet body joint, 2nd index: corresponding posenet body joint)
pairs = [(5,6),(5,7),(6,8),(7,9),(8,10),(11,12),(5,11),(6,12),(11,13),(12,14),(13,15),(14,16)]

# creates an array of frames from target video where each frame holds an array of keypoint values
# NOTE: Can't use convertKeypoints() due to incompatible indexing
frames = []
for frame in target['keyPoints']:
    joints = []
    for i in range(0,17):
        joints.append(np.array([ frame[i]['position']['x'] , frame[i]['position']['y'] ]))
    frames.append(joints)

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

# function to perform matching by outputting the differences
def matching(template_kp, target_kp, angle_deviation=30, size_deviation=1):
  devs = []
  templ_anchor = template_kp[0][1]
  targ_anchor = target_kp[0][1]
  for i in range(len(template_kp)):
    angles = (template_kp[i][0], target_kp[i][0])
    diff_angle = max(angles) - min(angles)
    templ_size = (template_kp[i][1],templ_anchor)
    try:
        templ_size = abs(min(templ_size) / max(templ_size))
    except ZeroDivisionError as error:
        templ_size = 0
    tar_size = (target_kp[i][1], targ_anchor)
    try:
        tar_size = abs(min(tar_size) / max(tar_size))
    except ZeroDivisionError as error:
        tar_size = 0
    if diff_angle > angle_deviation:
      devs.append(i)
    elif max(tar_size,templ_size) - min(tar_size,templ_size) > size_deviation:
      devs.append(i)
  return devs

# function to generate array of referenced keypoints, indexed properly
def convertKeypoints(reference):
    keypoints = []
    for i in range(0,17):
        keypoints.append(np.array([ reference['keyPoints'][i]['position']['x'] , reference['keyPoints'][i]['position']['y'] ]))
    return keypoints

# function to get all references and run them against target keypoints
def getMatches(reference, frames_values):
    matches = []

    joints = convertKeypoints(reference)

    # creates an array of the reference angle and length values 
    values = []
    for pair in pairs:
        values.append(angle_length(joints[pair[0]], joints[pair[1]]))

    # store the differences of angles/lengths from every frame to the reference
    smallest_length = 10000  # number of angles never exceeds this amount
    differences = []
    for i in range(len(frames_values)):
        current_diff = matching(values, frames_values[i])

        if(len(current_diff) < smallest_length):
            smallest_length = len(current_diff)

        differences.append(current_diff)
    
    # find the best matches aka potentials
    for i, diff in enumerate(differences):
        if(len(diff) == smallest_length):
            # print('frame #', i , ': ', diff)
            matches.append(i)
        i = i + 1

    return matches

# get all matches
address_matches   = getMatches(references[0], frames_values)
takeaway_matches  = getMatches(references[1], frames_values)
backswing_matches = getMatches(references[2], frames_values)
downswing_matches = getMatches(references[3], frames_values)
impact_matches    = getMatches(references[4], frames_values)
follow_matches    = getMatches(references[5], frames_values)

# print matches
print('address matches: ',   address_matches)
print('takeaway matches: ',  takeaway_matches)
print('backswing matches: ', backswing_matches)
print('downswing matches: ', downswing_matches)
print('impact matches: ',    impact_matches)
print('follow matches: ',    follow_matches)