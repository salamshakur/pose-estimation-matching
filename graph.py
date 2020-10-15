from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

bodyJoints = {
    "nose"          :   0,
    "leftEye"       :   1,
    "rightEye"      :   2,
    "leftEar"       :   3,
    "rightEar"      :   4,
    "leftShoulder"  :   5,
    "rightShoulder" :   6,
    "leftElbow"     :   7,
    "rightElbow"    :   8,
    "leftWrist"     :   9,
    "rightWrist"    :   10,
    "leftHip"       :   11,
    "rightHip"      :   12,
    "leftKnee"      :   13,
    "rightKnee"     :   14,
    "leftAnkle"     :   15,
    "rightAnkle"    :   16
}

def graphPose (json, bodyJoint):
    front_data1 = pd.read_json(json)

    yVals = []
    frames = []
    frame = 0

    for keypoints in front_data1['keyPoints']:
        yVals.append(keypoints[bodyJoints[bodyJoint]]['position']['y'])
        frames.append(frame)
        frame = frame + 1

    plt.figure(figsize=(10, 10))
    curve = savgol_filter(yVals, 101, 5) # window size 101, polynomial order 5
    plt.plot(frames, yVals)
    plt.plot(frames, curve, color='red')
    plt.title(json + '-' + bodyJoint)
    plt.xlabel('frame')
    plt.ylabel('y-pos')
    plt.show()


# MAIN #
## CONFIGURE IF YOU WOULD LIKE TO GRAPH OTHER JOINTS AND JSON FILES ##
### NOTE: ALL FOLDERS MUST FOLLOW THE SAME EXACT FILE STRUCTURE ###
joints = ['leftWrist', 'rightWrist']
directories = [r'front_swings', r'back_swings']

for directory in directories:
    for subdir, dirs, files in os.walk(directory):
        if os.path.basename(subdir) == 'allKeyPoints':
            for file in files:
                if file.endswith('.json'):
                    for joint in joints:
                        graphPose(subdir + '/' + file, joint)
        