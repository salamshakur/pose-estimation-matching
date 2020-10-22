import numpy as np
import pandas as pd
import math

vid_1 = pd.read_json('target/front1.json')
vid_2 = pd.read_json('target/front2.json')

print('front 1 has', len(vid_1), 'frames')
print('front 2 has', len(vid_2), 'frames')