import numpy as np

left_hand_jpos = np.array([1,2,3,4,5,6,7])
right_hand_jpos = np.array([1,2,3,4,5,6,7])
left_jpos = np.array([1,2,3,4,5,6,7])
right_jpos = np.array([1,2,3,4,5,6,7])

qpos = np.concatenate((left_hand_jpos[3:5], right_hand_jpos[3:5], left_jpos, right_jpos))

print(f"qpos: {qpos}")