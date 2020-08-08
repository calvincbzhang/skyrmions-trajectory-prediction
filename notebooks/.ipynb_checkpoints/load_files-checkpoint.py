import os
import numpy as np

frames = []

directory = 'Rec_EDGE_300K_1L_50MA.out'
# for every file
for filename in os.listdir(directory):
  # if ovf
  if filename.endswith(".ovf"):
    coordinates = None
    print("*** Working on file %s ***" %(os.path.join(directory, filename)))

    with open(os.path.join(directory, filename)) as file:
      lines = file.readlines()
    
    # for each line
    for l in lines:
      if not l.startswith('#'):
        point = l.split()
        point = list(map(float, point))
        if (abs(point[0]) < 0.1) or (abs(point[1]) < 0.1) or (abs(point[2]) < 0.1):
          continue
        else:
          coordinates = np.concatenate((coordinates, [point]), axis=0) if coordinates is not None else [point]

    frames.append(list(coordinates))
    print(np.shape(frames))