import numpy as np
from tqdm import tqdm  # for progress bar

# Put data in frames format
def get_frames(data):
    frames = []
    # iterate through the frames
    for f in tqdm(data['frame'].unique()):
        coordinates = None
        for p in data[data['frame'] == f]['particle']:
            particle = data[(data['frame'] == f) & (data['particle'] == p)]
            coordinates = np.append(coordinates, [particle['x'].values[0], particle['y'].values[0]]) if coordinates is not None else [particle['x'].values[0], particle['y'].values[0]]
        frames.append(list(coordinates))
    return frames