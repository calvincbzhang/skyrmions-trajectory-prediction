import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm  # for progress bar
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Show particles that will be taken into consideration on first frame
def show_tracked(df, directory=''):
    img = cv2.imread(directory + '/m000000.png')
    radius = 10
    color = (255, 0, 0)
    thickness = 2

    # for each row in the data in the first frame
    for index, row in df[df['frame'] == 0].iterrows():
        img = cv2.circle(img, (int(row['x']), int(row['y'])), radius, color, thickness) 

    plt.figure(figsize=(20, 10))
    plt.imshow(img)
    plt.show()

# Put data in frames format
def get_frames(data):
    frames = []
    # for each frame
    for f in tqdm(data['frame'].unique(), desc="Getting frames"):
        coordinates = None
        # for each particle
        for p in data[data['frame'] == f]['particle']:
            particle = data[(data['frame'] == f) & (data['particle'] == p)]
            coordinates = np.append(coordinates, [particle['x'].values[0], particle['y'].values[0]]) if coordinates is not None else [particle['x'].values[0], particle['y'].values[0]]
        frames.append(list(coordinates))
    return frames

# Split in training and testing
def split(df, test_size = 0.2):
    # index of separation of training and testing
    last_x_pct = df.index.values[-int(test_size*max(df.index.values))]
    # testing set
    test_df = df[(df.index >= last_x_pct)]
    # training set
    train_df = df[(df.index < last_x_pct)]
    
    first_col = df.columns[0]
    second_col = df.columns[1]
    
    # return train_X, train_y, test_X, test_y
    return train_df[first_col].tolist(), train_df[second_col].tolist(), test_df[first_col].tolist(), test_df[second_col].tolist()

# Print evaluation metrics
def evaluate(X, y, model):
    # predict with given model
    y_predict = model.predict(X)
    rmse = (np.sqrt(mean_squared_error(y, y_predict)))
    r2 = r2_score(y, y_predict)
    
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    print("\n")
    
    return y_predict

# Place predictions in dataframe
def get_predictions_df(y_predict):
    predict_df = pd.DataFrame(columns=['x', 'y', 'frame', 'particle'])

    # for each predicted frame
    for i in tqdm(range(len(y_predict)), desc='Prediction to dataframe'):
        # for each particle
        for j in range(0, len(y_predict[i]), 2):
            predict_df = predict_df.append({'x': y_predict[i][j], 'y':y_predict[i][j+1], 'frame': i, 'particle': j//2}, ignore_index=True)
    
    return predict_df

# Plot prediction
def plot_prediction(train_predict_df, test_predict_df, data, particle=0):
    plt.figure(figsize=(20, 10))
    plt.grid(True, axis='x')
    plt.xticks(np.arange(0, max(data['x'])+1, 1000.0))
    plt.ylim(0, 200)
    # invert axis so that origin is in top left
    plt.gca().invert_yaxis()

    # vertical line diving train and test
    plt.axvline(x=train_predict_df[train_predict_df['particle'] == particle]['x'].iloc[-1], ymin=0, ymax=1, label='training vs testing', color='g')

    x = pd.concat([train_predict_df[train_predict_df['particle'] == particle]['x'], test_predict_df[test_predict_df['particle'] == particle]['x']])
    y = pd.concat([train_predict_df[train_predict_df['particle'] == particle]['y'], test_predict_df[test_predict_df['particle'] == particle]['y']])

    plt.plot(x, y, label='prediction')
    plt.plot(data[data['particle'] == particle]['x'], data[data['particle'] == particle]['y'], label='ground truth')
    plt.legend()