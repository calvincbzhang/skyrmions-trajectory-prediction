import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm  # for progress bar
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def show_tracked(df, directory=''):
    """
    Shows particles being tracked and taken into consideration in first frame.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the skyrmions to be tracked
    directory : str
        The directory where the data of each simulation is stored
    """

    img = cv2.imread(directory + '/m000000.png')
    radius = 10
    color = (255, 0, 0)
    thickness = 2

    # for each row in the data in the first frame
    for index, row in df[df['frame'] == 0].iterrows():
        img = cv2.circle(img, (int(row['x']), int(row['y'])), radius, color, thickness) 

    plt.figure(figsize=(20, 10))
    plt.title('Skyrmions taken into cosideration')
    plt.imshow(img)
    plt.show()


def get_frames(data):
    """
    Places data in list of frames. Each element of the list if another list
    containing the x and y coordinates of the skyrmions present in the frame.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with columns ['y', 'x', 'frame', 'particle']
    
    Returns
    -------
    list
        a list of lists where each element represents a frame
    """
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


def split(df, test_size = 0.2):
    """
    Splits data into training and testing.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to be split
    test_size: float, optional
        Percentage of data to be used to testing (default is 0.2, i.e. 20%)

    Returns
    -------
    list, list, list, list
        a tuple of 4 lists: train_X, train_y, test_X, test_y
    """

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


def evaluate(X, y, model):
    """
    Predict data given input data and evaluate against ground truth with
    RMSE and R2.

    Parameters
    ----------
    X : numpy.ndarray
        Input data
    y : numpy.ndarray
        Ground truth
    model : object
        Model for prediction
    
    Returns
    -------
    numpy.ndarray
        an array with the predicted values
    """

    # predict with given model
    y_predict = model.predict(X)
    rmse = (np.sqrt(mean_squared_error(y, y_predict)))
    r2 = r2_score(y, y_predict)
    
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    print("\n")
    
    return y_predict


def get_predictions_df(y_predict, ids=[]):
    """
    Places a list of predictions in a dataframe.

    Parameters
    ----------
    y_predict : numpy.ndarray
        Array of predictions
    ids : list, optional
        List of skyrmions predicted (default=[], in this case consider all
        skyrmions)

    Returns
    -------
    pd.dataframe
        a dataframe containing the predicted skyrmions
    """

    predict_df = pd.DataFrame(columns=['x', 'y', 'frame', 'particle'])

    # for each predicted frame
    for i in tqdm(range(len(y_predict)), desc='Prediction to dataframe'):
        
        if not ids:  # if no ids are given, then all particles are considered
            for j in range(0, len(y_predict[i]), 2):
                predict_df = predict_df.append({'x': y_predict[i][j], 'y':y_predict[i][j+1], 'frame': i, 'particle': j//2}, ignore_index=True)
        else:  # if ids are given
            for j, k in zip(ids, range(0, len(y_predict[i]), 2)):
                predict_df = predict_df.append({'x': y_predict[i][k], 'y':y_predict[i][k+1], 'frame': i, 'particle': j}, ignore_index=True)

    return predict_df


def plot_prediction(data, pred_df, train_predict_df=None, ids=[]):
    """
    Plots predictions, given the data for the ground truth, the prediction on
    some set and, optionally, the prediction on the training and the ids.

    Parameters:
    ----------
    data : pd.DataFrame
        The dataframe containing the ground truth
    pred_df : pd.DataFrame
        The dataframe containing the prediction
    train_predict_df : pandas.DataFrame, optional
        The dataframe containing the prediction on the training set if it makes
        for the data to be plotted with the pred_df dataframe (default is None)
    ids : list, optional
        A list containing the ids we want to plot (default is [], which means
        we will plot all the particles)
    """
    plt.figure(figsize=(20, 10))
    plt.grid(True, axis='x')
    plt.xticks(np.arange(0, max(data['x'])+1, 1000.0))
    plt.ylim(0, 200)
    # invert axis so that origin is in top left
    plt.gca().invert_yaxis()
    
    if not ids:
        ids = data.particle.unique().tolist()
    
    for particle in ids:
        if train_predict_df is not None:
            # vertical line diving train and test
            plt.axvline(x=train_predict_df[train_predict_df['particle'] == particle]['x'].iloc[-1], ymin=0, ymax=1, label='training vs testing', color='g')
            x = pd.concat([train_predict_df[train_predict_df['particle'] == particle]['x'], pred_df[pred_df['particle'] == particle]['x']])
            y = pd.concat([train_predict_df[train_predict_df['particle'] == particle]['y'], pred_df[pred_df['particle'] == particle]['y']])
        else:
            x = pred_df[pred_df['particle'] == particle]['x']
            y = pred_df[pred_df['particle'] == particle]['y']

        plt.plot(x, y, label='prediction ' + str(particle), color='tab:blue')
        plt.plot(data[data['particle'] == particle]['x'], data[data['particle'] == particle]['y'], label='ground truth ' + str(particle), color='tab:orange')

    plt.legend()


def frames_to_xy(frames):
    """
    Given a list of frames it transforms it into two lists: X, y.

    Parameters
    ----------
    frames : list
        List of frames to transform
    
    Returns
    -------
    list, list
        two lists, namely a frame list and a next_frame list
    """
    df = pd.DataFrame(columns=['frame', 'next_frame'])
    
    for i in range(1, len(frames)):
        df = df.append({'frame': frames[i-1], 'next_frame': frames[i]}, ignore_index=True)
    
    return df['frame'].tolist(), df['next_frame'].tolist()