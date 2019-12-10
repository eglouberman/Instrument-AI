import tensorflow.keras
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import librosa.display
import re
import ast
import os
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

def generate_file_names(file):
    df = pd.read_csv(file)
    #limit the instrument source of the dataframe
    df2 = df[df["instrument_source_str"] == "acoustic"]
    file_names =[]
    for col in df2["instrument_family_str"].value_counts().index:
        if (col!= "bass" and col != "organ"):
            df3 = df2[df2["instrument_family_str"] == col].reset_index()
            arr = np.arange(len(df3))
            np.random.shuffle(arr)
            for j in arr[:3000]:
                file_names.append(df3["Unnamed: 0.1"][j])
    # file_names contains file list of 3000 samples from each instrument
    return file_names


def compute_mel_spect(file):
    data, sr = librosa.load(file)
    n_fft = 2048
    n_mels = 128
    hop_length = 512
    S = librosa.feature.melspectrogram(data, sr=sr, n_fft=n_fft, 
                                   hop_length=hop_length, 
                                   n_mels=n_mels)
    min_nonzero = np.min(S[np.nonzero(S)])
    S[S == 0] = min_nonzero
#     plt.plot(S)
    
    #calculate mfccs
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    mfccsscaled = np.mean(mfccs.T,axis=0)

    return S, mfccs


def plot_confusion_matrix(y_true, y_pred, accuracy, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix, \nAvg testing accuracy: ' + str(accuracy)
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return ax

def get_trains_tests(col, df):
    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(col.tolist())
    y = np.array(df.instrument.tolist())

    # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y)) 

    # split the dataset 
    from sklearn.model_selection import train_test_split 

    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)
    print(yy.shape)
    return x_train, x_test, y_train, y_test,le