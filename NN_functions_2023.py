#Functions for Neural_networks_2023.py

from scipy.stats import circmean
from scipy.stats import circstd
import numpy as np
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras import optimizers
import keras.backend as K
import tensorflow


#finds difference between predictions of model and correct answer
#puts angle on 0-360 scale
def test(model_pred, test_answers):
    angdiff = []
    for (each1, each2) in zip(model_pred, test_answers):
        if each2 < 0:
            each2_0_2pi = 2*np.pi + each2
        else:
            each2_0_2pi = each2
        diff = each1 - each2_0_2pi
        if (diff>np.pi):
            diff_out = diff-2*np.pi
        elif (diff<(-np.pi)):
            diff_out = 2*np.pi+diff
        else:
            diff_out = diff
        angdiff.append(diff_out[0]*360/(2*np.pi))
    meanangdiff = circmean(angdiff, low=-180, high=180)
    stdangdiff = circstd(angdiff, low=-180, high=180)
    print("\nCircular mean angle difference:", meanangdiff)
    print("\nCircular std of angle difference:", stdangdiff)
    to_return = np.array([angdiff])
    to_return = np.asarray(to_return, dtype=float)
    to_return = np.reshape(to_return, (1, len(angdiff)))
    to_return = np.ndarray.transpose(to_return)
    return(to_return)


def format_angles_0_1(angles_in, ind_ang_d):
    angs_length = len(angles_in)
    temp_angs = [None]*angs_length
    for i in range(angs_length):
        temp_angs[i] = angles_in[i] + ind_ang_d
        if temp_angs[i] > 2*np.pi:
            temp_angs[i] = temp_angs[i] - 2*np.pi
        elif temp_angs[i] < 0:
            temp_angs[i] = 2*np.pi + temp_angs[i]

    ang_01 = [angles_i/(2*np.pi) for angles_i in temp_angs]
    ang_01 = np.array([ang_01])
    ang_01 = np.asarray(ang_01, dtype=float)
    ang_01 = np.reshape(ang_01, (1, angs_length))
    ang_01 = np.ndarray.transpose(ang_01)
    return(ang_01)

def format_angles_0_1_discrete(angles_in, ind_ang_d, n_cats):
    angs_length = len(angles_in)
    temp_angs = [None]*angs_length
    for i in range(angs_length):
        temp_angs[i] = angles_in[i] + ind_ang_d
        temp_angs[i] = np.round(temp_angs[i]/(np.pi/(n_cats/2)))
        if temp_angs[i] >= n_cats:
            temp_angs[i] = temp_angs[i] - n_cats
        elif temp_angs[i] < 0:
            temp_angs[i] = n_cats + temp_angs[i]

    angs_cat = np.array([temp_angs])
    angs_cat = np.asarray(angs_cat, dtype=float)
    angs_cat = np.reshape(angs_cat, (1, angs_length))
    angs_cat = np.ndarray.transpose(angs_cat)
    train_ang_cat = tensorflow.keras.utils.to_categorical(angs_cat, num_classes=n_cats)
    return(train_ang_cat)


def format_angles_0_2pi(angles_to_format, ind_ang_dv):
    angs_length = len(angles_to_format)
    temp_angs = [None]*angs_length
    for i in range(angs_length):
        temp_angs[i] = angles_to_format[i] - ind_ang_dv/(2*np.pi)
        if temp_angs[i] > 1:
            temp_angs[i] = temp_angs[i] - 1
        elif temp_angs[i] < 0:
            temp_angs[i] = 1 + temp_angs[i]
    ang_0_2pi = [angles_i*(2*np.pi) for angles_i in temp_angs]
    ang_0_2pi = np.array([ang_0_2pi])
    ang_0_2pi = np.asarray(ang_0_2pi, dtype=float)
    ang_0_2pi = np.reshape(ang_0_2pi, (1, angs_length))
    ang_0_2pi = np.ndarray.transpose(ang_0_2pi)
    return(ang_0_2pi)

def format_angles_discrete_0_2pi(angles_to_format, ind_ang_dv, n_cats):
    angs_length = len(angles_to_format)
    temp_angs = [None]*angs_length
    for i in range(angs_length):
        temp_angs[i] = angles_to_format[i]/n_cats
        temp_angs[i] = temp_angs[i] - ind_ang_dv/(2*np.pi)
        if temp_angs[i] > 1:
            temp_angs[i] = temp_angs[i] - 1
        elif temp_angs[i] < 0:
            temp_angs[i] = 1 + temp_angs[i]
    ang_0_2pi = [angles_i*(2*np.pi) for angles_i in temp_angs]
    ang_0_2pi = np.array([ang_0_2pi])
    ang_0_2pi = np.asarray(ang_0_2pi, dtype=float)
    ang_0_2pi = np.reshape(ang_0_2pi, (1, angs_length))
    ang_0_2pi = np.ndarray.transpose(ang_0_2pi)
    return(ang_0_2pi)



def format_np_array(array, size1, size2):
    array1 = np.asarray(array, dtype=float)
    array2 = np.reshape(array1, (size1, size2))
    array3 = np.ndarray.transpose(array2)
    return(array3)

def make_compile_NN_model(input):
    visible = Input(shape=(input.shape[1],))
    hidden1 = Dense(10, activation='relu')(visible)
    hidden2 = Dense(100, activation='relu')(hidden1)
    hidden3 = Dense(200, activation='relu')(hidden2)
    hidden4 = Dense(500, activation='relu')(hidden3)
    hidden5 = Dense(200, activation='relu')(hidden4)
    hidden6 = Dense(100, activation='relu')(hidden5)
    out1 = Dense(1, activation='sigmoid')(hidden6)

    model = Model(inputs=visible, outputs=out1)

    sgd = optimizers.SGD(learning_rate=0.001, decay=0, momentum=0.9, nesterov=True)

    def customLoss(true, predict):
        predict_0_1 = predict % 1
        diff = K.abs(true-predict_0_1)
        new_diff = K.switch(diff>0.5, 1-diff, diff)
        return(K.mean(K.square(new_diff), axis=-1))

    model.compile(
    optimizer = sgd,
    loss = customLoss,
    )
    return(model)

def make_compile_NN_model_categ(input, n_cats):
    visible = Input(shape=(input.shape[1],))
    hidden1 = Dense(10, activation='relu')(visible)
    hidden2 = Dense(100, activation='relu')(hidden1)
    hidden3 = Dense(200, activation='relu')(hidden2)
    hidden4 = Dense(500, activation='relu')(hidden3)
    hidden5 = Dense(200, activation='relu')(hidden4)
    hidden6 = Dense(100, activation='relu')(hidden5)
    out1 = Dense(n_cats, activation='sigmoid')(hidden6)

    model = Model(inputs=visible, outputs=out1)

    model.compile(
    loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']
    )
    return(model)
