#Trains neural networks and test them after various amounts of training
#Can train them in different gradient environments
#Has a categoric output from neural networks
#Requires script: NN_functions_2023.py

import sys
import numpy as np
import tensorflow
import keras
import keras.layers
import keras.models
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras import optimizers
import keras.backend as K
from datetime import date
from pandas import *
from NN_functions_2023 import test, format_angles_0_1, format_angles_0_1_discrete, format_angles_0_2pi, format_angles_discrete_0_2pi, format_np_array, make_compile_NN_model, make_compile_NN_model_categ

#True or false - to output results to file
file_out = True

# reading CSV file
training_data = read_csv("Training_data.csv")
print(training_data.columns)

test_data = read_csv("Testing_data.csv")
print(test_data.columns)
test_length = len(test_data)

gradient1 = 'A'
gradient2 = 'B3i'

no_runs = 200

training_test_sizes = [0, 100, 250, 500, 1000, 2000, 4000, 8000, 14000]
#the training dataset sizes at which the neural networks are tested

results_dict = {}
for ls in range(len(training_test_sizes)):
    results_dict[training_test_sizes[ls]] = []

#Number of categories into which the circle is divided up
#Number of output neurons of NN
N_cats = 20

correct_for_implementation_angle = True
#Corrects for implementation biases by rotating 'North' (output angle of 0) to a random bearing for each network


for rn in range(no_runs):
    print("Run", rn)

    training_data_rand_order = training_data.sample(frac=1)
    training_length = len(training_data_rand_order)

    if correct_for_implementation_angle == True:
        ind_angle_dev = np.random.uniform(low = 0, high = 2*np.pi, size = 1)
    else:
        ind_angle_dev = 0

    train_angles = training_data_rand_order['Training_TRUE'].tolist()
    train_ang_01_discrete = format_angles_0_1_discrete(train_angles, ind_angle_dev, N_cats)

    #prepares input for neural network
    grid = np.array([[training_data_rand_order[gradient1].tolist()], [training_data_rand_order[gradient2].tolist()]])
    grid = format_np_array(grid, 2, training_length)

    model = make_compile_NN_model_categ(grid, N_cats)

    test_grid = np.array([[test_data[gradient1].tolist()], [test_data[gradient2].tolist()]])
    test_grid = format_np_array(test_grid, 2, test_length)

    test_angles = test_data['TRUE'].tolist()

    test_pred = model.predict(test_grid)
    classes_x=np.argmax(test_pred,axis=1)

    test_pred_0_2pi = format_angles_discrete_0_2pi(classes_x, ind_angle_dev, N_cats)
    test_out_ang = test(test_pred_0_2pi, test_angles)
    results_dict[training_test_sizes[0]].append(test_pred_0_2pi)

    for sz in range(1, len(training_test_sizes)):
        model.fit(
        x = grid[training_test_sizes[sz-1]:training_test_sizes[sz]],
        y = train_ang_01_discrete[training_test_sizes[sz-1]:training_test_sizes[sz]],
        validation_split=0,
        batch_size = 1,
        epochs = 1,
        )

        test_pred = model.predict(test_grid)
        classes_x=np.argmax(test_pred,axis=1)

        test_pred_0_2pi = format_angles_discrete_0_2pi(classes_x, ind_angle_dev, N_cats)
        test_out_ang = test(test_pred_0_2pi, test_angles)
        results_dict[training_test_sizes[sz]].append(test_pred_0_2pi)


for sz1 in range(len(training_test_sizes)):
    dict_in = training_test_sizes[sz1]
    results_dict[dict_in] = format_np_array(results_dict[dict_in], no_runs, test_length)
    if file_out:
        file_name = "NN_categ_2023_out/NN2023outcateg_" + gradient1 + "_" + gradient2 + "_trainsize_" + str(dict_in) + "_n-nets_" + str(no_runs) + "_" + str(date.today()) + ".csv"
        np.savetxt(file_name, results_dict[dict_in], delimiter=",")
