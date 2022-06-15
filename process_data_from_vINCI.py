"""
Use this tiny teensy program to transform those nifty .json (.txt) files to .csv.
Everything is better with .csv files
Also this holds functions for the 3 types of data
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

all_shoe_data = []
all_watch_data = []
all_survey_data = []

### ---------------------------------------------------------------------------
### -------------------------------- Functions --------------------------------
### ---------------------------------------------------------------------------

# 1. getNumberOfStepsAndStepActivityAsColumns
# For shoe data, the 'data' column per se looks like this:
#       {"step_counter": 0, "step_activity": 2}
# It's a single string. So we need to split that into two columns to be added
# into our data matrix

def getNumberOfStepsAndStepActivityAsColumns(column):
    
    size = column.size
    no_steps = np.zeros(size)
    step_activity = np.zeros(size)
    
    for column_index in range(size):
    
        # number of steps starts at string index 17... 
        i = 17
        number_of_steps = column[column_index][i]
        i += 1
        
        while column[column_index][i] != ',':
            number_of_steps += column[column_index][i]
            i += 1
            
        no_steps[column_index] = number_of_steps
        
        
        # ... and 18 more for activity
        i += 18
        step_activ = column[column_index][i]
        i += 1
        
        while column[column_index][i] != '}':
            step_activ += column[column_index][i]
            i += 1
        
        step_activity[column_index] = step_activ
    
    return no_steps, step_activity  


# 2. getTimestampsDifferenceAsInt
# For survey data, there are two columns with timestamps, one for the start of 
# the survey, and the other for the end. Getting this difference and saving it
# as a column of integers could help the neural network. Function has a choice
# of whether to convert to seconds (s) or miliseconds (ms) (param difference_type)

def getTimestampsDifferenceAsInt(column_begin, column_end, difference_type):
    
    size = column_begin.size
    duration = np.zeros(size)
    
    for column_index in range(size):
        
        # first we get rid of the 'Z' in the end
        if column_end[column_index][23] == 'Z':
            column_end[column_index] = column_end[column_index][0:23]
        if column_begin[column_index][23] == 'Z':
            column_begin[column_index] = column_begin[column_index][0:23]
            
        # then we create the difference
        difference = (np.datetime64(column_end[column_index], difference_type) - 
                      np.datetime64(column_begin[column_index], difference_type)).astype(int)
        duration[column_index] = difference
    
    return duration


### ---------------------------------------------------------------------------
### -------------------------------- Smart way --------------------------------
### ---------------------------------------------------------------------------

shoe_ids = ["5206", "5651", "7051", "11597"]
# shoe_ids = ["5206"]
watch_ids = ['3151', '4552', '4553']
# watch_ids = ['3151']
survey_ids = ['3951', '11551', '11574', '11575', '11576', '11579', '11589', '11598', '12003', '12005', '12207']

shoe_file_path = "./vinci_data/shoe/"
watch_file_path = "./vinci_data/watch/"
survey_file_path = "./vinci_data/survey/"

shoe_file_name_prefix = "shoe_"
watch_file_name_prefix = "watchdeviceId_"
survey_file_name_prefix = "survey_device_Id_"

### -------------------------------- Get Data ---------------------------------

# Get shoe data
for i in range(len(shoe_ids)):
    string = shoe_file_path + shoe_file_name_prefix + shoe_ids[i]
    temp_data_panda = pd.read_json(string + '.txt')    
    if os.path.exists(string + '.csv') == False:
        temp_data_panda.to_csv(string + '.csv')
    
    temp_data = np.array(temp_data_panda)
    column_to_be_split = temp_data[:,1]
    no_steps, step_activity = getNumberOfStepsAndStepActivityAsColumns(column_to_be_split)
    
    # remove the 'data' column and replace it with the two resulting columns from above
    temp_data_shortened = np.delete(temp_data, 1, 1)
    temp_data = np.insert(temp_data_shortened, 1, no_steps, 1)
    temp_data = np.insert(temp_data, 2, step_activity, 1)  
    # temp_data[temp_data[:, 3].argsort()]
    
    temp_data_list = temp_data.tolist()
    all_shoe_data.append(temp_data_list)
    
    
# Get watch data
for i in range(len(watch_ids)):
    string = watch_file_path + watch_file_name_prefix + watch_ids[i]
    # https://stackoverflow.com/questions/6475328/how-can-i-read-large-text-files-line-by-line-without-loading-it-into-memory
    
    # count = 0
    # with open(string + '.txt') as infile:
    # for line in infile:
    #     count += 1
        
    # for line in open(string + '.txt'):
    #     count += 1
        
    # print(count)
        
    file1 = open(string + '.txt', 'r')
    Lines = file1.readlines()
    temp_data_list = np.empty((len(Lines) - 2, 4), dtype=object)
    
    for j in range(len(Lines)):
        if j >= 2:
            temp = Lines[j].split()
            temp_data_list[j - 2][0] = temp[0]
            temp_data_list[j - 2][1] = temp[2]
            temp_data_list[j - 2][2] = temp[4]
            temp_data_list[j - 2][3] = temp[6]
            
            
            data_cell = temp[2].split("\"")
            
    
    temp_data = np.array(temp_data_list)    
    temp_data_panda = pd.DataFrame(temp_data, columns = ['id','data','jhi_timestamp', 'device_id'])
    if os.path.exists(string + '.csv') == False:
        temp_data_panda.to_csv(string + '.csv')
    
    all_watch_data.append(temp_data_list)
    
    
# # Get survey data
# for i in range(len(survey_ids)):
#     string = survey_file_path + survey_file_name_prefix + survey_ids[i]
#     temp_data_panda = pd.read_json(string + '.txt')
#     if os.path.exists(string + '.csv') == False:
#         temp_data_panda.to_csv(string + '.csv')
    
#     temp_data = np.array(temp_data_panda)  
#     difference_column = getTimestampsDifferenceAsInt(temp_data[:,5], temp_data[:,6], 'ms')
    
#     # remove de timestamps columns and replace them with the difference column from above
#     temp_data_shortened = np.delete(temp_data, 6, 1)
#     temp_data_shortened = np.delete(temp_data_shortened, 5, 1)
#     temp_data = np.insert(temp_data_shortened, 5, difference_column, 1)    
    
#     temp_data_list = temp_data.tolist()
#     all_survey_data.append(temp_data_list)
    
#     plt.figure(1)
#     plt.plot(temp_data[:,5])
#     plt.show()
