"""
Use this tiny teensy program to transform those nifty .json (.txt) files to .csv.
Everything is better with .csv files
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

### ---------------------------------------------------------------------------
### -------------------------------- Smart way --------------------------------
### ---------------------------------------------------------------------------

shoe_ids = ["5206", "5651", "7051", "11597"]
watch_ids = ['3151', '4552', '4553']
survey_ids = ['3951', '11551', '11574', '11575', '11576', '11579', '11589', '11598', '12003', '12005', '12207']

shoe_file_path = "./vinci_data/shoe/"
watch_file_path = "./vinci_data/watch/"
survey_file_path = "./vinci_data/survey/"

shoe_file_name_prefix = "shoe_"
watch_file_name_prefix = "watchdeviceId_"
survey_file_name_prefix = "survey_device_Id_"

data_shoe_test = pd.read_json('./vinci_data/shoe/shoe_7051.txt')
# data_watch_test = pd.read_json('./vinci_data/watch/watchdeviceID_3151.txt')
# data_survey_test = pd.read_json('./vinci_data/survey/survey_device_Id_11576.txt')

### -------------------------------- Get Data ---------------------------------

# Get shoe data
for i in range(len(shoe_ids)):
    string = shoe_file_path + shoe_file_name_prefix + shoe_ids[i]
    temp_data_panda = pd.read_json(string + '.txt')
    temp_data_panda.to_csv(string + '.csv')
    
    temp_data = np.array(temp_data_panda)
    column_to_be_split = temp_data[:,1]
    no_steps, step_activity = getNumberOfStepsAndStepActivityAsColumns(column_to_be_split)
    
    temp_data_shortened = np.delete(temp_data, 1, 1)
    temp_data = np.insert(temp_data_shortened, 1, no_steps, 1)
    temp_data = np.insert(temp_data, 2, step_activity, 1)    
    
    temp_data_list = temp_data.tolist()
    all_shoe_data.append(temp_data_list)
    
    
     

    
    
# # Get watch data
# for i in len(watch_ids):
#     string = watch_file_path + watch_file_name_prefix + watch_ids(i)
#     temp_data_panda = pd.read_json(string + '.txt')
#     temp_watch = np.array(temp_data_panda)    
#     temp_watch_list = temp_watch.tolist()
#     all_watch_data.add(temp_watch_list)
#     temp_data_panda.to_csv(string + '.csv')
    
# # Get survey data
# for i in len(survey_ids):
#     string = survey_file_path + survey_file_name_prefix + survey_ids(i)
#     temp_data_panda = pd.read_json(string + '.txt')
#     temp_data = np.array(temp_data_panda)    
#     temp_data_list = temp_data.tolist()
#     all_survey_data.add(temp_data_list)
#     temp_data_panda.to_csv(string + '.csv')

### --------- Step by step way because the smart way doesn't work ffs ---------

# data_shoe_5206 = pd.read_json('./vinci_data/shoe/shoe_5206.txt')
# data_shoe_5651 = pd.read_json('./vinci_data/shoe/shoe_5651.txt')
# data_shoe_7051 = pd.read_json('./vinci_data/shoe/shoe_7051.txt')
# data_shoe_11597 = pd.read_json('./vinci_data/shoe/shoe_11597.txt')

