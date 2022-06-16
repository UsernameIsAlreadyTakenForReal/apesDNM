"""
Use this tiny teensy program to transform those nifty .json (.txt) files to .csv.
Everything is better with .csv files
Also this holds functions for the 3 types of data
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json

all_shoe_data = []
all_watch_data = []
all_survey_data = []

overwrite_csv_files = True

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

### ----------------------------- Get Shoe Data -------------------------------

# Get shoe data
for i in range(len(shoe_ids)):
    string = shoe_file_path + shoe_file_name_prefix + shoe_ids[i]
    temp_data_panda = pd.read_json(string + '.txt')  
    
    temp_data = np.array(temp_data_panda)
    column_to_be_split = temp_data[:,1]
    no_steps, step_activity = getNumberOfStepsAndStepActivityAsColumns(column_to_be_split)
    
    # remove the 'data' column and replace it with the two resulting columns from above
    temp_data_shortened = np.delete(temp_data, 1, 1)
    temp_data = np.insert(temp_data_shortened, 1, no_steps, 1)
    temp_data = np.insert(temp_data, 2, step_activity, 1)  
    
    temp_data_list = temp_data.tolist()
    all_shoe_data.append(temp_data_list)
    
    to_csv_panda_data = pd.DataFrame(temp_data, columns = ['id', 'step_counter','step_activity','timestamp','deviceID'])
    
    
    if os.path.exists(string + '.csv') == False:        
        to_csv_panda_data.to_csv(string + '.csv')
    else:
        if overwrite_csv_files == True:
            os.remove(string + '.csv')
            to_csv_panda_data.to_csv(string + '.csv')
        
        
    
### ---------------------------- Get Watch Data -------------------------------
    
# Get watch data
for i in range(len(watch_ids)):
    string = watch_file_path + watch_file_name_prefix + watch_ids[i]
        
    file1 = open(string + '.txt', 'r')
    Lines = file1.readlines()
    temp_data_list = np.empty((len(Lines) - 2, 41), dtype=object)
    
    hopefully_the_same_lenght = []
    
    for j in range(len(Lines)):
        if j >= 2:
            # print('Now reached row ', j)
            temp = Lines[j].split('|') 
            temp[0] = temp[0].strip()
            temp[1] = temp[1].strip()
            temp[2] = temp[2].strip()
            temp[3] = temp[3].strip()
            
            data_cell_as_json_dict = json.loads(temp[1])
            
            temp_data_list[j - 2][0] = temp[0]
            temp_data_list[j - 2][1] = temp[3]
            
            temp_data_list[j - 2][2] = data_cell_as_json_dict["_id"]
            temp_data_list[j - 2][3] = data_cell_as_json_dict["_xt"]
            temp_data_list[j - 2][4] = data_cell_as_json_dict["ei"]
            temp_data_list[j - 2][5] = data_cell_as_json_dict["si"]
            temp_data_list[j - 2][6] = data_cell_as_json_dict["dt"]
            temp_data_list[j - 2][7] = data_cell_as_json_dict["s"]
            
            c_column_split = data_cell_as_json_dict["c"].split(";")
                
            if len(c_column_split) == 29:
                temp_data_list[j - 2][8] = c_column_split[0]
                temp_data_list[j - 2][9] = c_column_split[1]
                temp_data_list[j - 2][10] = c_column_split[2]
                temp_data_list[j - 2][11] = c_column_split[3]
                temp_data_list[j - 2][12] = c_column_split[4]
                temp_data_list[j - 2][13] = c_column_split[5]
                temp_data_list[j - 2][14] = c_column_split[6]
                temp_data_list[j - 2][15] = c_column_split[7]
                temp_data_list[j - 2][16] = c_column_split[8]
                temp_data_list[j - 2][17] = c_column_split[9]
                temp_data_list[j - 2][18] = c_column_split[10]
                temp_data_list[j - 2][19] = c_column_split[11]
                temp_data_list[j - 2][20] = c_column_split[12]
                temp_data_list[j - 2][21] = c_column_split[13]
                temp_data_list[j - 2][22] = c_column_split[14]
                temp_data_list[j - 2][23] = c_column_split[15]
                temp_data_list[j - 2][24] = c_column_split[16]
                temp_data_list[j - 2][25] = c_column_split[17]
                temp_data_list[j - 2][26] = c_column_split[18]
                temp_data_list[j - 2][27] = c_column_split[19]
                temp_data_list[j - 2][28] = c_column_split[20]
                temp_data_list[j - 2][29] = c_column_split[21]
                temp_data_list[j - 2][30] = c_column_split[22]
                temp_data_list[j - 2][31] = c_column_split[23]
                temp_data_list[j - 2][32] = c_column_split[24]
                temp_data_list[j - 2][33] = c_column_split[25]
                temp_data_list[j - 2][34] = c_column_split[20]
                temp_data_list[j - 2][35] = c_column_split[21]
                temp_data_list[j - 2][36] = c_column_split[22]                
                pass
            
            if len(c_column_split) == 28:
                temp_data_list[j - 2][8] = c_column_split[0]
                temp_data_list[j - 2][9] = c_column_split[1]
                temp_data_list[j - 2][10] = c_column_split[2]
                temp_data_list[j - 2][11] = c_column_split[3]
                temp_data_list[j - 2][12] = c_column_split[4]
                temp_data_list[j - 2][13] = c_column_split[5]
                temp_data_list[j - 2][14] = c_column_split[6]
                temp_data_list[j - 2][15] = c_column_split[7]
                temp_data_list[j - 2][16] = c_column_split[8]
                temp_data_list[j - 2][17] = c_column_split[9]
                temp_data_list[j - 2][18] = c_column_split[10]
                temp_data_list[j - 2][19] = c_column_split[11]
                temp_data_list[j - 2][20] = c_column_split[12]
                temp_data_list[j - 2][21] = c_column_split[13]
                temp_data_list[j - 2][22] = c_column_split[14]
                temp_data_list[j - 2][23] = c_column_split[15]
                temp_data_list[j - 2][24] = c_column_split[16]
                temp_data_list[j - 2][25] = c_column_split[17]
                temp_data_list[j - 2][26] = c_column_split[18]
                temp_data_list[j - 2][27] = c_column_split[19]
                temp_data_list[j - 2][28] = c_column_split[20]
                temp_data_list[j - 2][29] = c_column_split[21]
                temp_data_list[j - 2][30] = c_column_split[22]
                temp_data_list[j - 2][31] = c_column_split[23]
                temp_data_list[j - 2][32] = c_column_split[24]
                temp_data_list[j - 2][33] = c_column_split[25]
                temp_data_list[j - 2][34] = c_column_split[20]
                temp_data_list[j - 2][35] = -1
                temp_data_list[j - 2][36] = c_column_split[22]                
                pass
            
            if len(c_column_split) == 11:
                temp_data_list[j - 2][8] = c_column_split[0]
                temp_data_list[j - 2][9] = c_column_split[1]
                temp_data_list[j - 2][10] = c_column_split[2]
                temp_data_list[j - 2][11] = c_column_split[3]
                temp_data_list[j - 2][12] = c_column_split[4]
                temp_data_list[j - 2][13] = c_column_split[5]                
                temp_data_list[j - 2][14] = -1
                temp_data_list[j - 2][15] = -1
                temp_data_list[j - 2][16] = -1
                temp_data_list[j - 2][17] = -1
                temp_data_list[j - 2][18] = -1
                temp_data_list[j - 2][19] = -1
                temp_data_list[j - 2][20] = -1
                temp_data_list[j - 2][21] = -1
                temp_data_list[j - 2][22] = -1
                temp_data_list[j - 2][23] = -1
                temp_data_list[j - 2][24] = -1
                temp_data_list[j - 2][25] = -1
                temp_data_list[j - 2][26] = -1
                temp_data_list[j - 2][27] = -1
                temp_data_list[j - 2][28] = -1
                temp_data_list[j - 2][29] = -1
                temp_data_list[j - 2][30] = -1
                temp_data_list[j - 2][31] = -1
                temp_data_list[j - 2][32] = c_column_split[6]
                temp_data_list[j - 2][33] = c_column_split[7]
                temp_data_list[j - 2][34] = c_column_split[8]
                temp_data_list[j - 2][35] = c_column_split[9]
                temp_data_list[j - 2][36] = c_column_split[10]                
                pass
            
            
            temp_data_list[j - 2][37] = data_cell_as_json_dict["y"]
            temp_data_list[j - 2][38] = data_cell_as_json_dict["m"]
            temp_data_list[j - 2][39] = data_cell_as_json_dict["d"]
            temp_data_list[j - 2][40] = data_cell_as_json_dict["p"] 
                
    
    """
    don't forget to substract 1 to find the list index!
    3.  _id: 60a3e80fb8162d6c4fd14a4b
    4.  _xt: 1621354511814
    5.  ei: 352413080064321
    6.  si: 9226103003325864
    7.  dt: 2021-05-18T16:15:11.814Z
    8.  s:  ::ffff:109.166.135.142
    9.  c_??: #@H02@#                   1
    10.  c_ei: 352413080064321          2 (same as 5?)
    11.  c_si: 9226103003325864         3 (same as 6?)
    12.  c_?!: 862182                   4
    13.  c_d1: 2021-05-18               5
    14.  c_t1: 19:15:08                 6
    15.  c_n1: 1                        7
    16.  c_l1: G                        8
    17.  c_m1: 0                        9
    18.  c_N1: N44.456400               10
    19.  c_E1: E26.074771               11
    20.  c_p1: 98.1                     12 
    21.  c_p2: 5.899                    13
    22.  c_m2: 0                        14
    23.  c_!?: 5076                     15
    24.  c_n2: 5                        16
    25.  c_n3: 6                        17
    26.  c_d2: 2021-05-18               18
    27.  c_t2: 19:03:36                 19
    28.  c_N2: N44.456315               20
    29.  c_E2: E26.075319               21
    30.  c_n4: 24                       22
    31.  c_n5: 50                       23
    32.  c_!!: 226@10@140@38121@54      24
    33.  c_00: 00                       25
    34.  c_0:  0                        26
    35.  c_L1: 1                        27
    36.  c_L2: -                        28
    37.  c_en: \u0001                   29
    38.  y: 2021
    39.  m: 5
    40.  d: 18
    41.  p: H02
    """
            
    temp_data = np.array(temp_data_list)    
    to_csv_panda_data = pd.DataFrame(temp_data, columns = ['id', 'device_id', '_id', '_xt', 'ei', 'si', 'dt', 's', 
                                                           'c_??', 'c_ei', 'c_si', 'c_?!', 'c_d1', 'c_t1', 'c_n1', 'c_l1',
                                                           'c_m1', 'c_N1', 'c_E1', 'c_p1', 'c_p2', 'c_m2', 'c_!?', 'c_n2', 
                                                           'c_n3', 'c_d2', 'c_t2', 'c_N2', 'c_E2', 'c_n4', 'c_n5', 'c_!!', 
                                                           'c_00', 'c_0', 'c_L1', 'c_L2', 'c_en',  'y', 'm', 'd', 'p'])
    
    all_watch_data.append(temp_data_list)
    
    if os.path.exists(string + '.csv') == False:        
        to_csv_panda_data.to_csv(string + '.csv')
    else:
        if overwrite_csv_files == True:
            os.remove(string + '.csv')
            to_csv_panda_data.to_csv(string + '.csv')
    
    
# ### ---------------------------- Get Survey Data ------------------------------

# # Get survey data
# for i in range(len(survey_ids)):
#     string = survey_file_path + survey_file_name_prefix + survey_ids[i]
#     to_csv_panda_data = pd.read_json(string + '.txt')
    
#     temp_data = np.array(temp_data_panda)  
#     difference_column = getTimestampsDifferenceAsInt(temp_data[:,5], temp_data[:,6], 'ms')
    
#     # remove de timestamps columns and replace them with the difference column from above
#     temp_data_shortened = np.delete(temp_data, 6, 1)
#     temp_data_shortened = np.delete(temp_data_shortened, 5, 1)
#     temp_data = np.insert(temp_data_shortened, 5, difference_column, 1)    
    
#     temp_data_list = temp_data.tolist()
#     all_survey_data.append(temp_data_list)
    
#     plt.figure(2)
#     plt.plot(temp_data[:,5])
#     plt.show()
    
#     if os.path.exists(string + '.csv') == False:        
#         to_csv_panda_data.to_csv(string + '.csv')
#     else:
#         if overwrite_csv_files == True:
#             os.remove(string + '.csv')
#             to_csv_panda_data.to_csv(string + '.csv')
