"""
This file contains useful functions to process data coming from vINCI. 
It usually comes as .txt files, with JSON format, or just as a .txt file made 
to look like a table
"""

import numpy as np

### ---------------------------------------------------------------------------
### --------------- 1. getNumberOfStepsAndStepActivityAsColumns ---------------
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
### --------------------- 2. getTimestampsDifferenceAsInt ---------------------
### ---------------------------------------------------------------------------

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