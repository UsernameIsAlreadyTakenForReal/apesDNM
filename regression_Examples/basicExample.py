import matplotlib.pyplot as plt
import numpy as np
import math

def run_regression(signal, regressors):
    
    remade_signal = []
    for i in range(len(signal)):        
        if i == 0:
            y = signal[0]
        else:
            y = 0
            for j in range(len(regressors)):
                if i - j > 0:
                    y += signal[i - j - 1]*regressors[j]
        
        remade_signal.append(y)
        
    return remade_signal

def get_normal_value(signal):
    
    sum_of_squared_elements = 0
    for i in range(len(signal)):
        sum_of_squared_elements += signal[i]**2 # la patrat
        
    return sum_of_squared_elements**(1/float(len(signal))) # suma^(1/n), unde n este nr de elemente

# creaza [0, 0.1, 0.2 ... 99.8, 99.9]
timeline = np.arange(0, 100, 0.1)
# face sinus din timeline
signal = np.sin(timeline)
# face un random de 1000 de elemente
noise = np.random.rand(1000)
# face un sinus cu zgomot
noisy_signal = signal + noise
# noisy_signal = signal

# bagam solutia la fiecare aici
solutions = np.empty([6, 1000])
final_regressors = []
# la fel cu eroarea
errors = np.empty([6, 1000])
# regresorii astia sunt unii pusi random
regressors = [0.5, 0.4, 0.3, 0.2, 0.1]
# pe care ii testam de complezenta
error = abs(noisy_signal - run_regression(noisy_signal, regressors))
error_normal = get_normal_value(error)

number_of_generations = 10

# --------------------- Tentativa de gasit regresori buni ---------------------
# si acu sa incercam un fel de 2 feluri
# sa gasim un numar n de solutii pt fiecare lungime de regresori 
# si care are eroarea mai buna, pastram

# mai intai complet random

for i in range(1, 7):
    
    
    regressors = np.random.rand(i)    
    solutions[i - 1] = run_regression(noisy_signal, regressors)
    final_regressors.append(regressors)
    errors[i - 1] = abs(noisy_signal - solutions[i - 1])
    med_error = get_normal_value(errors[i - 1])
    
    for _ in range(number_of_generations - 1):
        temp_regressors = np.random.rand(i)    
        temp_solution = run_regression(noisy_signal, regressors)
        temp_error = abs(noisy_signal - temp_solution)
        temp_med_error = get_normal_value(temp_error)
        
        if temp_med_error < get_normal_value(errors[i - 1]):
            final_regressors.pop()
            final_regressors.append(temp_regressors)
            solutions[i - 1] = temp_solution
            errors[i - 1] = temp_error
            

final_errors = []

for i in range(6):
    final_errors.append(get_normal_value(errors[i]))

# ---------------------------------- Complot ----------------------------------
fig1, (f1, f2, f3, f4) = plt.subplots(4, 1)
f1.plot(timeline, signal)
f2.plot(timeline, noisy_signal)
f3.plot(timeline, run_regression(noisy_signal, regressors))
f4.plot(timeline, error)

fig2, (f1, f2, f3, f4, f5, f6) = plt.subplots(6, 1)
f1.plot(timeline, solutions[0])
f2.plot(timeline, solutions[1])
f3.plot(timeline, solutions[2])
f4.plot(timeline, solutions[3])
f5.plot(timeline, solutions[4])
f6.plot(timeline, solutions[5])

fig3, (f1, f2, f3, f4, f5, f6) = plt.subplots(6, 1)
f1.plot(timeline, errors[0])
f2.plot(timeline, errors[1])
f3.plot(timeline, errors[2])
f4.plot(timeline, errors[3])
f5.plot(timeline, errors[4])
f6.plot(timeline, errors[5])
plt.show()