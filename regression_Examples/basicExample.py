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

def get_median_value(signal):    
    return float(sum(signal) / len(signal))

def mutate_individuals(individuals, length, chance): 
    for i in range(len(individuals) - 1):
        if np.random.randint(chance):
            individuals[i][np.random.randint(length)] = np.random.rand(1)
    return individuals

def print_generation(individuals, fitnesses, generation_number, length):
    print("Length " + str(length) + ", generation: " + str(generation_number))
    for i in range(len(individuals)):
        print("Individual " + str(i+1) + ": " + str(individuals[i]) + ", fitness: " + str(fitnesses[i]))
    
    print("")

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
error_normal = get_median_value(error)

number_of_generations = 10
individuals_per_generations = 4
chance_to_mutate = 5



# # --------------------- Tentativa de gasit regresori buni ---------------------
# # ------------------------- mai intai complet random --------------------------

# for i in range(1, 7):    
    
#     regressors = np.random.rand(i)    
#     solutions[i - 1] = run_regression(noisy_signal, regressors)
#     final_regressors.append(regressors)
#     errors[i - 1] = abs(noisy_signal - solutions[i - 1])
#     med_error = get_median_value(errors[i - 1])
    
#     for _ in range(number_of_generations - 1):
#         temp_regressors = np.random.rand(i)    
#         temp_solution = run_regression(noisy_signal, regressors)
#         temp_error = abs(noisy_signal - temp_solution)
#         temp_med_error = get_median_value(temp_error)
        
#         if temp_med_error < get_median_value(errors[i - 1]):
#             final_regressors.pop()
#             final_regressors.append(temp_regressors)
#             solutions[i - 1] = temp_solution
#             errors[i - 1] = temp_error
            

# final_errors = []
# for i in range(6):
#     final_errors.append(get_median_value(errors[i]))
    
    
# ------------------------- Tentativa cu alg genetic --------------------------

# Pentru fiecare dimensiune posibila de regresori > 1
#   | Cat timp fitness > Y SAU timp de number_of_generations generatii
#      | Creat generatia   
#      | Aflat fitness pentru toti invidizii (eroare)
#      | Sortat indivizi pe baza fitnessului
#      | Ales primii X indivizi pentru combinare (prima jumatate din primul + 
#      |                    a doua jumatate din al doilea, hopa individ nou!)
#      | Sansa de mutare
#   | Adaugat cel mai bun individ la solutions[i], final_regressors[i] etc


solutions = np.empty([6, 1000])
final_errors = np.empty([6, 1000])
final_med_errors = np.empty(6)

for i in range(1, 6):
    
    # prima generatie pentru fiecare lungime de regresori
    individuals = np.empty([individuals_per_generations, i + 1])    
    for j in range(individuals_per_generations):
        individuals[j] = np.random.rand(i + 1)
        
    outputs = np.empty([individuals_per_generations, 1000])
    errors = np.empty([individuals_per_generations, 1000])
    fitness = np.empty(individuals_per_generations)
        
    for gen_number in range(number_of_generations):   
        
        individuals = mutate_individuals(individuals, i + 1, chance_to_mutate)
        for j in range(individuals_per_generations):
            outputs[j] = run_regression(noisy_signal, individuals[j])
            errors[j] = abs(noisy_signal - outputs[j])
            fitness[j] = get_median_value(errors[j])
            
        for j in range(individuals_per_generations - 1):
            for k in range(j, individuals_per_generations):
                if fitness[j] > fitness[k]: 
                    aux = fitness[j]
                    fitness[j] = fitness[k]
                    fitness[k] = aux                    
                    aux = outputs[j]
                    outputs[j] = outputs[k]
                    outputs[k] = aux                    
                    aux = errors[j]
                    errors[j] = errors[k]
                    errors[k] = aux
        
        print_generation(individuals, fitness, gen_number, i + 1)
        
        temp_individuals = np.empty([individuals_per_generations, i + 1])
        temp_individuals[0] = individuals[0]
        temp_individuals[1] = individuals[1]
        
        for j in range(int(i/2)):
            temp_individuals[2][j] = individuals[2][j]
            temp_individuals[3][j] = individuals[3][j]
        
        for j in range(int(i/2), i):
            temp_individuals[2][j] = individuals[3][j]
            temp_individuals[3][j] = individuals[2][j]
        
        individuals = temp_individuals        
        
    
    final_regressors.append(individuals[0])
    solutions[i] = outputs[0]
    final_errors[i] = errors[0]
    final_med_errors[i] = fitness[0]


# ---------------------------------- Complot ----------------------------------
# ----------------------- te-ai prins ca e plot cu com? -----------------------
# fig1, (f1, f2, f3, f4) = plt.subplots(4, 1)
# f1.plot(timeline, signal)
# f2.plot(timeline, noisy_signal)
# f3.plot(timeline, run_regression(noisy_signal, regressors))
# f4.plot(timeline, error, label = "error")
# f4.axhline(y=error_normal, color='r', linestyle='-', label = "error normal")

fig1, (f1, f2, f3, f4) = plt.subplots(4, 1)
fig1.suptitle('1 Regressor' )
f1.plot(timeline, signal)
f2.plot(timeline, noisy_signal)
f3.plot(timeline, solutions[0])
f4.plot(timeline, final_errors[0])
f4.axhline(y=final_med_errors[0], color='r', linestyle='-')

fig2, (f1, f2, f3, f4) = plt.subplots(4, 1)
fig2.suptitle('2 Regressors')
f1.plot(timeline, signal)
f2.plot(timeline, noisy_signal)
f3.plot(timeline, solutions[1])
f4.plot(timeline, final_errors[1])
f4.axhline(y=final_med_errors[1], color='r', linestyle='-')

fig3, (f1, f2, f3, f4) = plt.subplots(4, 1)
fig3.suptitle('3 Regressors')
f1.plot(timeline, signal)
f2.plot(timeline, noisy_signal)
f3.plot(timeline, solutions[2])
f4.plot(timeline, final_errors[2])
f4.axhline(y=final_med_errors[2], color='r', linestyle='-')

fig4, (f1, f2, f3, f4) = plt.subplots(4, 1)
fig4.suptitle('4 Regressors')
f1.plot(timeline, signal)
f2.plot(timeline, noisy_signal)
f3.plot(timeline, solutions[3])
f4.plot(timeline, final_errors[3])
f4.axhline(y=final_med_errors[3], color='r', linestyle='-')

fig5, (f1, f2, f3, f4) = plt.subplots(4, 1)
fig5.suptitle('5 Regressors')
f1.plot(timeline, signal)
f2.plot(timeline, noisy_signal)
f3.plot(timeline, solutions[4])
f4.plot(timeline, final_errors[4])
f4.axhline(y=final_med_errors[4], color='r', linestyle='-')

fig6, (f1, f2, f3, f4) = plt.subplots(4, 1)
fig6.suptitle('6 Regressors')
f1.plot(timeline, signal)
f2.plot(timeline, noisy_signal)
f3.plot(timeline, solutions[5])
f4.plot(timeline, final_errors[5])
f4.axhline(y=final_med_errors[5], color='r', linestyle='-')


plt.show()