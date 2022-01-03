# -*- coding: utf-8 -*-

# ----------------- Comentarii -----------------
# Niste programele mici aici. Le-am despartit cat mai mult ca sa fie usor sa le selectezi
#   in intregime si sa le decomentezi/comentezi cum ai nevoie

# F5 sa rulezi cod. F9 sa rulezi codul selectat/linia curenta
# Vezi ca Python nu are { }, FOR-ul, IF-ul si alte instructiuni se termina acolo
#   unde schimbi tu identarea inapoi la cea precedenta!
# Ca sa comentezi linii multiple, ctrl+1. Tot ctrl+1 ca sa le decomentezi
# Naming conventions: functii_numite_asa, variabile_numite_tot_asa
#   vezi https://realpython.com/python-pep8/#naming-conventions


# ----------------- Importuri ------------------
# importurile aparent pot fi marcate cu un alias, ca sa nu scrii dita porcaria de fiecare data
# asta e ala care face grafice smechere gen matlab
import matplotlib.pyplot as plt
# asta e ca sa avem chestii misto numerice
import numpy as np
# asta e ca importez dintr-un alt fisier scris de mine
from functions.triassic import TriassicStuff
#
from array import *



# ------------ Declarari de functii ------------
# Functiile aparent se declara inainte de partea din cod in care pot fi folosite, sau in afara
# 
# Declaratie de functie care ridica la patrat
def square_function(x) :
    return x**2

def get_divisors(x):     
    array_of_divisors = []
    # o sa ia de la 1 la x/2 + 1, care poate fi float, nu integer. "range()" are nevoie de 
    # integer, de aia rotunjesc in jos prin "round()". M-am uitat pe variabila si am vazut ca
    # sunt rezultate mai bune daca folosesc x/2 + 1 fata de x/2. Sigur exista metode mai bune
    # de a face asta
    for i in range(1, round(x/2 + 1)):
        if x%i == 0:
            array_of_divisors.append(i)    
    return array_of_divisors

def factorial(x):
    if x == 0:
        return 1
    if x == 1:
        return 1
    else:
        return (x * factorial(x-1))

def fibonacci_recursive(N):    
    if N <= 1:
        return 1
    else:
        return fibonacci_recursive(N - 1) + fibonacci_recursive(N - 2)
    
def custom_fibonacci_recursive(N, X):    
    if N < 0:
        return 0
    if N == 0:
        return 1
    else:
        # asta face suma de mai sus, doar ca pentru fiecare membru x al vectorului X
        return sum(custom_fibonacci_recursive(N - x, X) for x in X)
    
def fibonacci(n):
    # aparent poti sa faci asignarea asa    
    a, b = 1, 2
    # aici nu folosim iteratorul deci pare ca nu trebuie nici macar specificat
    for _ in range(n - 1):
        # a = b
        # b = a + b
        a, b = b, a + b
    return a

def custom_fibonacci(n, X):
    
    # asta e echivalent cu cache = np.zeroes(n + 1)
    cache = [0 for _ in range(n + 1)]
    cache[0] = 1    
    for i in range(1, n + 1):
        cache[i] += sum(cache[i - x] for x in X if i - x >= 0)
    return cache[n]
    
        
def find_possible_combinations(N, possible_steps_taken, solving_array, solutions, tried_arrays):    
    if tried_arrays.count(solving_array) > 0:
        solving_array.pop(len(solving_array) - 1)
        return solving_array
    
    for i in range(len(possible_steps_taken)):
        
        solving_array.append(possible_steps_taken[i])
        print("\nCurrently managing: ", solving_array)
        
        if tried_arrays.count(solving_array) > 0:
            print(solving_array, " found in ", tried_arrays, "skipping...") 
            solving_array.pop(len(solving_array) - 1)
            print("Reduced to ", solving_array)
            return solving_array
        
        else:            
            if sum(solving_array) < N:                
                print("Sum smaller than N for: ", solving_array)
                temp_array = solving_array
                solving_array = find_possible_combinations(N, possible_steps_taken, solving_array, solutions, tried_arrays)
                if tried_arrays.count(solving_array) != 0:
                    solving_array = temp_array
                    
            if sum(solving_array) == N:
                if tried_arrays.count(solving_array) == 0:
                    solutions.append(solving_array.copy())
                    tried_arrays.append(solving_array.copy())
                    print("Found solution: ", solving_array, "!!!")
                    solving_array.pop(len(solving_array) - 1)
                    print("Reduced to ", solving_array)
                    
            if sum(solving_array) > N:                
                print("Sum bigger than N for: ", solving_array)
                tried_arrays.append(solving_array.copy())
                solving_array.pop(len(solving_array) - 1)
                if i == len(possible_steps_taken) - 1:                    
                    # solving_array.pop(len(solving_array) - 1)
                    pass
                print("Reduced to ", solving_array)
                return solving_array




# ----------------- Programu 1 -----------------
#
# Plot x^2 pentru intervalul [-1000, 1000], babeste. !00% exista deja functii
#
# Vector de numere definit prin interval
array_of_numbers = range(-1001, 1001)

# Vector gol
array_of_numbers_squared_up = []

# FOR loop
for i in array_of_numbers :
    array_of_numbers_squared_up.append(square_function(i))
    
# FOR loop, dar in care nu se intampla nimic. Echivalent cu for (i = 0; i < 5; i++) ;
for i in range(5) :
    pass

# asta e de la TD daca mai tii minte. Mai intai plotezi axa X, apoi axa Y
plt.plot(array_of_numbers, array_of_numbers_squared_up)
plt.show()



# # ----------------- Programu 2 -----------------
# #
# # Plot toate numerele prime pentru intervalul [0, 2000]
# #
# array_of_numbers = range(2001)
# array_of_prime_numbers = []

# for i in array_of_numbers:
#     # length(ceva)
#     if len(get_divisors(i)) == 1:
#         array_of_prime_numbers.append(i)
        
# # asta deseneaza buline (o) cu rosu (r)
# plt.plot(array_of_prime_numbers, 'ro')
# plt.show()



# # ----------------- Programu 3 -----------------
# #
# # Vezi fisierul Problema3.txt. Incercam recursiv aici, adica:
    
# # Pentru trepte posibile = [1, 2] avem urmatoarele combinatii posibile
# # N = 1 | 1 posibilitate | [1]
# # N = 2 | 2 posibilitati | [[1, 1], [2]]
# # N = 3 | 3 posibilitati | [[1, 1, 1], [1, 2], [2, 1]]
# # N = 4 | 5 posibilitati | [[1, 1, 1, 1], [1, 1, 2], [1, 2, 1], [2, 1, 1], [2, 2]]
# # N = 5 | 8 posibilitati | [[1, 1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 1], [1, 2, 1, 1], [1, 2, 2], [2, 1, 1, 1], [2, 1, 2], [2, 2, 1]]
# # Remarcam, deci, ca e sirul lui fibonacci aici

# # de la input vine string, asa ca ii facem cast la int
# N = int(input("Numarul de trepte ce trebuie urcate: "))
# TriassicStuff.log_stuff_going_on(N)
# chosen_steps = [1, 3, 5]
# chosen_steps.sort()

# print("[Recursiv] Numarul de variante posibile este ", fibonacci_recursive(N))
# print("[Secvential] Numarul de variante posibile este ", fibonacci(N))

# print("[Recursiv] Pentru sirul ", chosen_steps, ", numarul de variante posibile este ", custom_fibonacci_recursive(N, chosen_steps))
# print("[Secvential] Pentru sirul ", chosen_steps, ", numarul de variante posibile este ", custom_fibonacci(N, chosen_steps))


# # ----------------- Programu 4 -----------------
# #
# # Vezi fisierul Problema3.txt. Incercam sa aflam si combinatiile posibile, nu doar numarul lor
# # da smtbmm ca ma ia ceafa cu problema asta
# #
# N = input("Enter number of stairs to be walked up: ")
# print(N)
# TriassicStuff.log_stuff_going_on(N)

# possible_steps_taken = [1, 2]
# possible_steps_taken.sort()
# solving_array = []
# solutions = []
# tried_arrays = []

# find_possible_combinations(N, possible_steps_taken, solving_array, solutions, tried_arrays)
# print("Sa speram: ", solutions)
