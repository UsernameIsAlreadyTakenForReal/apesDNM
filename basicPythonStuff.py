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



# ------------ Declarari de functii ------------
# Functiile aparent se declara inainte de partea din cod in care pot fi folosite
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

def find_possible_combinations(N, possible_steps_taken, solving_array, index):        
    if sum(solving_array) == N:
        return solving_array
    else:     
        solving_array.append(possible_steps_taken[0]) 
        print("S-a facut pasul", index)
        return find_possible_combinations(N - 1, possible_steps_taken, solving_array, index + 1)   



# # ----------------- Programu 1 -----------------
# #
# # Plot x^2 pentru intervalul [-1000, 1000], babeste. !00% exista deja functii
# #
# # Vector de numere definit prin interval
# array_of_numbers = range(-1001, 1001)

# # Vector gol
# array_of_numbers_squared_up = []

# # FOR loop
# for i in array_of_numbers :
#     array_of_numbers_squared_up.append(square_function(i))
    
# # FOR loop, dar in care nu se intampla nimic. Echivalent cu for (i = 0; i < 5; i++) ;
# for i in range(5) :
#     pass

# # asta e de la TD daca mai tii minte. Mai intai plotezi axa X, apoi axa Y
# plt.plot(array_of_numbers, array_of_numbers_squared_up)
# plt.show()



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



# ----------------- Programu 3 -----------------
#
# Vezi fisierul Problema3.txt. Incercam recursiv aici
#
N = int(input("Enter number of stairs to be walked up: "))
print(N)
TriassicStuff.log_stuff_going_on(N)

possible_steps_taken = [1, 2]
solving_array = []

find_possible_combinations(N, possible_steps_taken, solving_array, 0)
print("O solutie arata astfel: ", solving_array)




# # ----------------- Programu 4 -----------------
# #
# # Vezi fisierul Problema3.txt. Incercam cu arbori aici
# #
# N = input("Enter number of stairs to be walked up: ")
# print(N)
# TriassicStuff.log_stuff_going_on(N)

# possible_steps_taken = [1, 2]
