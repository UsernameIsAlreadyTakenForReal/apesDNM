# -*- coding: utf-8 -*-

# ----------------- Comentarii -----------------
# F5 sa rulezi cod. F9 sa rulezi codul selectat/linia curenta
# Vezi ca Python nu are { }, FOR-ul, IF-ul si alte instructiuni se termina acolo
#   unde schimbi tu identarea inapoi la cea precedenta!
# Ca sa comentezi linii multiple, ctrl+1. Tot ctrl+1 ca sa le decomentezi


# ----------------- Importuri ------------------
# importurile aparent pot fi marcate cu un alias, ca sa nu scrii dita porcaria de fiecare data
# asta e ala care face grafice smechere gen matlab
import matplotlib.pyplot as plt
# asta e ca sa avem chestii misto numerice
import numpy as np



# ------------ Declarari de functii ------------
# Functiile aparent se declara inainte de partea din cod in care pot fi folosite
# Declaratie de functie care ridica la patrat
def squareFunction(x) :
    return x**2

def getDivisors(x):     
    array_of_divisors = []
    # o sa ia de la 1 la x/2 + 1, care poate fi float, nu integer. "range()" are nevoie de 
    # integer, de aia rotunjesc in jos prin "round()". M-am uitat pe variabila si am vazut ca
    # sunt rezultate mai bune daca folosesc x/2 + 1 fata de x/2. Sigur exista metode mai bune
    # de a face asta
    for i in range(1, round(x/2 + 1)):
        if x%i == 0:
            array_of_divisors.append(i)    
    return array_of_divisors

# ----------------- Programu 1 -----------------
#
# Plot x^2 pentru intervalul [-1000, 1000], babeste. !00% exista deja functii
#
# Vector de numere definit prin interval
# array_of_numbers = range(-1001, 1001)

# # Vector gol
# array_of_numbers_squared_up = []

# # FOR loop
# for i in array_of_numbers :
#     array_of_numbers_squared_up.append(squareFunction(i))
    
# # FOR loop, dar in care nu se intampla nimic. Echivalent cu for (i = 0; i < 5; i++) ;
# for i in range(5) :
#     pass

# # asta e de la TD daca mai tii minte. Mai intai plotezi axa X, apoi axa Y
# plt.plot(array_of_numbers, array_of_numbers_squared_up)
# plt.show()

# ----------------- Programu 2 -----------------
#
# Plot toate numerele prime pentru intervalul [0, 2000]
#
array_of_numbers = range(2001)
array_of_prime_numbers = []

for i in array_of_numbers:
    if len(getDivisors(i)) == 1:
        array_of_prime_numbers.append(i)
        
plt.plot(array_of_prime_numbers, 'ro')
plt.show()

