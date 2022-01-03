import matplotlib.pyplot as plt
import numpy as np

# creaza [0, 0.1, 0.2 ... 99.8, 99.9]
timeline = np.arange(0, 100, 0.1)
# face sinus din timeline
signal = np.sin(timeline)
# face un random de 1000 de elemente
noise = np.random.rand(1000)
# face un sinus cu zgomot
noisy_signal = signal + noise
# noisy_signal = signal

regresors_coef = [0.5, 0.4, 0.3, 0.2, 0.1]

remade_signal = []

for i in range(len(noisy_signal)):
    
    if i == 0:
        y = noisy_signal[0]
    else:
        y = 0
        for j in range(len(regresors_coef)):
            if i - j > 0:
                y += noisy_signal[i - j - 1]*regresors_coef[j]
    
    remade_signal.append(y)
    
    
error = abs(noisy_signal - remade_signal)

fig, (f1, f2, f3, f4) = plt.subplots(4, 1)
f1.plot(timeline, signal)
f2.plot(timeline, noisy_signal)
f3.plot(timeline, remade_signal)
f4.plot(timeline, error)
plt.show()