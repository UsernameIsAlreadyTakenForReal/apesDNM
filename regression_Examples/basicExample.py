import matplotlib.pyplot as plt
import numpy as np

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

# creaza [0, 0.1, 0.2 ... 99.8, 99.9]
timeline = np.arange(0, 100, 0.1)
# face sinus din timeline
signal = np.sin(timeline)
# face un random de 1000 de elemente
noise = np.random.rand(1000)
# face un sinus cu zgomot
noisy_signal = signal + noise
# noisy_signal = signal

solutions = np.empty([6, 1000])
errors = np.empty([6, 1000])
regressors = [0.5, 0.4, 0.3, 0.2, 0.1]
error = abs(noisy_signal - run_regression(noisy_signal, regressors))

for i in range(1, 7):
    
    regressors = np.random.rand(i)    
    solutions[i - 1] = run_regression(noisy_signal, regressors)
    errors[i - 1] = abs(noisy_signal - solutions[i - 1])


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