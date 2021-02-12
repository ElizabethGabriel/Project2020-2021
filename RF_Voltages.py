import numpy as np
import matplotlib.pyplot as plt

#generate realistic RF pulse
gyro = 42.58 #MHz/T
pulseDuration = 10000 #micro s
sliceThickness = 2 #mm
BWT = 10
Emp = 1
nT=1000
gradient = (BWT * (10**6)) / (gyro * pulseDuration * sliceThickness) #mT/m
t = np.linspace(-(pulseDuration/2), (pulseDuration/2), nT)
V_max = 1 #V
TR = 100 #ms
dt = (pulseDuration*(10**-6)) / nT #s

A_t = ((1+np.cos((2*(np.pi)*t)/pulseDuration))/2) * (np.sinc((np.pi) * gyro * gradient * Emp * sliceThickness * (10**-6) * t ))
V_t = V_max * A_t

plt.plot((t*(10**-6)),V_t)

plt.title("RF Function")
plt.ylabel("Voltage")
plt.xlabel("t (s)")
plt.show()

#mod or square
V_t_mod = np.abs(V_t)

#integrate
V_t_cumulative = np.cumsum(V_t_mod) * dt
V_total = V_t_cumulative[(len(V_t_cumulative)-1)]

#6 min equivalent voltage
V_6min = V_total * ((6*60)/(TR*(10**-3)))
V_6min_sq = V_6min ** 2





