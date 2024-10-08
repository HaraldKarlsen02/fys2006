#Source: https://github.com/jvierine/signal_processing_course/blob/main/code/027_sonar/sonar_deconv.py

import matplotlib.pyplot as plt
import numpy as np

x = np.fromfile("chirp_2024.bin", dtype = np.float32)
m = np.fromfile("chirp_rec_2024.bin", dtype = np.float32)

print("Length of transmission:", x.shape)
print("Length of measurement:", m.shape)

#Implementing lambda[n] and using it so compute c[n]
deconvolotuion_filter = x[::-1]
c =  np.convolve(x, deconvolotuion_filter, mode="same")

#Separating m[n] into multiple columns of measurements (10000 samples in each)
interpulse_period = 10000
interpulses = int(np.floor(len(m)/interpulse_period))
P = np.zeros([interpulses, interpulse_period])
#For each measurement we calculate the power of the transmitted signal
for i in range(interpulses):
    echo = m[(i*interpulse_period):(i+1)*interpulse_period]
    deconvolved = np.convolve(echo,deconvolotuion_filter, mode="same")
    P[i,:] = np.abs(deconvolved)**2

plt.plot(x)
plt.xlabel("Time (samples)")
plt.ylabel("Transmitted signal $x[n]$")
plt.show()

plt.plot(m[:10000])
plt.xlabel("Time (samples)")
plt.ylabel("Received signal $m[n]$")
plt.show()

plt.plot(c)
plt.xlim(4950,5050)
plt.xlabel("Time (samples)")
plt.ylabel("Autocorrelation function $c[n]$")
plt.show()

v_g = 343.0
f_s = 44100

#Convert power into desibel
P_dB = 10*np.log10(P)
#Create arrays for the time and range
t = np.arange(interpulses)*interpulse_period/f_s
r = v_g*np.arange(interpulse_period)/f_s

plt.figure(figsize=(12,6))
c = plt.pcolormesh(t,r, P_dB.T, shading="auto", cmap="jet", vmin = -30, vmax=30)
plt.colorbar(c, label="Scattered Power (dB)")
plt.xlabel("Time (s)")
plt.ylabel("Range (m)")
plt.show()

