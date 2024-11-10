import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal

#Problem 1
raw_H1 = h5py.File("H-H1_LOSC_4_V2-1126259446-32.hdf5", "r")
raw_L1 = h5py.File("L-L1_LOSC_4_V2-1126259446-32.hdf5", "r")
data_L1 = raw_L1["strain/Strain"][()]
data_H1 = raw_H1["strain/Strain"][()]
sample_rate = 4096

total_samples = data_L1.shape[0]
time_length = total_samples/sample_rate

print(f"x_H[n] has {data_H1.shape[0]} samples")
print(f"x_L[n] has {data_L1.shape[0]} samples")
print(f"The signals last for {time_length} seconds.")

t_values = np.linspace(0,time_length,total_samples,endpoint=True)

#Problem 3 Selecting window function
M = 4096

hamming = scipy.signal.windows.hamming(M)
hann = scipy.signal.windows.hann(M)

#3a/b
f_1 = 31.5
f_2 = 1234.56
A_1 = 10^(-5)
A_2 = 1
test_x = lambda x: A_1*np.cos(2*np.pi*f_1*x/sample_rate) + A_2*np.cos(2*np.pi*x/sample_rate)
n = np.arange(M)
x_vals = test_x(n)
x_vals_fft = np.fft.fftshift(np.fft.fft(x_vals))
x_vals_hz = x_vals_fft * sample_rate / (2*np.pi)

freqbin = np.fft.fftshift(np.fft.fftfreq(M, 1/sample_rate))
x_hann = hann*x_vals_hz
x_hamming = hamming*x_vals_hz
power_x_hann = 10*np.log10(abs(x_hann)**2)
power_x_hamming = 10*np.log10(abs(x_hamming)**2)
power_x = 10*np.log10(abs(x_vals_hz)**2)
idx1 = (np.abs(freqbin-f_1)).argmin()
idx2 = (np.abs(freqbin-f_2)).argmin()
#Hamming window detects the weak test-frequency the best
#Thus, we will use it on the actual signal
"""
#Plot 3g
fig,ax = plt.subplots(3,1,figsize=(8,8))
ax[0].plot(freqbin, power_x,zorder=1)
ax[0].set_title("$\hat{x}[k]$ Without tapering")
ax[0].scatter(freqbin[idx1],power_x[idx1],s=10, color="green", label=f"{f_1}Hz")
ax[0].scatter(freqbin[idx2],power_x[idx2],s=10, color="green", label=f"{f_2}Hz")
ax[1].plot(freqbin, power_x_hamming,zorder=1)
ax[1].set_title("$\hat{x}[k]$ w/ Hamming window")
ax[1].scatter(freqbin[idx1],power_x_hamming[idx1],s=10, color="green", label=f"{f_1}Hz")
ax[1].scatter(freqbin[idx2],power_x_hamming[idx2],s=10, color="green", label=f"{f_2}Hz")
ax[2].plot(freqbin, power_x_hann,zorder=1)
ax[2].set_title("$\hat{x}[k]$ w/ Hann window")
ax[2].set_xlabel("Frequency (Hz)")
ax[2].scatter(freqbin[idx1],power_x_hann[idx1],s=10, color="green", label=f"{f_1}Hz")
ax[2].scatter(freqbin[idx2],power_x_hann[idx2],s=10, color="green", label=f"{f_2}Hz")

for i in range(0,3):
    ax[i].set_ylabel("Power (dB)")
    #ax[i].legend()
plt.tight_layout()
plt.show()
"""


#Problem 4 Estimating spectrum of LIGO signal
window = scipy.signal.windows.hamming(total_samples)
windowed_xH = window * data_H1
windowed_xL = window * data_L1
fft_xH = np.fft.fftshift(np.fft.fft(windowed_xH))
fft_xL = np.fft.fftshift(np.fft.fft(windowed_xL))
#Convert units to hertz instead of radians per sample:
fft_xH = fft_xH * sample_rate / (2*np.pi)
fft_xL = fft_xL * sample_rate / (2*np.pi)
freqs = np.fft.fftshift(np.fft.fftfreq(total_samples, 1/sample_rate))
pos_freq_ind = np.where(freqs >= 0)
pos_freqs = freqs[pos_freq_ind]
fft_xH_pos = fft_xH[pos_freq_ind]
fft_xL_pos = fft_xL[pos_freq_ind]
power_xH = 10*np.log10(abs(fft_xH_pos)**2)
power_xL = 10*np.log10(abs(fft_xL_pos)**2)
narrowband_interference_H = [1470, 1000, 505, 7]
narrowband_interference_L = [1505, 1010, 520, 7]

"""
#Plot 4c
fig,ax = plt.subplots(2,1,figsize=(10,8))
ax[0].plot(pos_freqs, power_xH, color="red")
ax[0].set_title("Power spectrum of $\hat{x}_H(k)$")
ax[1].plot(pos_freqs, power_xL, color="blue")
ax[1].set_title("Power spectrum of $\hat{x}_L(k)$")
for i in narrowband_interference_H:
    ax[0].scatter(i, -220, s=10, color = "green", marker="x")
for i in narrowband_interference_L:
    ax[1].scatter(i, -200, s=10, color = "green", marker="x")
for i in range(0,2):
    ax[i].set_ylabel("Power (dB)")
    ax[i].set_xlabel("Frequency (Hz)")
    #ax[i].legend()
plt.tight_layout()
plt.show()
"""


#Problem 5 Whitening filter
whitening_xH = 1/np.abs(fft_xH)
whitening_xL = 1/np.abs(fft_xL)

hat_yH = whitening_xH * fft_xH
hat_yL = whitening_xL * fft_xL

yH = np.fft.ifftshift(np.fft.ifft(hat_yH))
yL = np.fft.ifftshift(np.fft.ifft(hat_yL))

#Problem 6 Lowpass filtering
omega = 2*np.pi*freqs/sample_rate
cutoff = 300
L = int(sample_rate/cutoff) #=13
transfer = abs(np.sin(omega*L/2)/(L*np.sin(omega/2)))
transfer = 10*np.log10(abs(transfer*L)**2)
#Time delay from filter = 6 samples
delay_samples = 6
delay_sec = delay_samples / sample_rate
filtered_yH = np.zeros(total_samples)
filtered_yL = np.zeros(total_samples)
#Applying running average filter
for i in range(0, total_samples):
    s1=0
    s2=0
    for k in range(0,L-1):
        s1 += yH[i-k]
        s2 += yL[i-k]
    filtered_yH[i] = 1/L * s1
    filtered_yL[i] = 1/L * s2
#Reversing time delay
shifted_t_values = t_values - delay_sec

#Problem 7
n0 = np.linspace(-10,10,21,endpoint=True)
tau = n0 / sample_rate
idx1 = (np.abs(shifted_t_values-16.1)).argmin()
idx2 = (np.abs(shifted_t_values-16.6)).argmin()
n0 = 30
tau = n0 / sample_rate #delay = 0.488ms
filtered_yH = np.roll(filtered_yH, n0)

#Problem 8
def spectrogram(x, M, N, delta_n): #Credit: Lecture notes Week 42 p. 283 & p. 287
    w = scipy.signal.windows.hamming(N)
    L = len(x)
    t_max = int(np.floor((L-N)/delta_n))
    H = np.zeros([t_max, M], dtype=np.complex64)
    sub_array = np.zeros(N)
    for i in range(t_max):
        sub_array[:N] = x[i*delta_n + np.arange(N)]
        H[i, :] = np.fft.fft(sub_array*w, M)
    return H

idx1 = (np.abs(shifted_t_values-15.5)).argmin()
idx2 = (np.abs(shifted_t_values-17.0)).argmin()
M = int(sample_rate/2)
delta_n = 10
S = spectrogram(filtered_yH[idx1:idx2], M, 128, delta_n)
frequencies = np.fft.fftfreq(M, 1/sample_rate)
times = delta_n * np.arange(S.shape[0])/sample_rate
S_dB = np.transpose(10*np.log10(abs(S[:, :int(M/2)])**2))

#Problem 9
#Want to apply a better lowpass filter
cutoff = 100
order = 5
nyquist = sample_rate/2
normalized_cutoff = cutoff/nyquist
b,a = scipy.signal.butter(order, normalized_cutoff, btype="low", analog=False)
new_yH = scipy.signal.lfilter(b,a,yH)
new_yL = scipy.signal.lfilter(b,a,yL)

#Plot 9.1
t1 = 16.2
t2 = 16.5
idx1 = (np.abs(t_values - t1)).argmin()
idx2 = (np.abs(t_values - t2)).argmin()

fig,ax = plt.subplots(2,1, figsize=(10,8))
#ax[0].plot(t_values, yH, color="red")
#ax[1].plot(t_values, yL, color="blue")
ax[0].plot(t_values[idx1:idx2], new_yH[idx1:idx2], color="red")
ax[1].plot(t_values[idx1:idx2], new_yL[idx1:idx2], color="blue")
ax[0].set_title("Hanford")
ax[1].set_title("Livingston")
ax[0].set_xlabel("Time (s)")
ax[1].set_xlabel("Time (s)")
ax[0].set_ylabel("Whitened and filtered $y_H[n]$")
ax[1].set_ylabel("Whitened and filtered $y_L[n]$")
plt.tight_layout()
plt.show()

#Plot 9.2
idx1 = (np.abs(shifted_t_values-15.5)).argmin()
idx2 = (np.abs(shifted_t_values-17.0)).argmin()
S = spectrogram(new_yH[idx1:idx2], M, 128, delta_n)
S_dB = np.transpose(10*np.log10(abs(S[:, :int(M/2)])**2))
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies[:int(M/2)], S_dB, shading='gouraud', vmin=-80)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time after 15.5s [s]')
plt.title('Dynamic Spectrum of $|\hat{x}_H[t,k]|$')
plt.colorbar(label='Intensity [dB]')
plt.ylim([0, 400])  # Adjust this as per your frequency content interest
plt.show()



"""
#Plot 8
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies[:int(M/2)], S_dB, shading='gouraud', vmin=-80)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time after 15.5s [s]')
plt.title('Dynamic Spectrum of $|\hat{x}_H[t,k]|$')
plt.colorbar(label='Intensity [dB]')
plt.ylim([0, 400])  # Adjust this as per your frequency content interest
plt.show()
"""

"""
#Plot 7a


plt.plot(shifted_t_values[idx1:idx2], np.abs(filtered_yH[idx1:idx2]), label="Hanford", color="red")
plt.plot(shifted_t_values[idx1:idx2], np.abs(filtered_yL[idx1:idx2]), label="Livingston")
plt.title(f"Magnitude of signals, $y_H$ shifted by {n0} samples")
plt.ylabel("Strain")
plt.xlabel("Time (s)")
plt.legend()
plt.show()
"""



"""
#Plot 6f
idx1 = (np.abs(t_values-16.1)).argmin()
idx2 = (np.abs(t_values-16.6)).argmin()
fig,ax = plt.subplots(2,1,figsize=(10,8))
ax[0].plot(shifted_t_values[idx1:idx2], filtered_yH[idx1:idx2], label="Hanford", color="red")
ax[1].plot(shifted_t_values[idx1:idx2], filtered_yL[idx1:idx2], label="Livingston")
ax[0].set_title("Hanford Filtered + Whitened Signal")
ax[1].set_title("Livingston Filtered + Whitened Signal")
for i in range(0,2):
    ax[i].set_ylabel("Strain")
    ax[i].set_xlabel("Time (s)")
    #ax[i].legend()
plt.tight_layout()
plt.show()
"""

"""
#Plot 6a/b
idx2 = (np.abs(freqs-320)).argmin()
idx1 = (np.abs(freqs-280)).argmin()
plt.plot(freqs, transfer)
#plt.plot(freqs[idx1:idx2], transfer[idx1:idx2],zorder=1)
plt.scatter(300,-6,color="green", marker="x", s = 12, zorder=2,label = "POI: (300Hz, -6dB)")
plt.title(f"Magnitude response L = {L}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
#plt.axhline(y=-6, color='red', linestyle='--', linewidth=2, label='y = -6')
#plt.axvline(x=300, color='red', linestyle='--', linewidth=2, label='x = 300')
plt.legend()
plt.show()
"""



"""
#Plot 2a
fig,ax = plt.subplots(2,1,figsize=(8,8))
ax[0].plot(t_values,data_H1,label="$x_H[n]$", color="red")
ax[1].plot(t_values, data_L1,label="$x_L[n]$",color="blue")
ax[0].set_title("Hanford, $x_H[n]$")
ax[1].set_title("Livingston, $x_L[n]$")
ax[0].set_xlabel("Time (s)")
ax[1].set_xlabel("Time (s)")
ax[0].set_ylabel("Strain")
ax[1].set_ylabel("Strain")
plt.tight_layout()
plt.show()
"""






"""
#Plot 3b
fig,ax = plt.subplots(3,1,figsize=(8,10))

ax[0].plot(n,x_vals)
ax[0].set_title("No window")
ax[0].set_xlabel("n")
ax[1].set_xlabel("n")
ax[2].set_xlabel("n")
ax[0].set_ylabel("x[n]")
ax[1].set_ylabel("x[n]")
ax[2].set_ylabel("x[n]")
ax[1].plot(n,x_vals*hamming)
ax[1].set_title("Hamming window")
ax[2].plot(n,x_vals*hann)
ax[2].set_title("Hann window")
plt.tight_layout()
plt.show()
"""





"""
#Plot 5c
t1 = 16.2
t2 = 16.5
idx1 = (np.abs(t_values - t1)).argmin()
idx2 = (np.abs(t_values - t2)).argmin()

fig,ax = plt.subplots(2,1, figsize=(10,8))
#ax[0].plot(t_values, yH, color="red")
#ax[1].plot(t_values, yL, color="blue")
ax[0].plot(t_values[idx1:idx2], yH[idx1:idx2], color="red")
ax[1].plot(t_values[idx1:idx2], yL[idx1:idx2], color="blue")
ax[0].set_title("Hanford")
ax[1].set_title("Livingston")
ax[0].set_xlabel("Time (s)")
ax[1].set_xlabel("Time (s)")
ax[0].set_ylabel("Whitened strain $y_H[n]$")
ax[1].set_ylabel("Whitened strain $y_L[n]$")
plt.tight_layout()
plt.show()
"""