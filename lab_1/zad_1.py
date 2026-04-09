import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.fftpack

data, fs = sf.read('sound1.wav', dtype='float32')  

print(data.dtype)
print(data.shape)

left_channel = data[:,0]
right_channel = data[:,1]
mix_channel = (left_channel + right_channel) / 2

sf.write('channels/sound_L.wav', left_channel, fs)
sf.write('channels/sound_R.wav', right_channel, fs)
sf.write('channels/sound_mix.wav', mix_channel, fs)

plt.figure()

plt.subplot(3,1,1)
plt.plot(left_channel)
plt.title("Left Channel")

plt.subplot(3,1,2)
plt.plot(right_channel)
plt.title("Right Channel")

plt.subplot(3,1,3)
plt.plot(mix_channel)
plt.title("Mixed Channel")

plt.tight_layout()
plt.savefig("img_out/zad_1/img.png", dpi=300)
plt.close()

data, fs = sf.read('SIN/sin_440Hz.wav', dtype=np.int32)

plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(0,data.shape[0])/fs,data)

plt.subplot(2,1,2)
yf = scipy.fftpack.fft(data)
plt.plot(np.arange(0,fs,1.0*fs/(yf.size)),np.abs(yf))
plt.savefig("img_out/zad_1/plot.png", dpi=300)
plt.close()

fsize=2**8

plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(0,data.shape[0])/fs,data)
plt.subplot(2,1,2)
yf = scipy.fftpack.fft(data,fsize)
plt.plot(np.arange(0,fs/2,fs/fsize),20*np.log10( np.abs(yf[:fsize//2])))
plt.savefig("img_out/zad_1/plot_2.png", dpi=300)
plt.close()