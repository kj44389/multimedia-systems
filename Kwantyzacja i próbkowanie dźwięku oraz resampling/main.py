import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import sounddevice as sd
import soundfile as sf
from scipy.interpolate import interp1d

def redukcja(bits, data):
    if bits >= 2 and bits <= 32:
        scale = (np.max(data) - np.min(data))/ 2**(bits-1)
        for i in range(len(data)):
            data[i] = data[i]/scale
        return data
    else:
        print("error only bits in range <2,32>")

def decymacja(n, data):
    return data[0:len(data):n]

def interpolacja(method, data, fs, new_fs):
    granica = len(data) / fs
    x = np.linspace(0, granica, len(data))
    y = data
    x1 = np.linspace(0, granica, round(granica*new_fs))
    if(method == 'lin'):
        metode_lin = interp1d(x, y)
        return metode_lin(x1)
    else:
        metode_nonlin = interp1d(x, y, kind='cubic')
        return metode_nonlin(x1)

def show(data, fs, n, titles):
    fsize = 2**8
    x = np.arange(0,data.shape[0]) / fs
    yf = scipy.fftpack.fft(data, 256)
    axs[n].plot(np.arange(0, fs / 2, fs / fsize), 20 * np.log10(np.abs(yf[:fsize // 2])))
    # axs[n].plot(x,data)
    axs[n].set_title(titles)

def show2(data, fs,title, fsize = 2**8):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, data.shape[0]) / fs, data)
    plt.subplot(2, 1, 2)
    plt.title(title)
    yf = scipy.fftpack.fft(data, fsize)
    plt.plot(np.arange(0, fs / 2, fs / fsize), 20 * np.log10(np.abs(yf[:fsize // 2])))
    plt.show()



if __name__ == '__main__':
    names = ['sin_60Hz.wav','sin_440Hz.wav','sin_8000Hz.wav']#,'sin_combined.wav']
             # 'sing_high1.wav','sing_high2.wav','sing_low1.wav','sing_low2.wav',
             # 'sing_medium1.wav','sing_medium2.wav']
    bits = [4,8,16]
    # fss = [2000,4000]#,8000,16000,24000,41000,16950]
    for name in names:

            # for new_fs in fss:
                data, fs = sf.read(name, dtype=np.int32)
                print(fs)


                # data_red = redukcja(bit, np.copy(data))
                # if (new_fs != 16950):
                #     data_dec = decymacja(10, np.copy(data_red))
                # data_int = interpolacja('lin', np.copy(data_red), fs, new_fs)
                # data_int2 = interpolacja('nolin', np.copy(data_red), fs, new_fs)

                #show data
                new_fs = [2000,4000,8000,16000,24000]
                probka = [24,12,6,3,2]
                fig, axs = plt.subplots(6,1)
                fig.set_size_inches(8,6)
                fig.tight_layout()
                show(data, fs, 0, 'oryginalny')
                show(decymacja(probka[0], data), fs / probka[0], 1, 'decymacja z krokiem ' + str(probka[0]))
                show(decymacja(probka[1], data), fs / probka[1], 2, 'decymacja z krokiem ' + str(probka[1]))
                show(decymacja(probka[2], data), fs / probka[2], 3, 'decymacja z krokiem ' + str(probka[2]))
                show(decymacja(probka[3], data), fs / probka[3], 4, 'decymacja z krokiem ' + str(probka[3]))
                show(decymacja(probka[4], data), fs / probka[4], 5, 'decymacja z krokiem ' + str(probka[4]))
                # show(interpolacja('nolin', np.copy(data), fs, new_fs[0]), new_fs[0], 1, 'interpolacja linear z czestotliwoscia '+str(new_fs[0]))
                # show(interpolacja('nolin', np.copy(data), fs, new_fs[1]), new_fs[1], 2, 'interpolacja linear z czestotliwoscia '+str(new_fs[1]))
                # show(interpolacja('nolin', np.copy(data), fs, new_fs[2]), new_fs[2], 3, 'interpolacja linear z czestotliwoscia '+str(new_fs[2]))
                # show(interpolacja('nolin', np.copy(data), fs, new_fs[3]), new_fs[3], 4, 'interpolacja linear z czestotliwoscia '+str(new_fs[3]))
                # show(interpolacja('nolin', np.copy(data), fs, new_fs[4]), new_fs[4], 5, 'interpolacja linear z czestotliwoscia '+str(new_fs[4]))
                # show2(data, fs, 'original  ' + name)
                # show2(data_red, fs, 'reduced  '+name)
                # show(data_dec, new_fs, 2, 'decimation  '+name+" --- "+str(bit)+"_"+str(new_fs))
                # show(data_int, new_fs, 3, 'interp lin  '+name+" --- "+str(bit)+"_"+str(new_fs))
                # show(data_int2, new_fs, 4, 'interp nolin  '+name+" --- "+str(bit)+"_"+str(new_fs))
                plt.show()

                # sf.write('D:\SEMESTR_5\SM\lab5\spraw\\'+name+"_"+str(bit)+"_"+str(fs)+".wav", data_red,fs)
                # sf.write('D:\SEMESTR_5\SM\lab5\spraw\\' + name + "_" + str(bit) + "_" + str(new_fs)+".wav", data_dec,new_fs)
                # sf.write('D:\SEMESTR_5\SM\lab5\spraw\\' + name + "_" + str(bit) + "_" + str(new_fs)+".wav", data_int,new_fs)
                # sf.write('D:\SEMESTR_5\SM\lab5\spraw\\' + name + "_" + str(bit) + "_" + str(new_fs)+".wav", data_int2,new_fs)
    # sd.play(data_red,fs)






