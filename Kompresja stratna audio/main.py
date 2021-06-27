import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.interpolate import interp1d


def colorfit(pixel, paleta):
    differences = []
    for color in paleta:
        differences.append(np.linalg.norm(color - pixel))
    index = np.argmin(differences)
    return paleta[index]

def get_y(data, paleta, kwantyzacja = True):
    qu = 255
    y = []
    for i in range(len(data)):
        if ((data[i]>= -1)&(data[i]<=1)):
            tmp = np.sign(data[i]) * ((np.log(1 + qu * np.abs(data[i]))) / (np.log(1 + qu)))
            if(kwantyzacja == True):
                y.append(colorfit(tmp,paleta))
            else:
                y.append(tmp)
    return y

def get_xprim(data):
    qu = 255
    xprim = []
    for i in range(len(data)):
        if ((data[i]>= -1)&(data[i]<=1)):
            tmp = np.sign(data[i]) * (1/qu)* (((1+qu)**np.abs(data[i]))-1)
            xprim.append(tmp)
    return xprim

def dpcm_decompress(sygnal):
    Y = sygnal
    X = np.zeros(Y.shape[0])
    X[0] = Y[0]
    for i in range(len(Y)):
        if i != 0:
            X[i] = X[i-1] + Y[i]
    return X

def dpcm_compress(sygnal, paleta):
    X = sygnal * 255
    Y = np.zeros(X.shape[0])
    print(Y.shape[0])
    E = X[0]
    Y[0] = colorfit(X[0],paleta)
    for i in range(len(X)):
        if i != 0:
            Y[i] = colorfit(X[i]-E,paleta)
            E += Y[i]
    return Y

def mu_law(data,bity):
    paleta = np.linspace(-1, 1, bity)
    y = get_y(np.copy(data), paleta, kwantyzacja = True)
    xprim = get_xprim(y)
    return xprim

def dpcm(data,bity):
    paleta_dpcm = np.linspace(-255, 255, bity)
    compress = dpcm_compress(data, paleta_dpcm)
    decompress = dpcm_decompress(compress)
    return decompress

if __name__ == '__main__':
    # x=np.linspace(-1,1,1000)
    name = 'sing_low1.wav'
    data, fs = sf.read(name, dtype=np.float)
    bity = 2**8

    print(name)
    mulaw = mu_law(np.copy(data),bity)
    print('mu_done')
    sdpcm = dpcm(np.copy(data),bity)
    print('dpcm_done')

    sf.write(str('sprawko\\mu'+name+''), mulaw, fs)
    print('mu_zap_done')
    sf.write(str('sprawko\\dpcm' + name + ''), sdpcm, fs)
    print('dpcm_zap_done')
