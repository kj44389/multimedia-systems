import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a.astype(int), axis=0, norm='ortho' ), axis=1, norm='ortho' )
def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.astype(float), axis=0 , norm='ortho'), axis=1 , norm='ortho')
def split8x8_withquant(dane, tablica):
    for h in range(0,dane.shape[0], 8):
        for w in range(0,dane.shape[1], 8):
            dane[h:h+8, w:w+8] = kwantyzacja(dct2(dane[h:h+8, w:w+8]), tablica)
    return dane
def toarray_withquant(dane,tablica):
    for h in range(0, dane.shape[0], 8):
        for w in range(0, dane.shape[1], 8):
            dane[h:h + 8, w:w + 8] = idct2(kwantyzacja(dane[h:h + 8, w:w + 8], tablica, dekwantyzacja=True))
    return dane
def ycrkom(RGB):
    return cv2.cvtColor(RGB, cv2.COLOR_RGB2YCrCb).astype(int)
def rgbkom(YCrCb):
    return cv2.cvtColor(YCrCb.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
def kwantyzacja(dane, tablica, dekwantyzacja = False):
    if (dekwantyzacja == False):
        return np.round(dane/tablica).astype(int)
    else:
        return (dane*tablica)



def kompresja(dane, typ='4:4:4', type="qy"):

    ycr = ycrkom(dane)
    y = ycr[:, :, 0]
    cr = ycr[:, :, 1]
    cb = ycr[:, :, 2]

    y = y - 128
    cr = cr - 128
    cb = cb - 128
    if (typ == "4:2:2"):
        cr = np.delete(cr, np.arange(1, cr.shape[1], 2), axis=1)
        cb = np.delete(cb, np.arange(1, cb.shape[1], 2), axis=1)

    if(type == "qy"):
        tablica = np.array([
        [16, 11, 10, 16, 24,  40,  51,  61],
        [12, 12, 14, 19, 26,  58,  60,  55],
        [14, 13, 16, 24, 40,  57,  69,  56],
        [14, 17, 22, 29, 51,  87,  80,  62],
        [18, 22, 37, 56, 68,  109, 103, 77],
        [24, 36, 55, 64, 81,  104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],])
        y = split8x8_withquant(y, tablica)
        tablica = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        ])
        cr = split8x8_withquant(cr, tablica)
        cb = split8x8_withquant(cb, tablica)
    else:
        tablica = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ])
        y = split8x8_withquant(y, tablica)
        cr = split8x8_withquant(cr, tablica)
        cb = split8x8_withquant(cb, tablica)

    return [y, cr, cb]
def dekompresja(dane, typ='4:4:4', type = 'qy'):

    if (type == "qy"):
        tablica = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 36, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ])
        dane[0] = toarray_withquant(dane[0], tablica)
        tablica = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
        ])
        dane[1] = toarray_withquant(dane[1], tablica)
        dane[2] = toarray_withquant(dane[2], tablica)
    else:
        tablica = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ])
        dane[0] = toarray_withquant(dane[0], tablica)
        dane[1] = toarray_withquant(dane[1], tablica)
        dane[2] = toarray_withquant(dane[2], tablica)


    if (typ == "4:2:2"):
        dane[1] = np.repeat(dane[1], 2, axis=1)
        dane[2] = np.repeat(dane[2], 2, axis=1)


    dane[0] = dane[0] + 128
    dane[1] = dane[1] + 128
    dane[2] = dane[2] + 128
    dane[0] = np.clip(dane[0], 0, 255)
    dane = np.dstack([np.array(dane[0]), np.array(dane[1]), np.array(dane[2])]).astype(np.uint8)

    return rgbkom(dane)


if __name__ == '__main__':
    names = ['0004.png']
    for name in names:
        oryginał = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)

        fig, axs = plt.subplots(3, 3, sharey=True)
        fig.set_size_inches(9, 13)

        axs[0, 0].imshow(oryginał[0:256, 0:256])
        axs[1, 0].imshow(oryginał[300:556, 300:556])
        axs[2, 0].imshow(oryginał[500:756, 500:756])

        image = kompresja(dane=oryginał, typ="4:2:2", type="1")
        image = dekompresja(dane=image, typ="4:2:2", type="1")
        axs[0, 1].imshow(image[0:256, 0:256])
        axs[1, 1].imshow(image[300:556, 300:556])
        axs[2, 1].imshow(image[500:756, 500:756])

        image = kompresja(dane=oryginał, typ="4:4:4", type="qy")
        image = dekompresja(dane=image, typ="4:4:4", type="qy")
        axs[0, 2].imshow(image[0:256, 0:256])
        axs[1, 2].imshow(image[300:556, 300:556])
        axs[2, 2].imshow(image[500:756, 500:756])

        # axs[0, 0].imshow(oryginał[0:512, 0:512])
        # axs[1, 0].imshow(oryginał[1000:1512, 1000:1512])
        # axs[2, 0].imshow(oryginał[1500:2012, 1500:2012])

        # axs[0, 2].imshow(image[0:512, 0:512])
        # axs[1, 2].imshow(image[1500:2012, 1500:2012])
        # axs[2, 2].imshow(image[2000:2512, 2000:2512])


        plt.show()
