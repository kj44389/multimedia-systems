# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf





if __name__ == '__main__':
    img = plt.imread('pic1.png')
    img2 = plt.imread('pic2.jpg')
    print(img.dtype)
    print(img.shape)
    print(img2.dtype)
    print(img2.shape)

    #zmienna float 32 zakres <-3.4 * 10^38 , 3.4 * 10^38>
    #rozmiar: kolumn 1920 pikseli rzędów 1080 ilość kanałów (wartość > 0 jezeli obraz jest w kolorze)
    #dla plikow sa rozne shapy i typ danych. dla jpg mamy unsigned int 8 bitowy. Czyli jpg przyjmuje wartości naturalne na piksel

    # R = img[:,:,0]
    # plt.imshow(img)
    # plt.show()
    #
    # R = img[:,:,0]
    # plt.imshow(R)
    # plt.show()
    #
    # ################jakie wartosci maja przyjac vmin vmax !
    # plt.imshow(R, cmap=plt.cm.gray)
    # plt.show()
    Y1 = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]

    img_grey1 = np.asarray(img).copy()
    img_grey1[:, :, 0] = Y1
    img_grey1[:, :, 1] = Y1
    img_grey1[:, :, 2] = Y1

    Y2 = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]

    img_grey2 = np.asarray(img).copy()
    img_grey2[:, :, 0] = Y2
    img_grey2[:, :, 1] = Y2
    img_grey2[:, :, 2] = Y2

    print(Y2)
    plt.imshow(img_grey1)
    plt.show()
    plt.imshow(img_grey2)
    plt.show()

    for img in [img2, img]:
        plt.subplot(3, 3, 1)
        plt.imshow(img)

        Y1 = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        img_grey1 = np.asarray(img).copy()
        img_grey1[:, :, 0] = Y1
        img_grey1[:, :, 1] = Y1
        img_grey1[:, :, 2] = Y1
        plt.subplot(3, 3, 2)
        plt.imshow(img_grey1)

        Y2 = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]

        img_grey2 = np.asarray(img).copy()
        img_grey2[:, :, 0] = Y2
        img_grey2[:, :, 1] = Y2
        img_grey2[:, :, 2] = Y2
        plt.subplot(3, 3, 3)
        plt.imshow(img_grey2)

        plt.subplot(3, 3, 4)
        plt.imshow(img[:, :, 0])

        plt.subplot(3, 3, 5)
        plt.imshow(img[:, :, 1])

        plt.subplot(3, 3, 6)
        plt.imshow(img[:, :, 2])

        plt.subplot(3, 3, 7)
        img_cp = img.copy()
        img_cp[:, :, 1] = 0
        img_cp[:, :, 2] = 0
        plt.imshow(img_cp)

        plt.subplot(3, 3, 8)
        img_cp = img.copy()
        img_cp[:, :, 0] = 0
        img_cp[:, :, 2] = 0
        plt.imshow(img_cp)

        plt.subplot(3, 3, 9)
        img_cp = img.copy()
        img_cp[:, :, 0] = 0
        img_cp[:, :, 1] = 0
        plt.imshow(img_cp)

        plt.show()

    data, fs = sf.read('sound1.wav', dtype='float32')
    print(data.dtype)
    print(data.shape)

    x = np.arange(0, data.shape[0] / fs, 1 / fs)

    # sd.play(data, fs)
    # status = sd.wait()

    plt.subplot(2, 1, 1)
    plt.plot(x, data[:, 0])
    plt.subplot(2, 1, 2)
    plt.plot(x, data[:, 1])
    plt.show()

    # plt.plot(x,data)
    # plt.show()

    sf.write('sound_L.wav', data[:, 0], fs)
    sf.write('sound_R.wav', data[:, 1], fs)
    new_data = (data[:, 0] + data[:, 1]) / 2.0
    sf.write('sound_mix.wav', new_data, fs)
