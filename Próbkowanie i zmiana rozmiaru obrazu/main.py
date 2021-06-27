from tkinter import *
from tkinter import ttk, filedialog
import numpy as np
import io
import cv2
import matplotlib
from PIL import Image
from PIL import ImageTk
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from skimage import io


def openimage(name='0001.jpg'):
    image_read = io.imread(name)
    new_width = int(image_read.shape[0] * size)
    new_height = int(image_read.shape[1] * size)
    new_img = np.zeros(shape=[new_width, new_height, image_read.shape[2]])
    w_ratio = image_read.shape[0] / new_width
    h_ratio = image_read.shape[1] / new_height
    return [image_read, new_img, new_width, new_height, w_ratio, h_ratio]

def NN(new_width, new_height, new_img, image_read, w_ratio, h_ratio):
    for W in range(new_width):
        for H in range(new_height):
            tmp_w = int(W * w_ratio)
            tmp_h = int(H * h_ratio)
            if (tmp_h >= image_read.shape[1]):
                tmp_h = tmp_h - 1
            if (tmp_w >= image_read.shape[0]):
                tmp_w = tmp_w - 1
            new_img[W, H] = image_read[tmp_w, tmp_h]
    return new_img

def bilinear(new_width, new_height, new_img, image_read, w_ratio, h_ratio):
    for W in range(new_width):
        for H in range(new_height):
                tmp_w = W * w_ratio
                tmp_h = H * h_ratio
                x = tmp_w - int(tmp_w)
                y = tmp_h - int(tmp_h)
                tmp_w = int(tmp_w)
                tmp_h = int(tmp_h)
                if((tmp_w == image_read.shape[0]-1) and (tmp_h == image_read.shape[1]-1)):
                    new_img[W, H] = (image_read[tmp_w, tmp_h] * (1 - x) * (1 - y)
                                     + image_read[tmp_w, tmp_h] * x * (1 - y)
                                     + image_read[tmp_w, tmp_h] * (1 - x) * y
                                     + image_read[tmp_w, tmp_h] * x * y)
                elif(tmp_w == image_read.shape[0]-1):
                    new_img[W, H] = (image_read[tmp_w, tmp_h] * (1 - x) * (1 - y)
                                     + image_read[tmp_w,tmp_h] * x * (1-y)
                                     + image_read[tmp_w, tmp_h + 1] * (1 - x) * y
                                     + image_read[tmp_w, tmp_h + 1] * x * y)
                elif(tmp_h == image_read.shape[1]-1):
                    new_img[W, H] = (image_read[tmp_w, tmp_h] * (1 - x) * (1 - y)
                                     + image_read[tmp_w + 1, tmp_h] * x * (1 - y)
                                     + image_read[tmp_w,tmp_h] * ( 1 - x ) * y
                                     + image_read[tmp_w + 1, tmp_h] * x * y)
                else:
                    new_img[W, H] = (image_read[tmp_w, tmp_h] * (1 - x) * (1 - y)
                                     + image_read[tmp_w+1, tmp_h] * x * (1 - y)
                                     + image_read[tmp_w, tmp_h+1] * (1 - x) * y
                                     + image_read[tmp_w+1, tmp_h+1] * x * y)
                # else:
                #     print(tmp_w,tmp_h)
                #     if (tmp_w == image_read.shape[0]-1):
                #         tmp_w = image_read.shape[0] - 2
                #     if (tmp_h == image_read.shape[1]-1):
                #         tmp_h = image_read.shape[1] - 2
                #     new_img[W, H] = (image_read[tmp_w, tmp_h] * (1 - x) * (1 - y)
                #                      + image_read[tmp_w + 1, tmp_h] * x * (1 - y)
                #                      + image_read[tmp_w, tmp_h + 1] * (1 - x) * y
                #                      + image_read[tmp_w + 1, tmp_h + 1] * x * y)
    return new_img

def zmniejszanieSM(new_width, new_height, image_read, new_img_s, new_img_m, w_ratio, h_ratio):
    for W in range(new_width):
        for H in range(new_height):
            for dim in range(image_read.shape[2]):
                count = 0
                tmp_w = int(W * w_ratio)
                tmp_h = int(H * h_ratio)
                sumx = []
                if (tmp_w == 0 and tmp_h == 0):  # lg
                    count = 4
                    sumx = [image_read[tmp_w, tmp_h, dim]
                        , image_read[tmp_w, tmp_h + 1, dim]
                        , image_read[tmp_w + 1, tmp_h, dim]
                        , image_read[tmp_w + 1, tmp_h + 1, dim]]
                elif (tmp_w == 0 and tmp_h == image_read.shape[1] - 1):  # pg
                    count = 4
                    sumx = [image_read[tmp_w, tmp_h, dim]
                        , image_read[tmp_w, tmp_h - 1, dim]
                        , image_read[tmp_w - 1, tmp_h, dim]
                        , image_read[tmp_w - 1, tmp_h - 1, dim]]
                elif (tmp_w == image_read.shape[0] - 1 and tmp_h == 0):  # ld
                    count = 4
                    sumx = [image_read[tmp_w, tmp_h, dim]
                        , image_read[tmp_w, tmp_h + 1, dim]
                        , image_read[tmp_w - 1, tmp_h, dim]
                        , image_read[tmp_w - 1, tmp_h + 1, dim]]
                elif (tmp_w == image_read.shape[0] - 1 and tmp_h == image_read.shape[1] - 1):  # pd
                    count = 4
                    sumx = [image_read[tmp_w, tmp_h, dim]
                        , image_read[tmp_w, tmp_h - 1, dim]
                        , image_read[tmp_w - 1, tmp_h, dim]
                        , image_read[tmp_w - 1, tmp_h - 1, dim]]
                elif (tmp_w == 0 and tmp_h > 0 and tmp_h < image_read.shape[1] - 1):  # gorna linia
                    count = 6
                    sumx = [image_read[tmp_w, tmp_h, dim]
                        , image_read[tmp_w, tmp_h - 1, dim]
                        , image_read[tmp_w + 1, tmp_h - 1, dim]
                        , image_read[tmp_w + 1, tmp_h, dim]
                        , image_read[tmp_w + 1, tmp_h + 1, dim]
                        , image_read[tmp_w, tmp_h + 1, dim]]
                elif (tmp_w == image_read.shape[0] - 1 and tmp_h > 0 and tmp_h < image_read.shape[
                    1] - 1):  # dolna linia
                    count = 6
                    sumx = [image_read[tmp_w, tmp_h, dim]
                        , image_read[tmp_w, tmp_h - 1, dim]
                        , image_read[tmp_w - 1, tmp_h - 1, dim]
                        , image_read[tmp_w - 1, tmp_h, dim]
                        , image_read[tmp_w - 1, tmp_h + 1, dim]
                        , image_read[tmp_w, tmp_h + 1, dim]]
                elif (tmp_w == image_read.shape[0] - 1 and tmp_h == image_read.shape[1] - 1):  # lewa sciana
                    count = 6
                    sumx = [image_read[tmp_w, tmp_h, dim]
                        , image_read[tmp_w - 1, tmp_h, dim]
                        , image_read[tmp_w - 1, tmp_h + 1, dim]
                        , image_read[tmp_w, tmp_h + 1, dim]
                        , image_read[tmp_w + 1, tmp_h + 1, dim]
                        , image_read[tmp_w + 1, tmp_h, dim]]
                elif (tmp_w == image_read.shape[0] - 1 and tmp_h == image_read.shape[1] - 1):  # prawa sciana
                    count = 6
                    sumx = [image_read[tmp_w, tmp_h, dim]
                        , image_read[tmp_w - 1, tmp_h, dim]
                        , image_read[tmp_w - 1, tmp_h - 1, dim]
                        , image_read[tmp_w, tmp_h - 1, dim]
                        , image_read[tmp_w + 1, tmp_h - 1, dim]
                        , image_read[tmp_w + 1, tmp_h, dim]]
                else:
                    count = 9
                    sumx = [image_read[tmp_w, tmp_h, dim]
                        , image_read[tmp_w - 1, tmp_h, dim]
                        , image_read[tmp_w - 1, tmp_h + 1, dim]
                        , image_read[tmp_w, tmp_h + 1, dim]
                        , image_read[tmp_w + 1, tmp_h + 1, dim]
                        , image_read[tmp_w + 1, tmp_h, dim]
                        , image_read[tmp_w + 1, tmp_h - 1, dim]
                        , image_read[tmp_w, tmp_h - 1, dim]
                        , image_read[tmp_w - 1, tmp_h - 1, dim]]

                new_img_s[W, H, dim] = np.mean(np.asarray(sumx))
                new_img_m[W, H, dim] = np.median(np.asarray(sumx))
    return new_img_s, new_img_m

def sredniawazona(new_width, new_height, image_read, new_img_sw, w_ratio, h_ratio):
    for W in range(new_width):
        for H in range(new_height):
            for dim in range(image_read.shape[2]):
                weight = 0
                tmp_w = int(W * w_ratio)
                tmp_h = int(H * h_ratio)
                sumx = []
                if (tmp_w == 0 and tmp_h == 0):  # lg

                    sumx = [image_read[tmp_w, tmp_h, dim] * 0.5
                        , image_read[tmp_w, tmp_h + 1, dim] * 0.1
                        , image_read[tmp_w + 1, tmp_h, dim] * 0.1
                        , image_read[tmp_w + 1, tmp_h + 1, dim] * 0.2]
                    weight = 0.9
                elif (tmp_w == 0 and tmp_h == image_read.shape[1] - 1):  # pg
                    count = 4
                    sumx = [image_read[tmp_w, tmp_h, dim] * 0.5
                        , image_read[tmp_w, tmp_h - 1, dim] * 0.1
                        , image_read[tmp_w - 1, tmp_h, dim] * 0.1
                        , image_read[tmp_w - 1, tmp_h - 1, dim] * 0.2]
                    weight = 0.9
                elif (tmp_w == image_read.shape[0] - 1 and tmp_h == 0):  # ld
                    count = 4
                    sumx = [image_read[tmp_w, tmp_h, dim] * 0.5
                        , image_read[tmp_w, tmp_h + 1, dim] * 0.1
                        , image_read[tmp_w - 1, tmp_h, dim] * 0.1
                        , image_read[tmp_w - 1, tmp_h + 1, dim] * 0.2]
                    weight = 0.9
                elif (tmp_w == image_read.shape[0] - 1 and tmp_h == image_read.shape[1] - 1):  # pd
                    count = 4
                    sumx = [image_read[tmp_w, tmp_h, dim] * 0.5
                        , image_read[tmp_w, tmp_h - 1, dim] * 0.1
                        , image_read[tmp_w - 1, tmp_h, dim] * 0.1
                        , image_read[tmp_w - 1, tmp_h - 1, dim] * 0.2]
                    weight = 0.9
                elif (tmp_w == 0 and tmp_h > 0 and tmp_h < image_read.shape[1] - 1):  # gorna linia
                    count = 6
                    sumx = [image_read[tmp_w, tmp_h, dim] * 0.5
                        , image_read[tmp_w, tmp_h - 1, dim] * 0.1
                        , image_read[tmp_w + 1, tmp_h - 1, dim] * 0.2
                        , image_read[tmp_w + 1, tmp_h, dim] * 0.1
                        , image_read[tmp_w + 1, tmp_h + 1, dim] * 0.2
                        , image_read[tmp_w, tmp_h + 1, dim] * 0.1]
                    weight = 1.2
                elif (tmp_w == image_read.shape[0] - 1 and tmp_h > 0 and tmp_h < image_read.shape[
                    1] - 1):  # dolna linia
                    count = 6
                    sumx = [image_read[tmp_w, tmp_h, dim] * 0.5
                        , image_read[tmp_w, tmp_h - 1, dim] * 0.1
                        , image_read[tmp_w - 1, tmp_h - 1, dim] * 0.2
                        , image_read[tmp_w - 1, tmp_h, dim] * 0.1
                        , image_read[tmp_w - 1, tmp_h + 1, dim] * 0.2
                        , image_read[tmp_w, tmp_h + 1, dim] * 0.1]
                    weight = 1.2
                elif (tmp_w == image_read.shape[0] - 1 and tmp_h == image_read.shape[1] - 1):  # lewa sciana
                    count = 6
                    sumx = [image_read[tmp_w, tmp_h, dim] * 0.5
                        , image_read[tmp_w - 1, tmp_h, dim] * 0.1
                        , image_read[tmp_w - 1, tmp_h + 1, dim] * 0.2
                        , image_read[tmp_w, tmp_h + 1, dim] * 0.1
                        , image_read[tmp_w + 1, tmp_h + 1, dim] * 0.2
                        , image_read[tmp_w + 1, tmp_h, dim] * 0.1]
                    weight = 1.2
                elif (tmp_w == image_read.shape[0] - 1 and tmp_h == image_read.shape[1] - 1):  # prawa sciana
                    count = 6
                    sumx = [image_read[tmp_w, tmp_h, dim] * 0.5
                        , image_read[tmp_w - 1, tmp_h, dim] * 0.1
                        , image_read[tmp_w - 1, tmp_h - 1, dim] * 0.2
                        , image_read[tmp_w, tmp_h - 1, dim] * 0.1
                        , image_read[tmp_w + 1, tmp_h - 1, dim] * 0.2
                        , image_read[tmp_w + 1, tmp_h, dim] * 0.1]
                    weight = 1.2
                else:
                    count = 9
                    sumx = [image_read[tmp_w, tmp_h, dim] * 0.5
                        , image_read[tmp_w - 1, tmp_h, dim] * 0.1
                        , image_read[tmp_w - 1, tmp_h + 1, dim] * 0.2
                        , image_read[tmp_w, tmp_h + 1, dim] * 0.1
                        , image_read[tmp_w + 1, tmp_h + 1, dim] * 0.2
                        , image_read[tmp_w + 1, tmp_h, dim] * 0.1
                        , image_read[tmp_w + 1, tmp_h - 1, dim] * 0.2
                        , image_read[tmp_w, tmp_h - 1, dim] * 0.1
                        , image_read[tmp_w - 1, tmp_h - 1, dim] * 0.2]
                    weight = 1.7

                new_img_sw[W, H, dim] = (np.sum(np.asarray(sumx))) / weight
    return new_img_sw

def showimg(sub1, sub2, sub3, desc, img):
    plt.subplot(sub1, sub2, sub3)
    plt.title(desc)
    plt.imshow(img.astype("int"))

if __name__ == '__main__':

    size = 0.05
    image_read, new_img, new_width, new_height, w_ratio, h_ratio = openimage('0006.jpg')

    print("image size : ", image_read.shape)
    print("w_ratio:", w_ratio,", h_ratio:",h_ratio)
    print("Resized image size : ", new_img.shape)


    print('skalowanie: ')
    print("NN")

    new_img = NN(new_width, new_height, new_img, image_read, w_ratio, h_ratio)

    showimg(2,3,1,"oryginal", image_read)
    showimg(2, 3, 2, "NN", new_img)

    print("bilinear")

    new_img = np.zeros(shape=[new_width, new_height, image_read.shape[2]])
    new_img = bilinear(new_width, new_height, new_img, image_read, w_ratio, h_ratio)
    showimg(2,3,3,"bilinear", new_img)

    if size < 1.0:
        print('zmniejszanie')
        #inicjalizacja tablic do obrazka z sredniej i mediany
        new_img_s = np.zeros(shape=[new_width, new_height, image_read.shape[2]])
        new_img_m = np.zeros(shape=[new_width, new_height, image_read.shape[2]])
        new_img_s, new_img_m = zmniejszanieSM(new_width,new_height,image_read, new_img_s, new_img_m, w_ratio, h_ratio)

        # inicjalizacja tablicy do obrazka z sredniej wazonej
        new_img_sw = np.zeros(shape=[new_width, new_height, image_read.shape[2]])

        showimg(2,3,4,'srednia',new_img_s)
        showimg(2,3,5,'mediana',new_img_m)

        new_img_sw = sredniawazona(new_width,new_height,image_read,new_img_sw, w_ratio, h_ratio)

        #wykrywanie krawedzi
        tmp_img = cv2.Canny(np.uint8(new_img_sw),1000,1000)

        showimg(2,3,6,'wagi',new_img_sw)
        #showimg(3,3,6,'krawedzie',tmp_img)

    plt.show()