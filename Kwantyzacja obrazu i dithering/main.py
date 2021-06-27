import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

import cv2



def colorfit(pixel, paleta):
    differences = []
    for color in paleta:
        differences.append(np.linalg.norm(color - pixel))
    index = np.argmin(differences)
    return paleta[index]

def bitsreduce(image, paleta):
    for w in range(image.shape[0]):
        for h in range(image.shape[1]):
            image[w,h] = colorfit(image[w,h], paleta)

    return image

def randomdithering(image):

    for w in range(image.shape[0]):
        for h in range(image.shape[1]):
            random_value = np.random.rand()
            img_copy = np.copy(image[w,h])
            if(random_value >= img_copy):
                image[w,h] = 0
            else:
                image[w,h] = 1
    return image

def organizeddithering(paleta, image , n, m, greyscale=0):
    image = np.copy(image)
    if(greyscale == 1):
        if m == 1:
            m1 = np.array([[0.0,2.0],[3.0,1.0]])
            m1 = ((1/4) * (m1 + 1)) - 0.5
            #print(m1)

            for w in range(image.shape[0]):
                for h in range(image.shape[1]):
                        m1_w = w%2
                        m1_h = h%2
                        fit = colorfit((image[w,h] + m1[m1_w,m1_h]),paleta)
                        #fit = colorfit(dim/255,paleta)
                        #image[w, h] = np.array(((dim / 255) + m1[m1_w, m1_h])) * 255
                        image[w,h] = fit

        else:
            m2 = np.array([[0.0, 8.0,2.0,10.0], [12.0,4.0,14.0,6.0],[3.0,11.0,1.0,9.0],[15.0,7.0,13.0,5.0]])
            m2= ((1 / 16) * (m2 + 1)) - 0.5
            for w in range(image.shape[0]):
                for h in range(image.shape[1]):
                    m2_w = w % 4
                    m2_h = h % 4
                    #print(colorfit((image[w, h] + m2[m2_w, m2_h]), paleta))
                    fit = colorfit((image[w, h] + m2[m2_w, m2_h]), paleta)
                    image[w, h] = fit
    else:
        if m == 1:
            m1 = np.array([[0, 2], [3, 1]])
            m1 = (1 / (2 ** n)) * (m1 + 1) - 0.5
            # print(m1)

            for w in range(image.shape[0]):
                for h in range(image.shape[1]):
                    m1_w = w % 2
                    m1_h = h % 2
                    fit = colorfit((image[w, h] + m1[m1_w, m1_h]), paleta)
                    # fit = colorfit(dim/255,paleta)
                    # image[w, h] = np.array(((dim / 255) + m1[m1_w, m1_h])) * 255
                    image[w, h] = fit

        else:
            m2 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
            m2 = ((1 / (4 ** n)) * (m2 + 1)) - 0.5

            for w in range(image.shape[0]):
                for h in range(image.shape[1]):
                    m2_w = w % 4
                    m2_h = h % 4
                    #print(colorfit((image[w, h] + m2[m2_w, m2_h]), paleta))
                    fit = colorfit((image[w, h] + m2[m2_w, m2_h]), paleta)
                    image[w, h] = fit



    return image

def floyddithering(image, paleta, greyscale = 0):
    image2 = np.copy(image)

    #print(image2)
    for w in range(image.shape[0]):
        for h in range(image.shape[1]):

            oldpixel = np.copy(image2[w,h])
            newpixel =  colorfit(oldpixel, paleta)
            if(greyscale == 1):
                image2[w,h] = newpixel
                quant_error = oldpixel - newpixel

                if(w<image2.shape[0]-1):

                    image2[w + 1,h] = (image2[w + 1,h] + (quant_error * (7 / 16)))

                if(w>0 and h < image2.shape[1]-1):
                    image2[w -1,h+1] = (image2[w -1,h+1] + (quant_error * (3 / 16)))

                if(h<image2.shape[1]-1):
                    image2[w,h+1] = (image2[w,h+1] + (quant_error * (5 / 16)))

                if(w < image2.shape[0]-1 and h < image2.shape[1]-1):
                    image2[w + 1,h+1] = (image2[w + 1,h+1] + (quant_error * (1 / 16)))
            else:
                image2[w, h] = newpixel
                quant_error = oldpixel - newpixel

                if (w < image2.shape[0] - 1):
                    image2[w + 1, h] = (np.array(image2[w + 1, h]) + (quant_error * (7 / 16)))

                if (w > 0 and h < image2.shape[1] - 1):
                    image2[w - 1, h + 1] = (np.array(image2[w - 1, h + 1]) + (quant_error * (3 / 16)))

                if (h < image2.shape[1] - 1):
                    image2[w, h + 1] = (np.array(image2[w, h + 1]) + (quant_error * (5 / 16)))

                if (w < image2.shape[0] - 1 and h < image2.shape[1] - 1):
                    image2[w + 1, h + 1] = (np.array(image2[w + 1, h + 1]) + (quant_error * (1 / 16)))

    return image2

if __name__ == '__main__':
    paleta = [[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 1., 1.], [1., 0., 0.], [1., 0., 1.], [1., 1., 0.],
              [1., 1., 1.]]
    paleta2 = [[0,0,0],[0,1,1],[0,0,1],[1,0,1],[0,0.5,0],[0.5,0.5,0.5],[0,1,0],[0.5,0,0],[0,0,0.5],[0.5,0.5,0],[0.5,0,0.5],[1,0,0],[0.75,0.75,0.75],[0,0.5,0.5],[1,1,1],[1,1,0]]
    paleta3 = [[200/255,199/255,189/255],
               [86/255,46/255,41/255],
               [34/255,49/255,106/255],
               [97/255,103/255,134/255],
               [128/255,99/255,67/255],
               [133/255,114/255,105/255],
               [136/255,141/255,157/255],
               [163/255,152/255,142/255],
               [60 / 255, 92 / 255, 180 / 255]]


    image = cv2.imread('0016.jpg')
    image = np.array(image).astype('float32')
    image = image/255

    greyscale = 0
    if(greyscale == 1):
        paleta = np.linspace(0, 1, 2 ** 1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    reduced_image = image

    reduced_image2 = bitsreduce(np.copy(image),paleta)
    #r_dither = randomdithering(np.copy(reduced_image))
    or_dither = organizeddithering(paleta, np.copy(reduced_image), 4, 2, greyscale)
    flo_dither = floyddithering(np.copy(reduced_image),paleta, greyscale)

    cv2.imwrite('D:\SEMESTR_5\SM\lab4\spraw\o1reduced.png', reduced_image2 * 255);
    #cv2.imwrite('D:\SEMESTR_5\SM\lab4\spraw\o1r.png',r_dither*255);
    cv2.imwrite('D:\SEMESTR_5\SM\lab4\spraw\o1rg.png', or_dither*255);
    cv2.imwrite('D:\SEMESTR_5\SM\lab4\spraw\o1flo.png', flo_dither*255);


    #rd_image = randomdithering(np.copy(rd_image))

    # print(rd_image.shape)
    # plt.imshow(rd_image)
    # plt.show()

