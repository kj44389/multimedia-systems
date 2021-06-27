

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import scipy.fftpack


if __name__ == '__main__':

    ##############################################################################
    ######   Konfiguracja       ##################################################
    ##############################################################################

    kat='.\\'                               # katalog z plikami wideo
    plik="clip_1.mp4"                       # nazwa pliku
    ile=15                                # ile klatek odtworzyc? <0 - calosc
    key_frame_counter=3                     # co ktora klatka ma byc kluczowa i nie podlegac kompresji
    plot_frames=np.array([15])           # automatycznie wyrysuj wykresy
    auto_pause_frames=np.array([15])        # automatycznie zapauzuj dla klatki i wywietl wykres
    typ_samplingu="4:2:0"                     # parametry dla chorma subsamplingu
    sub_name = '4_2_0'
    wyswietlaj_kaltki=False                  # czy program ma wyswietlac kolejene klatki
    stala = 1

    typ_loop = ['4:4:4', '4:4:0', '4:2:0', '4:2:2', '4:1:1', '4:1:0']
    name_loop = ['4_4_4', '4_4_0', '4_2_0', '4_2_2', '4_1_1', '4_1_0']
    stale = [1,2,4,8]

    for x in [0,1,2,3,4,5]:
        for n in stale:
            print(x)
            typ_samplingu = typ_loop[x]
            sub_name = name_loop[x]
            stala = n
            ##############################################################################
            ####     Kompresja i dekompresja    ##########################################
            ##############################################################################
            class data:
                def init(self):
                    self.Y=None
                    self.Cb=None
                    self.Cr=None


            def subsampling(dane, typ):
                if (typ == "4:2:2"):
                    dane = np.delete(dane, np.arange(1, dane.shape[1], 2), axis=1)


                elif (typ == "4:4:0"):

                    dane = np.delete(dane, np.arange(1, dane.shape[0], 2), axis=0)


                elif (typ == "4:2:0"):

                    dane = np.delete(dane, np.arange(1, dane.shape[0], 2), axis=0)
                    dane = np.delete(dane, np.arange(1, dane.shape[1], 2), axis=1)

                elif (typ == "4:1:0"):
                    dane = np.delete(dane, np.arange(1, dane.shape[0], 2), axis=0)
                    dane = np.delete(dane, np.arange(1, dane.shape[1], 4), axis=1)
                    dane = np.delete(dane, np.arange(1, dane.shape[1], 3), axis=1)
                    dane = np.delete(dane, np.arange(1, dane.shape[1], 2), axis=1)


                elif (typ == "4:1:1"):
                    dane = np.delete(dane, np.arange(1, dane.shape[1], 4), axis=1)
                    dane = np.delete(dane, np.arange(1, dane.shape[1], 3), axis=1)
                    dane = np.delete(dane, np.arange(1, dane.shape[1], 2), axis=1)

                return dane
            def resampling(dane, typ):
                if (typ == "4:2:2"):

                    dane = np.repeat(dane, 2, axis=1)


                elif (typ == "4:4:0"):

                    dane = np.repeat(dane, 2, axis=0)


                elif (typ == "4:2:0"):

                    dane = np.repeat(dane, 2, axis=1)
                    dane = np.repeat(dane, 2, axis=0)


                elif (typ == "4:1:0"):

                    dane = np.repeat(dane, 4, axis=1)
                    dane = np.repeat(dane, 2, axis=0)


                elif (typ == "4:1:1"):

                    dane = np.repeat(dane, 4, axis=1)

                return dane
            def rle_encode(data):
                o_shape = data.shape
                ret = np.empty(np.prod(o_shape) * 2)
                data2 = data.flatten()

                prev = data2[0]
                count = 0
                flaga = 0
                tmp = 0
                # for el in data:
                for i in tqdm(range(len(data2))):
                    if data2[i] != prev:
                        ret[tmp] = count
                        ret[tmp + 1] = prev
                        # ret = np.append(ret, int(count))
                        # ret = np.append(ret, prev)
                        # print(count, prev)
                        tmp = tmp + 2
                        count = 1
                        prev = data2[i]
                        flaga = 0
                    else:
                        count += 1
                        flaga = 1
                if flaga == 1:
                    ret[tmp] = count
                    ret[tmp + 1] = prev
                    tmp = tmp + 2
                # print("halo: ", tmp)
                return np.array([o_shape, ret[0:tmp]])
                # return ret
            def rle_decode(data):
                decode = np.empty(np.prod(data[0]))
                arr = data[1]

                tmp = 0
                for i in tqdm(range(len(arr))):
                    if i % 2 == 0:
                        for j in range(int(arr[i])):
                            # decode = np.append(decode, arr[i + 1])
                            decode[tmp] = arr[i + 1]
                            tmp = tmp + 1

                decode = np.reshape(decode, data[0])
                return decode
            def compress(Y,Cb,Cr, key_frame_Y, key_frame_Cb, key_frame_Cr, stala):
                data.Cr = subsampling(Cr, typ_samplingu)
                data.Cb = subsampling(Cb, typ_samplingu)

                key_f_Cr = subsampling(key_frame_Cr, typ_samplingu)
                key_f_Cb = subsampling(key_frame_Cb, typ_samplingu)

                data.Y = (Y - key_frame_Y) // stala
                data.Cb = (data.Cb - key_f_Cb) // stala
                data.Cr = (data.Cr - key_f_Cr) //stala
                return data
            def decompress(data,  key_frame_Y, key_frame_Cb, key_frame_Cr , stala):
                key_frame_Cr = subsampling(key_frame_Cr, typ_samplingu)
                key_frame_Cb = subsampling(key_frame_Cb, typ_samplingu)

                data.Y = data.Y * stala + key_frame_Y
                data.Cr = data.Cr * stala + key_frame_Cr
                data.Cb = data.Cb * stala + key_frame_Cb

                data.Cr = resampling(data.Cr, typ_samplingu)
                data.Cb = resampling(data.Cb, typ_samplingu)

                frame = np.dstack([data.Y,data.Cr,data.Cb]).astype(np.uint8)
                return frame



            ##############################################################################
            ####     Głowna petla programu      ##########################################
            ##############################################################################

            cap = cv2.VideoCapture(kat+'\\'+plik)

            if ile<0:
                ile=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            cv2.namedWindow('Normal Frame')
            cv2.namedWindow('Decompressed Frame')

            compression_information=np.zeros((3,ile))

            for i in range(ile):
                ret, frame = cap.read()
                if wyswietlaj_kaltki:
                    cv2.imshow('Normal Frame',frame)
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
                if (i % key_frame_counter)==0: # pobieranie klatek kluczowych
                    key_frame=frame
                    cY=frame[:,:,0]
                    cCb= subsampling(frame[:,:,2],typ_samplingu)
                    cCr= subsampling(frame[:,:,1],typ_samplingu)

                    cY = rle_encode(cY)
                    cCb = rle_encode(cCb)
                    cCr = rle_encode(cCr)

                    data.Y = rle_decode(cY)
                    data.Cb = rle_decode(cCb)
                    data.Cr = rle_decode(cCr)
                    data.cY = cY
                    data.Cb = resampling(data.Cb, typ_samplingu)
                    data.Cr = resampling(data.Cr, typ_samplingu)

                    frame = np.dstack([ data.cY,data.Cr,data.Cb]).astype(np.uint8)

                    d_frame=frame

                else: # kompresja
                    cdata=compress(frame[:,:,0],frame[:,:,2],frame[:,:,1], key_frame[:,:,0], key_frame[:,:,2], key_frame[:,:,1], stala)


                    cdata.Y = rle_encode(cdata.Y)
                    cdata.Cb = rle_encode(cdata.Cb)
                    cdata.Cr = rle_encode(cdata.Cr)

                    cY = cdata.Y
                    cCb = cdata.Cb
                    cCr = cdata.Cr

                    cdata.Y = rle_decode(cdata.Y)
                    cdata.Cb = rle_decode(cdata.Cb)
                    cdata.Cr = rle_decode(cdata.Cr)



                    d_frame= decompress(cdata, key_frame[:,:,0], key_frame[:,:,2], key_frame[:,:,1], stala)


                compression_information[0,i]= (frame[:,:,0].size - cY[1].size)/frame[:,:,0].size
                compression_information[1,i]= (frame[:,:,1].size - cCb[1].size)/frame[:,:,1].size
                compression_information[2,i]= (frame[:,:,2].size - cCr[1].size)/frame[:,:,2].size
                if wyswietlaj_kaltki:
                    cv2.imshow('Decompressed Frame',cv2.cvtColor(d_frame,cv2.COLOR_YCrCb2BGR))

                if (i % key_frame_counter == 0):
                    pass
                    # if not os.path.exists('sprawko1/kf_'+str(i)+'_'+str(stala)):
                    #     os.mkdir('sprawko1/kf_'+str(i)+'_'+str(stala))
                    # cv2.imwrite('sprawko1/kf_'+str(i)+'_'+str(stala)+'/decompressed_frame_typ_' + sub_name + '_kf_' + str(i) + '_stala_'+str(stala)+'.jpg',
                    #             cv2.cvtColor(d_frame, cv2.COLOR_YCrCb2BGR))

                if np.any(plot_frames==i): # rysuj wykresy
                    # bardzo słaby i sztuczny przyklad wykrozystania tej opcji
                    fig, axs = plt.subplots(1, 3 , sharey=True   )
                    fig.set_size_inches(16,5)
                    axs[0].imshow(frame)
                    axs[2].imshow(d_frame)

                    diff=frame-d_frame
                    print(np.sum(diff[diff>0]))
                    # print("______________")
                    # print(frame)
                    # print("______________")
                    # print(d_frame)
                    # print(np.min(diff),np.max(diff))
                    axs[1].imshow(diff,vmin=np.min(diff),vmax=np.max(diff))

                if np.any(auto_pause_frames==i):
                    cv2.waitKey(-1) #wait until any key is pressed

                k = cv2.waitKey(1) & 0xff

                if k==ord('q'):
                    break
                elif k == ord('p'):
                    cv2.waitKey(-1) #wait until any key is pressed

            plt.figure()
            plt.title(typ_samplingu+'_'+stala)
            plt.plot(np.arange(0,ile),compression_information[0,:]*100)
            plt.plot(np.arange(0,ile),compression_information[1,:]*100)
            plt.plot(np.arange(0,ile),compression_information[2,:]*100)
            plt.show()