import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import sys
from tqdm import tqdm
from operator import add
from functools import reduce


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])


def rle_encode(data):
    o_shape = data.shape
    print(o_shape)
    ret = np.empty(np.prod(o_shape)*2)
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
    return decode.astype(int)


def check(data1, data2):
    comparison = data1 == data2
    equal_arrays = comparison.all()
    return equal_arrays


def split4(image):
    half_split = np.array_split(image, 2)
    res = map(lambda x: np.array_split(x, 2, axis=1), half_split)
    return reduce(add, res)


def concatenate4(north_west, north_east, south_west, south_east):
    top = np.concatenate((north_west, north_east), axis=1)
    bottom = np.concatenate((south_west, south_east), axis=1)
    return np.concatenate((top, bottom), axis=0)


def calculate_mean(img):
    return np.mean(img, axis=(0, 1))


def checkEqual(myList):
    first = myList[0]
    return all((x == first).all() for x in myList)


class QuadTree:

    # def insert(self, img, level=0):
    #     self.level = level
    #     self.mean = calculate_mean(img).astype(int)
    #     self.resolution = (img.shape[0], img.shape[1])
    #     self.final = True

    #     if not checkEqual(img):
    #         split_img = split4(img)

    #         self.final = False
    #         self.north_west = QuadTree().insert(split_img[0], level + 1)
    #         self.north_east = QuadTree().insert(split_img[1], level + 1)
    #         self.south_west = QuadTree().insert(split_img[2], level + 1)
    #         self.south_east = QuadTree().insert(split_img[3], level + 1)

    #     return self

    # def get_image(self, level):
    #     if(self.final or self.level == level):
    #         return np.tile(self.mean, (self.resolution[0], self.resolution[1], 1))

    #     return concatenate4(
    #         self.north_west.get_image(level),
    #         self.north_east.get_image(level),
    #         self.south_west.get_image(level),
    #         self.south_east.get_image(level))
    def insert(self, img, level=0):
        self.level = level
        self.mean = calculate_mean(img).astype(int)
        self.resolution = (img.shape[0], img.shape[1])
        self.final = True
        self.img = None
        if img.size < 32:
            self.img = img
        if not checkEqual(img):
            split_img = split4(img)

            self.final = False
            self.north_west = QuadTree().insert(split_img[0], level + 1)
            self.north_east = QuadTree().insert(split_img[1], level + 1)
            self.south_west = QuadTree().insert(split_img[2], level + 1)
            self.south_east = QuadTree().insert(split_img[3], level + 1)

        return self

    def get_image(self, level):
        if(self.final or self.level == level):
            return np.tile(self.mean, (self.resolution[0], self.resolution[1], 1))
        if(self.img is not None):
            return self.img
        return concatenate4(
            self.north_west.get_image(level),
            self.north_east.get_image(level),
            self.south_west.get_image(level),
            self.south_east.get_image(level))


original = plt.imread('images.jpg')
#quadtree = QuadTree().insert(original)
#ob = quadtree.get_image(100)

# plt.imshow(quadtree.get_image(1))
# plt.show()
# plt.imshow(quadtree.get_image(3))
# plt.show()
# plt.imshow(quadtree.get_image(7))
# plt.show()
# plt.imshow(quadtree.get_image(10))
# plt.show()


#original = plt.imread('Skan_20210311.jpg')
# plt.imshow(original)
# plt.title('cos tam')
# plt.show()
#print("size original", sys.getsizeof(original))

com = rle_encode(original)
# print(com[0])
# print(com[1])
# print("1 ", original.shape)
# print("2 ", com[0])
# print("3 ", com[1].shape)
print("wielkosc oryginal", original.size)
print("wielkosc rle_en", com[1].size)
print("kompresja ", com[1].size / original.size)
print("size encoded", sys.getsizeof(com))

dec = rle_decode(com)

#print("size decoded", sys.getsizeof(dec))

#print(check(original, dec))
# print(original)
# print(dec)
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('before and after')
ax1.imshow(original)
ax2.imshow(dec)
plt.show()
