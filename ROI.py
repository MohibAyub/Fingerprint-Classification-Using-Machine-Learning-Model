import pickle, cv2, glob
from PCA import PCA
import matplotlib.pyplot as plt
from enhance import image_enhance
from skimage.morphology import skeletonize, thin
import numpy as np
np.set_printoptions(threshold=np.nan)
import time
start = time.time()

def removedot(invertThin):
    temp0 = np.array(invertThin[:])
    temp0 = np.array(temp0)
    temp1 = temp0 / 255
    temp2 = np.array(temp1)
    temp3 = np.array(temp2)

    enhanced_img = np.array(temp0)
    filter0 = np.zeros((10, 10))
    W, H = temp0.shape[:2]
    filtersize = 6

    for i in range(W - filtersize):
        for j in range(H - filtersize):
            filter0 = temp1[i:i + filtersize, j:j + filtersize]

            flag = 0
            if sum(filter0[:, 0]) == 0:
                flag += 1
            if sum(filter0[:, filtersize - 1]) == 0:
                flag += 1
            if sum(filter0[0, :]) == 0:
                flag += 1
            if sum(filter0[filtersize - 1, :]) == 0:
                flag += 1
            if flag > 3:
                temp2[i:i + filtersize, j:j +
                                          filtersize] = np.zeros((filtersize, filtersize))

    return temp2


def get_descriptors(img, imageName, database):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = image_enhance.image_enhance(img)
    img = np.array(img, dtype=np.uint8)

    # Threshold
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Normalize to 0 and 1 range
    img[img == 255] = 1

    # Thinning
    skeleton = skeletonize(img)
    skeleton = np.array(skeleton, dtype=np.uint8)
    skeleton = removedot(skeleton)


    DescriptorList = list()
    print("0")
    for i in range(0, 399, 133):
        print("1")
        for j in range(0, 273, 91):
            blockImg = img[i:i + 133, j:j + 91]
            print("2")
            hog = cv2.HOGDescriptor()
            hog_des = hog.compute(blockImg)
            print("3")

            DescriptorList.append(hog_des)

    database.update({imageName: DescriptorList})
    # return (keypoints, des)
    # return database

def main():
    # database = dict()
    # images = glob.glob("train/*.jpg")
    # for image in images:
    #     img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    #     get_descriptors(img, image, database)

    # Pickling the descriptors
    # pickle_out = open("ROI_9_Block_data.pickle", "wb")
    # pickle.dump(database, pickle_out)
    # pickle_out.close()

    pickle_in = open("ROI_9_Block_data.pickle", "rb")
    dta = pickle.load(pickle_in)

    # print(dta['train\\3.jpg'][0])

    X_array = np.array((7, 9))
    X_array = [[0.14, 2.14, 4.14, 6.14, 8.14, 10.14, 12.14, 14.14, 16.14],
               [0.28, 2.28, 4.28, 6.28, 8.28, 10.28, 12.28, 14.28, 16.28],
               [0.42, 2.42, 4.42, 6.42, 8.42, 10.42, 12.42, 14.42, 16.42],
               [0.56, 2.56, 4.56, 6.56, 8.56, 10.56, 12.56, 14.56, 16.56],
               [0.7, 2.7, 4.7, 6.7, 8.7, 10.7, 12.7, 14.7, 16.7],
               [0.84, 2.84, 4.84, 6.84, 8.84, 10.84, 12.84, 14.84, 16.84],
               [0.98, 2.98, 4.98, 6.98, 8.98, 10.98, 12.98, 14.98, 16.98]]

    iter = 0

    for i in range(2, 9):
        single_error_point = list()
        compared = 'train\\' + str(i) + '.jpg'
        for j in range(0, 9):
            error_array = np.absolute(dta['train\\1.jpg'][j] - dta[compared][j])
            row, col = error_array.shape
            single_error_point.append(np.sum(error_array)/row)

        plt.scatter(X_array[iter], single_error_point)
        iter = iter+1


    plt.xlabel("block")
    plt.ylabel("error")
    # plt.show()
    print('It took', time.time() - start, 'seconds.')


if __name__ == "__main__":
    try:
        main()
    except:
        raise