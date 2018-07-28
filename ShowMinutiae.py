import cv2
import pickle, os
import glob
import numpy as np
import matplotlib.pyplot as plt
from enhance import image_enhance
from skimage.morphology import skeletonize, thin

def removedot(invertThin):
    temp0 = np.array(invertThin[:])
    temp0 = np.array(temp0)
    temp1 = temp0/255
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


def get_descriptors(img, imageName, database, KeypointsLength):
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

    # Creating Block Size of 144 x 96 total of 8 blocks for an image and then generating descriptors and keypoints
    # Storing these 4 descriptors and 8 keypoints of one image in a Mainlist and key(image name) is associated to represent this
    # list in a dictionary. So we store all these lists of individual in dictionary and pickling dictionary

    # Creating List Format ( [[keypoints][descriptors]]): List initialization
    DescriptorList = list()

    imageiter = 1
    InnerList = list()
    for i in range(0, 399, 133):

        for j in range(0, 273, 91):
            blockImg = img[i:i + 133, j:j + 91]

            # Harris corners
            harris_corners = cv2.cornerHarris(blockImg, 3, 3, 0.04)
            harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
            threshold_harris = 125
            # Extract keypoints
            KeypointsList = []
            blockKeypoints = list()
            for x in range(0, harris_normalized.shape[0]):
                for y in range(0, harris_normalized.shape[1]):
                    if harris_normalized[x][y] > threshold_harris:
                        blockKeypoints.append(cv2.KeyPoint(y, x, 1))
            # Define descriptor
            orb = cv2.ORB_create()
            # Compute descriptors
            _, blockDescriptors = orb.compute(blockImg, blockKeypoints)

            KeypointsList.append(blockKeypoints)
            DescriptorList.append(blockDescriptors)
            os.chdir("E:\ML\Fingerprint ML Project\Fingerprint\Image_Blocks")
            blockImg[blockImg == 1] = 255

            InnerList.append(len(blockKeypoints))

            if imageiter == 5:
                Kimg = cv2.drawKeypoints(blockImg, blockKeypoints, outImage=None)
                return Kimg

            imageiter += 1

    KeypointsLength.append(InnerList)

    # database.update({imageName: {'img':blockImg, 'keypoints': KeypointsList, 'Descriptors': DescriptorList}})


    # return (keypoints, des)


def main():
    iterI, iterJ = 0, 0
    database = dict()
    KeypointsLength = list()
    f, axarr = plt.subplots(3, 3)
    images = glob.glob("train/*.jpg")
    for image in images:
        os.chdir("E:\ML\Fingerprint ML Project\Fingerprint")
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        Kimg = get_descriptors(img, image, database, KeypointsLength)
        if iterJ > 2:
            iterI += 1
            iterJ = 0
            axarr[iterI][iterJ].imshow(Kimg)
            iterJ += 1
        else:
            axarr[iterI][iterJ].imshow(Kimg)
            iterJ += 1


    plt.show()

    # print(KeypointsLength)
    # print()
    # print(np.sum(KeypointsLength, axis=0))
    #
    # print(np.mean(KeypointsLength, axis=0))
    # print(len(KeypointsLength))

    # # Pickling the descriptors
    # pickle_out = open("minutiae.pickle", "wb")
    # pickle.dump(database, pickle_out)
    # pickle_out.close()
    #
    # pickle_in = open("minutiae.pickle", "rb")
    # dta = pickle.load(pickle_in)

    # # Plot keypoints
    # f, axarr = plt.subplots(3, 3)
    # for i in range(9):
    #     blockImg = cv2.drawKeypoints(dta['train\\1.jpg']['img'], dta['train\\1.jpg']['keypoints'], outImage=None)
    #     axarr[i].imshow(blockImg)
    #
    # plt.show()


if __name__ == "__main__":
	try:
		main()
	except:
		raise