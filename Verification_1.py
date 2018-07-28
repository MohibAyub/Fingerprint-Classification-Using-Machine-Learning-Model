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


def get_descriptors(img):
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

    # Harris corners
    harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
    harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    threshold_harris = 125;
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
    _, blockDescriptors = orb.compute(img, blockKeypoints)

    KeypointsList.append(blockKeypoints)
    img[img == 1] = 255

    Kimg = cv2.drawKeypoints(img, blockKeypoints, outImage=None)
    return Kimg

def showImageBlocks():
    iterI, iterJ = 0, 0
    f, axarr = plt.subplots(3, 3)
    for i in range(1, 9):
        img = cv2.imread("Image_Blocks/"+str(i)+"_5.jpg", 0)
        Kimg = get_descriptors(img)
        if iterJ > 2:
            iterI += 1
            iterJ = 0
            axarr[iterI][iterJ].imshow(Kimg)
            iterJ += 1
        else:
            axarr[iterI][iterJ].imshow(Kimg)
            iterJ += 1

    plt.show()


def main():
    iterI = 1
    database = dict()
    KeypointsLength = list()
    showImageBlocks()
    # images = glob.glob("Image_Blocks/1_*.jpg")
    # for image in images:
    #     # os.chdir("E:\ML\Fingerprint ML Project\Fingerprint")
    #     img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    #     get_descriptors(img, image, database, KeypointsLength, iterI)
    #     iterI += 1

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