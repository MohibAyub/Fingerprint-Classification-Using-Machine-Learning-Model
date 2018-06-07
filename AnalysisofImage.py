import os, sys
import pickle, cv2, glob
import matplotlib.pyplot as plt
from enhance import image_enhance
from skimage.morphology import skeletonize, thin
import numpy as np


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
    cv2.imshow('skeleton image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
    # Storing these 8 descriptors and 8 keypoints of one image in a Mainlist and key(image name) is associated to represent this
    # list in a dictionary. So we store all these lists of individual in dictionary and pickling dictionary

    # Creating List Format ( [[keypoints][descriptors]]): List initialization
    DescriptorList = list()

    for i in range(0, 384, 192):
        for j in range(0, 288, 144):
            blockImg = img[i:i + 144, j:j + 192]
            # Harris corners
            harris_corners = cv2.cornerHarris(blockImg, 3, 3, 0.04)
            harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
            threshold_harris = 125;
            # Extract keypoints
            keypoints = []
            for x in range(0, harris_normalized.shape[0]):
                for y in range(0, harris_normalized.shape[1]):
                    if harris_normalized[x][y] > threshold_harris:
                        keypoints.append(cv2.KeyPoint(y, x, 1))
            # Define descriptor
            orb = cv2.ORB_create()
            # Compute descriptors
            _, des = orb.compute(blockImg, keypoints)
            DescriptorList.append(des)

    database.update({imageName: DescriptorList})
    # return (keypoints, des);
    # return database


def ScoreCalc(matches, ScorePercent, column):
    for i in range(4):
        score = 0
        for match in matches[i]:
            score += match.distance
        score_threshold = 33
        k = 100 - (score / len(matches[i]))
        ScorePercent[i][column] = k

def BarGraph(Data):

    x = [0,2,4,6]

    plt.ylim(0, 100)
    plt.bar(x, Data, label = 'Matching %')

    plt.xlabel('Blocks')
    plt.ylabel('Percentage')
    plt.title('Matching Percentage of blocks')
    plt.legend()
    plt.show()


def main():
    # database = dict()
    # images = glob.glob("train/*.jpg")
    # for image in images:
    #     img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    #     get_descriptors(img, image, database)

    # img1 = cv2.imread("12train.jpg", cv2.IMREAD_GRAYSCALE)
    # imageName1 = '12train.jpg'
    # img2 = cv2.imread("12test.jpg", cv2.IMREAD_GRAYSCALE)
    # imageName2 = '12test.jpg'
    # # kp2, des2 = get_descriptors(img2)
    # db1 = get_descriptors(img1,imageName1,database)
    # db2 = get_descriptors(img2, imageName2,database)

    # Pickling the descriptors
    # pickle_out = open("BlockWiseData.pickle", "wb")
    # pickle.dump(database, pickle_out)
    # pickle_out.close()

    pickle_in = open("BlockWiseData.pickle", "rb")
    dta = pickle.load(pickle_in)
    # print(dta['train\\12 (1).jpg'][0])



    # # Matching between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ScorePercent = np.zeros([4, 7], dtype=float)

    for i in range(2, 9):
        matches = list()
        compared = 'train\\12 (' + str(i) + ').jpg'
        for j in range(0, 4):
            matches.append(sorted(bf.match(dta['train\\12 (1).jpg'][j], dta[compared][j]), key=lambda match: match.distance))
        ScoreCalc(matches, ScorePercent, i-2)

    Avg_ScorePercent = np.mean(ScorePercent, axis=1)
    BarGraph(Avg_ScorePercent)
    # print(ScorePercent)
    # # Plot keypoints
    # img4 = cv2.drawKeypoints(img, kp1, outImage=None)
    # img5 = cv2.drawKeypoints(img2, kp2, outImage=None)
    # f, axarr = plt.subplots(1,2)
    # axarr[0].imshow(img4)
    # axarr[1].imshow(img5)
    # plt.show()


# # Plot matches
# img3 = cv2.drawMatches(img, kp1, img2, kp2, matches, flags=2, outImg=None)
# plt.imshow(img3)
# plt.show()



# if score/len(matches) < score_threshold:
#    	print("Fingerprint matches.")
# else:
#     print("Fingerprint does not match.")


if __name__ == "__main__":
    try:
        main()
    except:
        raise
