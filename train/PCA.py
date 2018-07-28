import numpy as np
np.set_printoptions(threshold=np.nan)
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import glob
# start = time.time()


def PCA(img):
    sess = tf.InteractiveSession()

    imgD = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    imgMean = imgD.mean(0)  # (275, )
    imgMean = imgMean.reshape(1, -1)  # 1 x 275

    height, width = img.shape[:2]

    img_meanNew = np.repeat(imgMean, height, axis=0)  # 400 x 275
    imgAdjust = imgD - img_meanNew  # 400 x 275

    cov_img = np.cov(imgAdjust.T)  # 275 x 275
    E, V = tf.linalg.eigh(cov_img)

    V_trans = tf.transpose(V)
    imgAdjust_trans = imgAdjust.T  # 275 x 400
    finalImg = tf.matmul(V_trans.eval(), imgAdjust_trans.astype(float))  # 275 x 400
    # End of PCA

    # Start of Inverse PCA Code

    OriginalImg_trans = tf.matmul(tf.matrix_inverse(V_trans), finalImg)
    OriginalImg = tf.add(tf.transpose(OriginalImg_trans), img_meanNew.astype(float))

    # End of Inverse PCA

    # Image Compression
    PCs = 20
    PCs = width - int(PCs)
    Reduced_V = V.eval()
    Reduced_V = Reduced_V[:, PCs:width]

    Y = tf.matmul(Reduced_V.T.astype(float), imgAdjust_trans.astype(float))
    CompressedImg = tf.matmul(Reduced_V, Y)
    CompressedImg = tf.add(tf.transpose(CompressedImg), img_meanNew.astype(float))

    # cv2.imwrite("c"+str(i)+".jpg", (255 * (CompressedImg.eval())))
    sess.close()
    return CompressedImg.eval()


# def init(des):
#
#     # images = glob.glob("*.jpg")
#     # i =1
#     # for image in images:
#     #     img = cv2.imread(image, 0)  # 400 x 275
#     #     PCA(img, i)
#     #     i += 1
#
#     return PCA(des)

    # print('It took', time.time() - start, 'seconds.')


