import glob
import cv2
import numpy as np

## Pre-processing kmeans 
def segmentation_kmeans(img, K=5):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts=10
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    return result_image

DATA_PATH = "../../dataset_mri/lgg-mri-segmentation_kmeans_preprocessed/kaggle_3m/"
files_regex = DATA_PATH + "**/*[0-9].tif"

filenames = [f for f in sorted(glob.glob(files_regex, recursive=True))]

for filename in filenames:
    img = cv2.imread(filename)

    preprocessed_image = segmentation_kmeans(img, 5)

    # Save new image
    new_filename = filename[:-4] + ".png"
    cv2.imwrite(new_filename, preprocessed_image)