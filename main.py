# from __future__ import print_function
import numpy as np
import cv2
import argparse

# SIFT
img = cv2.imread("pic1.jpg")
sift_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT.create()
sift_kp = sift.detect(sift_gray, None)
img = cv2.drawKeypoints(sift_gray, sift_kp, img)
cv2.imwrite('v2_pic1_sift.jpg', img)
img = cv2.drawKeypoints(sift_gray, sift_kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('v2_pic1_sift2.jpg', img)
sift = cv2.xfeatures2d.SIFT_create()
sift_kp, sift_des = sift.detectAndCompute(sift_gray, None)
print(len(sift_kp), len(sift_des))

image1 = cv2.imread(filename='image1.jpg', flags=cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(filename='image2.jpg', flags=cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT.create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
BFMatcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
output = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:100], image2, flags=2)
cv2.imwrite('v2_image_sift.jpg', output)

image3 = cv2.imread(filename='image3.jpg', flags=cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread(filename='image4.jpg', flags=cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT.create()
keypoints3, descriptors3 = sift.detectAndCompute(image3, None)
keypoints4, descriptors4 = sift.detectAndCompute(image4, None)
BFMatcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck = True)
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(descriptors3, descriptors4)
matches = sorted(matches, key=lambda x: x.distance)
output = cv2.drawMatches(image3, keypoints3, image4, keypoints4, matches[:100], image4, flags=2)
cv2.imwrite('v2_image2_sift.jpg', output)

# SURF
'''
img = cv2.imread("pic1.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
surf = cv2.SURF(30)
kp = surf.detect(gray, None)
img = cv2.drawKeypoints(gray, kp, img)
cv2.imwrite('pic1_surf.jpg', img)
img = cv2.drawKeypoints(gray, kp, img, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('pic1_surf2.jpg', img)
sift = cv2.xfeatures2d.SIFT_create()
kp, des = surf.detectAndCompute(gray, None)
print(len(kp), len(des))
'''
'''
img = cv2.imread("pic1.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
surf = cv2.xfeatures2d_SURF()
kp = surf.detect(gray, None)
img = cv2.drawKeypoints(gray, img, kp)
cv2.imwrite('pic1_surf.jpg', img)
img = cv2.drawKeypoints(gray, kp, img, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('pic1_surf2.jpg', img)
sift = cv2.xfeatures2d.SIFT_create()
kp, des = surf.detectAndCompute(gray, None)
print(len(kp), len(des))
'''
'''
img = cv2.imread("pic1.jpg", cv2.IMREAD_GRAYSCALE)
surf = cv2.xfeatures2d.SURF_create()
keypoints_surf, descriptors = surf.detectAndCompute(img, None)
img = cv2.drawKeypoints(img, keypoints_surf, None)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
parser = argparse.ArgumentParser(description='Code for Feature Detection tutorial.')
parser.add_argument('--input', help='Path to input image.', default='box.png')
args = parser.parse_args()
img = cv2.imread('pic1.jpg', cv2.IMREAD_GRAYSCALE)
minHessian = 400
detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
keypoints = detector.detect(img)
img_keypoints = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
img = cv2.drawKeypoints(img, keypoints, img_keypoints)
cv2.imwrite('pic1_surf.jpg', img)
img = cv2.drawKeypoints(gray, keypoints, img, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('pic1_surf2.jpg', img)
'''

# BRIEF
img = cv2.imread("pic1.jpg")
brief_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
brief_star = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
brief_kp = brief_star.detect(img, None)
brief_kp, brief_des = brief.compute(img, brief_kp)
img = cv2.drawKeypoints(brief_gray, brief_kp, img)
cv2.imwrite('v2_pic1_brief.jpg', img)
img = cv2.drawKeypoints(brief_gray, brief_kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('v2_pic1_brief2.jpg', img)

image1 = cv2.imread(filename='image1.jpg', flags=cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(filename='image2.jpg', flags=cv2.IMREAD_GRAYSCALE)
star = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
brief2 = cv2.cv2.BRISK_create()
keypoints1, descriptors1 = brief2.detectAndCompute(image1, None)
keypoints2, descriptors2 = brief2.detectAndCompute(image2, None)
BFMatcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
matches = BFMatcher.match(queryDescriptors=descriptors1, trainDescriptors=descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
output = cv2.drawMatches(img1=image1, keypoints1=keypoints1, img2=image2,
                         keypoints2=keypoints2, matches1to2=matches[:100],
                         outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('v2_image_brief.jpg', output)

image3 = cv2.imread(filename='image3.jpg', flags=cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread(filename='image4.jpg', flags=cv2.IMREAD_GRAYSCALE)
star = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
brief2 = cv2.cv2.BRISK_create()
keypoints3, descriptors3 = brief2.detectAndCompute(image3, None)
keypoints4, descriptors4 = brief2.detectAndCompute(image4, None)
BFMatcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
matches = BFMatcher.match(queryDescriptors=descriptors3, trainDescriptors=descriptors4)
matches = sorted(matches, key=lambda x: x.distance)
output = cv2.drawMatches(img1=image3, keypoints1=keypoints3, img2=image4,
                         keypoints2=keypoints4, matches1to2=matches[:100],
                         outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('v2_image2_brief.jpg', output)

# ORB
img = cv2.imread("pic1.jpg")
orb_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
orb = cv2.cv2.ORB_create(nfeatures=1500)
orb_kp, descriptors = orb.detectAndCompute(img, None)
img = cv2.drawKeypoints(orb_gray, orb_kp, None)
cv2.imwrite('v2_pic1_orb.jpg', img)
img = cv2.drawKeypoints(orb_gray, orb_kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('v2_pic1_orb2.jpg', img)

image1 = cv2.imread(filename='image1.jpg', flags=cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(filename='image2.jpg', flags=cv2.IMREAD_GRAYSCALE)
orb = cv2.cv2.ORB_create(nfeatures=1500)
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
BFMatcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
matches = BFMatcher.match(queryDescriptors=descriptors1, trainDescriptors=descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
output = cv2.drawMatches(img1=image1, keypoints1=keypoints1, img2=image2,
                         keypoints2=keypoints2, matches1to2=matches[:100],
                         outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('v2_image_orb.jpg', output)

image3 = cv2.imread(filename='image3.jpg', flags=cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread(filename='image4.jpg', flags=cv2.IMREAD_GRAYSCALE)
orb = cv2.cv2.ORB_create(nfeatures=1500)
keypoints3, descriptors3 = orb.detectAndCompute(image3, None)
keypoints4, descriptors4 = orb.detectAndCompute(image4, None)
BFMatcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
matches = BFMatcher.match(queryDescriptors=descriptors3,
                          trainDescriptors=descriptors4)
matches = sorted(matches, key=lambda x: x.distance)
output = cv2.drawMatches(img1=image3, keypoints1=keypoints3, img2=image4,
                         keypoints2=keypoints4, matches1to2=matches[:100],
                         outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('v2_image2_orb.jpg', output)

# KAZE
img = cv2.imread('pic1.jpg')
kaze_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kaze = cv2.KAZE_create()
kaze_kp = kaze.detect(kaze_gray, None)
img = cv2.drawKeypoints(kaze_gray, kaze_kp, img)
cv2.imwrite('v2_pic1_kaze.jpg', img)
img = cv2.drawKeypoints(kaze_gray, kaze_kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('v2_pic1_kaze2.jpg', img)

image1 = cv2.imread(filename='image1.jpg', flags=cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(filename='image2.jpg', flags=cv2.IMREAD_GRAYSCALE)
kaze = cv2.KAZE_create()
keypoints1, descriptors1 = kaze.detectAndCompute(image1, None)
keypoints2, descriptors2 = kaze.detectAndCompute(image2, None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
descriptors1 = np.float32(descriptors1)
descriptors2 = np.float32(descriptors2)
FLANN = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
matches = FLANN.knnMatch(queryDescriptors=descriptors1,
                         trainDescriptors=descriptors2, k=2)
ratio_thresh = 0.7
good_matches = []
for m, n in matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
output = cv2.drawMatches(img1=image1, keypoints1=keypoints1, img2=image2,
                         keypoints2=keypoints2, matches1to2=good_matches,
                         outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('v2_image_kaze.jpg', output)

image3 = cv2.imread(filename='image3.jpg', flags=cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread(filename='image4.jpg', flags=cv2.IMREAD_GRAYSCALE)
kaze = cv2.KAZE_create()
keypoints3, descriptors3 = kaze.detectAndCompute(image3, None)
keypoints4, descriptors4 = kaze.detectAndCompute(image4, None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
descriptors3 = np.float32(descriptors3)
descriptors4 = np.float32(descriptors4)
FLANN = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
matches = FLANN.knnMatch(queryDescriptors=descriptors3,
                         trainDescriptors=descriptors4, k=2)
ratio_thresh = 0.7
good_matches = []
for m, n in matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
output = cv2.drawMatches(img1=image3, keypoints1=keypoints3, img2=image4,
                         keypoints2=keypoints4, matches1to2=good_matches,
                         outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('v2_image2_kaze.jpg', output)

# AKZAE
img = cv2.imread('pic1.jpg')
akaze_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
akaze = cv2.AKAZE_create()
akaze_kp = akaze.detect(akaze_gray, None)
img = cv2.drawKeypoints(akaze_gray, akaze_kp, img)
cv2.imwrite('v2_pic1_akaze.jpg', img)
img = cv2.drawKeypoints(akaze_gray, akaze_kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('v2_pic1_akaze2.jpg', img)

image1 = cv2.imread(filename='image1.jpg', flags=cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(filename='image2.jpg', flags=cv2.IMREAD_GRAYSCALE)
akaze = cv2.AKAZE_create()
keypoints1, descriptors1 = akaze.detectAndCompute(image1, None)
keypoints2, descriptors2 = akaze.detectAndCompute(image2, None)
BFMatcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
matches = BFMatcher.match(queryDescriptors=descriptors1, trainDescriptors=descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
output = cv2.drawMatches(img1=image1, keypoints1=keypoints1, img2=image2,
                         keypoints2=keypoints2, matches1to2=matches[:100],
                         outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('v2_image_akaze.jpg', output)

image3 = cv2.imread(filename='image3.jpg', flags=cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread(filename='image4.jpg', flags=cv2.IMREAD_GRAYSCALE)
akaze = cv2.AKAZE_create()
keypoints3, descriptors3 = akaze.detectAndCompute(image3, None)
keypoints4, descriptors4 = akaze.detectAndCompute(image4, None)
BFMatcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
matches = BFMatcher.match(queryDescriptors=descriptors3, trainDescriptors=descriptors4)
matches = sorted(matches, key=lambda x: x.distance)
output = cv2.drawMatches(img1=image3, keypoints1=keypoints3, img2=image4,
                         keypoints2=keypoints4, matches1to2=matches[:100],
                         outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('v2_image2_akaze.jpg', output)

# BRISK
img = cv2.imread('pic1.jpg')
brisk_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
brisk = cv2.BRISK_create()
brisk_kp = brisk.detect(brisk_gray, None)
img = cv2.drawKeypoints(brisk_gray, brisk_kp, img)
cv2.imwrite('v2_pic1_brisk.jpg', img)
img = cv2.drawKeypoints(brisk_gray, brisk_kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('v2_pic1_brisk2.jpg', img)

image1 = cv2.imread(filename='image1.jpg', flags=cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(filename='image2.jpg', flags=cv2.IMREAD_GRAYSCALE)
BRISK = cv2.BRISK_create()
keypoints1, descriptors1 = BRISK.detectAndCompute(image1, None)
keypoints2, descriptors2 = BRISK.detectAndCompute(image2, None)
BFMatcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
matches = BFMatcher.match(queryDescriptors=descriptors1, trainDescriptors=descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
output = cv2.drawMatches(img1=image1, keypoints1=keypoints1, img2=image2,
                         keypoints2=keypoints2, matches1to2=matches[:100],
                         outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('v2_image_brisk.jpg', output)

image3 = cv2.imread(filename='image3.jpg', flags=cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread(filename='image4.jpg', flags=cv2.IMREAD_GRAYSCALE)
BRISK = cv2.BRISK_create()
keypoints3, descriptors3 = BRISK.detectAndCompute(image3, None)
keypoints4, descriptors4 = BRISK.detectAndCompute(image4, None)
BFMatcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
matches = BFMatcher.match(queryDescriptors=descriptors3, trainDescriptors=descriptors4)
matches = sorted(matches, key=lambda x: x.distance)
output = cv2.drawMatches(img1=image3, keypoints1=keypoints3, img2=image4,
                         keypoints2=keypoints4, matches1to2=matches[:100],
                         outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('v2_image2_brisk.jpg', output)
