import cv2, json, numpy as np
d = json.load(open('intrinsics.json'))
K = np.array(d['K']); dist = np.array(d['dist']); w, h = d['image_width'], d['image_height']
img = cv2.imread('captures/cal_000.png')
newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
und = cv2.undistort(img, K, dist, None, newK)
cv2.imwrite('undist_check.png', und)
print('newK fx,fy,cx,cy =', newK[0,0], newK[1,1], newK[0,2], newK[1,2])