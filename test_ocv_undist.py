from __future__ import print_function
import numpy as np
import cv2
from common import splitfn
import os

if __name__ == '__main__':
    import sys
    import getopt
    from glob import glob

    square_size = 2.0

    pattern_size = (6, 4)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    
    fn = 'test_frame.jpg'
    img = cv2.imread(fn, 0)

    h, w = img.shape[:2]
    refPt = []
    # helps to find out where to make next point 
    st_model = "\n - - - \n - - - \n - - - \n - - - \n"
    # get coordinates of mouse cursor after clicking
    def get_coords(event, x, y, flags, param):
        # grab references to the global variables
        global refPt, img, st_model

        if event == cv2.EVENT_LBUTTONDOWN:
            refPt.append([x, y])
            cv2.rectangle(copy_img, (x-2, y-2), (x + 2, y + 2), 255, 3)
            # lighting next point
            st_model = st_model.replace(" ", "*", 1)
            print(st_model)
    
    copy_img = np.copy(img)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("img", get_coords)
    # beginning process of choosing points on img
    # lighting first point
    st_model = st_model.replace(" ", "*", 1)
    print(st_model)
    while True:
        cv2.imshow("img", copy_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if len(refPt) != 24:
        raise BaseException("Wrong number of points(should be 24)")

    refPt = np.asarray(refPt, dtype=np.float32)
    refPt = np.expand_dims(refPt, axis=1)
    print(refPt)
    img_points.append(refPt.reshape(-1, 2))
    obj_points.append(pattern_points)

    print('ok')

    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    cv2.destroyAllWindows()