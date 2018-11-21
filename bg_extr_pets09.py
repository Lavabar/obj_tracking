import os
import cv2
import BgExtractor
import numpy as np

path = "test_videos/Crowd_PETS09/S1/L1/Time_13-59/View_002/"
ls = os.listdir(path)
#print(ls)
ls.sort()

bg = cv2.imread(path+ls[0])
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
bgext = BgExtractor.BgExtractor(nframes=50, img_w=bg.shape[1], img_h=bg.shape[0], color_size=256)

for fr_name in ls:
    img = cv2.imread(path+fr_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = bgext.update_bg(gray, bg)
    
    #cv2.imshow("frame", img)
    cv2.imshow("bg", BgExtractor.median_filter(np.array(bg, np.uint8), 3))
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()