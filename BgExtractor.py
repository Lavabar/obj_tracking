import cv2
import numpy as np
from scipy.ndimage import median_filter


class BgExtractor:
    def __init__(self, nframes=100, img_w=640, img_h=480, color_size=256):
        self.nframes = nframes
        self.stats = np.zeros((img_h, img_w, color_size))
        self.img_h = img_h
        self.img_w = img_w
        self.color_size = color_size
        self.n = 0

    def update_bg(self, img, bg):
        # TODO - optimization
        for y in range(self.img_h):
            for x in range(self.img_w):
                self.stats[y, x, img[y, x]] += 1
    
        self.n += 1

        # alternative variant (slower)
        #img = to_categorical(img, num_classes=256)
        #stats = np.add(stats, img)

        if self.n == self.nframes:
            bg = np.argmax(self.stats, axis=2)
            stats = np.zeros((self.img_h, self.img_w, self.color_size))
            self.n = 0
        
        return bg


if __name__ == "__main__":
    bgext = BgExtractor()
    cap = cv2.VideoCapture(0)
    ret, bg = cap.read()
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        bg = bgext.update_bg(gray, bg)

        cv2.imshow("bg", median_filter(np.array(bg, np.uint8), 3))
        cv2.imshow("frame", gray)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
