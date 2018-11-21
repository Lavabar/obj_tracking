import cv2
import numpy as np
import alt_undistortion as undist

# path to test image
path = "test_imgs/undist_test.jpg"
image = cv2.imread(path)
# list of points
refPt = []

# helps to find out where to make next point 
st_model = "\n - - - \n - - - \n - - - \n - - - \n"

def get_coords(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, image, st_model

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append([x, y])
        cv2.rectangle(image, (x-2, y-2), (x + 2, y + 2), (0, 0, 255), 3)
        # lighting next point
        st_model = st_model.replace(" ", "*", 1)
        print(st_model)

cv2.namedWindow("image")
cv2.setMouseCallback("image", get_coords)
# lighting first point
st_model = st_model.replace(" ", "*", 1)
print(st_model)
while True:
    cv2.imshow("image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(refPt)

if len(refPt) != 16:
    raise BaseException("Wrong number of points(should be 16)")

refPt = np.asarray(refPt)
refPt = np.reshape(refPt, (4, 4, 2))
print(refPt)

# making list of lines [[p1, p2, p3, p4],...]
def make_lines_bypoints(refPt):
    lines = []
    lines.extend(refPt)
    for i in range(4):
        line = []
        for j in range(4):
            line.append(refPt[j, i])
        lines.append(line)

    lines = np.asarray(lines)
    print(lines)
    return lines

# function for calcualting distance between point and line
def distance(line):
    p1 = line[0]
    p2 = line[1]
    p3 = line[2]
    p4 = line[3]

    d1 = abs((p4[1] - p1[1]) * p2[0] - (p4[0] - p1[0]) * p2[1] + p4[0] * p1[1] - p4[1] * p1[0]) /\
         ((p4[1] - p1[1]) ** 2 + (p4[0] - p1[0]) ** 2) ** 0.5

    d2 = abs((p4[1] - p1[1]) * p3[0] - (p4[0] - p1[0]) * p3[1] + p4[0] * p1[1] - p4[1] * p1[0]) /\
         ((p4[1] - p1[1]) ** 2 + (p4[0] - p1[0]) ** 2) ** 0.5

    return d1, d2

# function for calculating standard deviation
def standard_dev(lines):
    ds = []
    for line in lines:
        d1, d2 = distance(line)
        print("d1, d2 = %f, %f" % (d1, d2))
        ds.append(d1)
        ds.append(d2)

    sum_ds = sum(ds)
    mo_ds = sum_ds / len(ds)
    ds = np.asarray(ds)
    st_dev = (np.sum(np.square(np.subtract(ds, mo_ds))) / ds.shape[0]) ** 0.5

    print("st_dev:")
    print(st_dev)

    return st_dev

lines = make_lines_bypoints(refPt)
st_dev = standard_dev(lines)

# begining iterations
strength = 0.3
zoom = 1.0
lk_params = dict( winSize = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while st_dev >= 2.0:
    new_image = undist.undistortion(image, strength=strength, zoom=zoom)
