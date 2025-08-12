import cv2 as cv
import numpy as np

DEBUG_PATH = ""
PTS = []

def rect_ref(image, pts, ):

    pts_ordered = order_points(pts)
    clipped_img = clip(image, pts_ordered)
    mW, mH = compute_output_size(pts_ordered)
    dst = np.array([
        [0, 0],
        [mW - 1, 0],
        [mW - 1, mH - 1],
        [0, mH - 1]
    ], dtype="float32")

    M = cv.getPerspectiveTransform(pts_ordered, dst)
    warped = cv.warpPerspective(image, M, (mW, mH))

    return warped


def compute_output_size(pts):
    (tl, tr, br, bl) = pts

    # Compute the width of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    return maxWidth, maxHeight

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)

    # Step 1: sort by y-coordinate (top to bottom)
    y_sorted = pts[np.argsort(pts[:, 1])]

    # Step 2: split into top 2 and bottom 2
    top_two = y_sorted[:2]
    bottom_two = y_sorted[2:]

    # Step 3: sort top and bottom pairs by x to get left/right
    top_left, top_right = top_two[np.argsort(top_two[:, 0])]
    bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)



def clip(img, src_points):

    max_w = int(np.max(src_points[..., 0]))
    min_w = int(np.min(src_points[..., 0]))
    max_h = int(np.max(src_points[..., 1]))
    min_h = int(np.min(src_points[..., 1]))

    clipped = img[min_h:max_h, min_w:max_w]
    return clipped


# def main():
#     image = cv.imread(DEBUG_PATH)
#     rect_image = rect_ref(image, np.array(PTS))
#     cv.show(rect_image)

# #SCIPY optimize
# if __name__ == "__main__":
#     main()