import cv2
import numpy as np

# Load the image
NAME = '0.JPG'
image = cv2.imread(f'/home/dejvax/PycharmProjects/SticherProd/data/phone1/images/0.jpg')
if image is None:
    raise ValueError("Image not found or unable to load.")

# Constants
WINDOW_NAME = 'Large Image Viewer'
zoom_scale = 1.0
pan_x, pan_y = 0, 0
is_panning = False
start_x, start_y = 0, 0
clicked = False
clicked_x, clicked_y = 0, 0
window_x, window_y = 0, 0
last_clicked = (0, 0)
corner_points = []
warped_img = image.copy()
# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global pan_x, pan_y, start_x, start_y, is_panning

    if event == cv2.EVENT_RBUTTONDOWN:
        is_panning = True
        start_x, start_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_panning:
            dx = x - start_x
            dy = y - start_y
            pan_x += dx
            pan_y += dy
            start_x, start_y = x, y
            render_image()
    elif event == cv2.EVENT_RBUTTONUP:
        is_panning = False

    elif event == cv2.EVENT_LBUTTONDOWN:
        # Left-click event
        global clicked_x, clicked_y, clicked, window_y, window_x
        clicked_x = x
        clicked_y = y
        # h, w = image.shape[:2]
        # new_w, new_h = int(w * zoom_scale), int(h * zoom_scale)
        # x1 = max(0, min(new_w - w, -pan_x))
        # y1 = max(0, min(new_h - h, -pan_y))
        # clicked_x = int((x + x1) / zoom_scale)
        # clicked_y = int((y + y1) / zoom_scale)
        # print(f"Clicked at: ({clicked_x}, {clicked_y}) in the original image")
        clicked = True
        render_image()

def draw_rect(display):
    pass
# Render the image based on zoom and pan
def render_image():
    global zoom_scale, pan_x, pan_y, clicked, last_clicked
    h, w = image.shape[:2]
    new_w, new_h = int(w * zoom_scale), int(h * zoom_scale)

    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Calculate the region of interest
    x1 = max(0, min(new_w - w, -pan_x))
    y1 = max(0, min(new_h - h, -pan_y))
    x2 = min(new_w, x1 + w)
    y2 = min(new_h, y1 + h)


    # Display the region
    display = resized[y1:y2, x1:x2]
    if clicked:
        origin_x = int((clicked_x + x1) / zoom_scale)
        origin_y = int((clicked_y + y1) / zoom_scale)
        print(f"Clicked at: ({clicked_x}, {clicked_y}) in the original image")
        font = cv2.FONT_HERSHEY_SIMPLEX
        display = cv2.circle(display, (clicked_x, clicked_y), 15, (255, 0, 0), -15)
        cv2.putText(display, f"({origin_x}, {origin_y})", (clicked_x + int(50*zoom_scale), clicked_y + int(50 * zoom_scale)), font, 5 * zoom_scale, (255, 0, 0), 3, cv2.LINE_AA)
        clicked = False
        last_clicked = (origin_x, origin_y)
    draw_rect(display)
    cv2.imshow(WINDOW_NAME, display)

def clip(img, src_points):

    max_w = int(np.max(src_points[..., 0]))
    min_w = int(np.min(src_points[..., 0]))
    max_h = int(np.max(src_points[..., 1]))
    min_h = int(np.min(src_points[..., 1]))

    clipped = img[min_h:max_h, min_w:max_w]
    return clipped


def clip_rect_img(img):
    global warped_img
    # Coordinates of the document's corners in the original image
    # Replace these with your actual corner coordinates
    src_points = np.array(corner_points, dtype=np.float32)
    clipped_img = clip(img, src_points)

    # Define the width and height of the output image
    height, width = clipped_img.shape[:2]

    # Destination points in the output image
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation
    warped_img = cv2.warpPerspective(image, M, (width, height))

    cv2.imwrite(f"../plots/crop_rect/{NAME}", warped_img)



def render_warped():
    cv2.imshow(WINDOW_NAME, warped_img)


# Keyboard callback function
def keyboard_callback(key):
    global zoom_scale, pan_x, pan_y
    if key == ord('+') or key == ord('='):
        zoom_scale *= 1.1
        pan_x = int(pan_x * 1.1)
        pan_y = int(pan_y * 1.1)
        render_image()
    elif key == ord('-') or key == ord('_'):
        zoom_scale /= 1.1
        pan_x = int(pan_x / 1.1)
        pan_y = int(pan_y / 1.1)
        render_image()
    elif key == ord('s'):
        corner_points.append(last_clicked)
        print(f"Saved {last_clicked} to {corner_points}")
        render_image()
    elif key == ord('d'):
        if len(corner_points) > 0:
            deleted = corner_points.pop()
            print(f"Deleted {deleted} to {corner_points}")
        else:
            print(f"Empty list of points")
    elif key == ord('c'):
        clip_rect_img(image)
        print(f"Clipping with {corner_points}")
    elif key == ord('w'):
        render_warped()
# Main function
def main():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    render_image()

    while True:
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to exit
            break
        keyboard_callback(key)

    cv2.destroyAllWindows()


#SCIPY optimize
if __name__ == "__main__":
    main()