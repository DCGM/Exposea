#from PIL.Image                         import open
from torchvision.transforms.functional import to_tensor


def read_image(path):
    """Returns a normalized image represented as [C, H, W] tensor."""
    from PIL.Image import open
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 1000000000 # To avoid DecompressionBombError
    with open(path) as im:
        return to_tensor(im.convert('RGB'))


def make_resize_image(max_size):
    from torchvision.transforms.functional import resize
    from torchvision.transforms            import InterpolationMode
    BILINEAR = InterpolationMode.BILINEAR
    def resize_image(image):
        _, h, w   = image.shape
        scale     = max_size / max(h, w)
        return [resize(img           = image,
                      size          = [round(scale * h), round(scale * w)],
                      antialias     = True,
                      interpolation = BILINEAR),
                scale]
    return resize_image


def warp_image(image, H):
    from cv2 import warpPerspective
    _, h, w = image.shape
    return warpPerspective(to_numpy_image(image), H, [w, h])


def to_numpy_image(tensor_image):
    """Converts tensor image to numpy image [C, H, W] -> [H, W, C]."""
    tensor_image[1 < tensor_image] = 1 # Silence warnings by clamping to <0, 1>
    return tensor_image.permute(1, 2, 0).numpy()



def make_line_between_axes(figure, src_axes, dst_axes):
    """Plots a line connecting two axes."""
    from matplotlib.patches import ConnectionPatch
    src_transform = src_axes.transData
    dst_transform = dst_axes.transData
    def line_between_axes(xy0, xy1, color):
        figure.add_artist(ConnectionPatch(
            xyA = xy0, coordsA = src_transform,
            xyB = xy1, coordsB = dst_transform, color = color))
    return line_between_axes