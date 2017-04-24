#import matplotlib.pyplot as plt
import numpy as np

#from skimage import data
import matplotlib.pyplot as plt
import cv2
from skimage.util import img_as_ubyte
from skimage.filter.rank import entropy
from skimage.morphology import disk

# First example: object detection.
def entro(image):
    if len(image.shape) > 2:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    noise_mask = 28 * np.ones((128, 128), dtype=np.uint8)
    noise_mask[32:-32, 32:-32] = 30

    noise = (noise_mask * np.random.random(noise_mask.shape) - 0.5 *
             noise_mask).astype(np.uint8)
    img = noise + 128

    entr_img = entropy(img, disk(10))

    """fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

    ax0.imshow(noise_mask, cmap='gray')
    ax0.set_xlabel("Noise mask")
    ax1.imshow(img, cmap='gray')
    ax1.set_xlabel("Noisy image")
    ax2.imshow(entr_img, cmap='viridis')
    ax2.set_xlabel("Local entropy")

    fig.tight_layout()

    # Second example: texture detection.

    image = img_as_ubyte(data.camera())

    fig, (ax0, ax1) = plt.subplots(ncols=2,
                                   figsize=(12, 4),
                                   sharex=True,
                                   sharey=True,
                                   subplot_kw={"adjustable": "box-forced"})

    img0 = ax0.imshow(image, cmap=plt.cm.gray)
    ax0.set_title("Image")
    ax0.axis("off")
    fig.colorbar(img0, ax=ax0)
    img1 = ax1.imshow(entropy(image, disk(5)), cmap='gray')
    ax1.set_title("Entropy")
    ax1.axis("off")
    fig.colorbar(img1, ax=ax1)

    fig.tight_layout()

    plt.show()
    """
    return entropy(image, disk(5))
if __name__=="__main__":
    i=cv2.imread("amit1.jpg")
    gray=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
    print i
    plt.imshow(entro(gray),cmap="gray")
    plt.show()
