import cv2
import numpy as np
import matplotlib.pyplot as plt


ORIGINAL_IMAGE = 'img-orig.jpg'
GRAY_IMAGE = 'img-gray.png'


def load_image(image_path: str) -> np.ndarray:
    return cv2.imread(image_path)


def store_image(image: np.ndarray, image_path: str) -> None:
    cv2.imwrite(image_path, image)


def get_gray_image(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_image_histogram(image: np.ndarray) -> np.ndarray:
    return cv2.calcHist([image], [0], None, [256], [0, 256])


def get_num_pixels(image: np.ndarray) -> int:
    return image.shape[0] * image.shape[1]


def normalize_histogram(histogram: np.ndarray) -> np.ndarray:
    hist_cpy = histogram.copy()
    N = get_num_pixels(gray_image)
    for i in range(len(hist_cpy)):
        hist_cpy[i] = hist_cpy[i] / N

    return hist_cpy


def plot_histogram(histogram: np.ndarray) -> None:
    hist_cpy = normalize_histogram(histogram)
    plt.plot(hist_cpy)
    plt.title('Image Histogram')
    plt.xlabel('Gray Level')
    plt.ylabel('Probability')
    plt.show()


def quantize_image(image: np.ndarray, representing_bits: int) -> np.ndarray:
    # Our new valeus we'll use to quantize the image
    quanta = 256 / 2 ** representing_bits

    # First we floor the image to the nearest quanta, then add half the quanta to get the midpoint
    return np.floor(np.floor(image / quanta) * (quanta) + (quanta / 2))


def get_mse(image1: np.ndarray, image2: np.ndarray) -> float:
    return np.mean((image1 - image2) ** 2)


def plot_mse(image: np.ndarray, max_bits: int) -> None:
    mse = []
    for i in range(1, max_bits + 1):
        quan_image = quantize_image(image, i)
        mse.append(get_mse(image, quan_image))

    plt.plot(range(1, max_bits + 1), mse)
    plt.title('MSE vs. Quantization Bits')
    plt.xlabel('Quantization Bits')
    plt.ylabel('MSE')
    plt.show()


def get_decision_levels(representing_bits: int) -> np.ndarray:
    quanta = 256 / 2 ** representing_bits
    return  np.floor(np.arange(0, 256 + 1, quanta))

def get_representation_levels(representing_bits: int) -> np.ndarray:
    quanta = 256 / 2 ** representing_bits
    return np.floor(np.arange(1, 256, quanta))+ (quanta / 2)

def plot_decision_and_representation_levels(representing_bits: int) -> None:
    decision_levels = get_decision_levels(representing_bits)
    representation_levels = get_representation_levels(representing_bits)

    # plot vertical lines for decision levels
    for level in decision_levels:
        plt.axvline(x=level, color='r', linestyle='dotted')
    plt.axvline(x=256, color='r', linestyle='dotted', label='Decision Levels')

    # plot horizontal lines for representation levels in between decision levels
    for i in range(len(representation_levels)):
        plt.hlines(y=representation_levels[i], xmin=decision_levels[i], xmax=decision_levels[i+1], color='g', linestyle='-')
    plt.hlines(y=256, color='g', linestyle='-', label='Representation Levels', xmin=decision_levels[-1], xmax=256)

    plt.xlim(0, 256)
    plt.ylim(0, 256)
    plt.plot([0, 256], [0, 256], color='b', linestyle='-', label='y = x')

    plt.title(f'Representation and Decision Levels - {representing_bits} bits of representation')
    plt.xlabel('Decision Level')
    plt.ylabel('Representation Level')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    # Part 1
    # Prepare the images
    orig_image = load_image(ORIGINAL_IMAGE)
    gray_image = get_gray_image(orig_image)
    #store_image(gray_image, GRAY_IMAGE)

    # Question 1
    # hist = get_image_histogram(gray_image)
    # plot_histogram(hist)

    # Question 2
    #plot_mse(gray_image, 8)
    #for i in range(1, 9):
    #    plot_decision_and_representation_levels(i)

    # Question 3




