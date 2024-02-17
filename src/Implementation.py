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


def plot_decision_and_representation_levels(decision_levels: np.ndarray, representation_levels: np.ndarray) -> None:
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

    plt.title(f'Representation and Decision Levels - {int(np.log2(len(representation_levels)))} bits of representation')
    plt.xlabel('Decision Level')
    plt.ylabel('Representation Level')
    plt.legend()

    plt.show()


def plot_decision_and_representation_levels_bits(representing_bits: int) -> None:
    decision_levels = get_decision_levels(representing_bits)
    representation_levels = get_representation_levels(representing_bits)

    plot_decision_and_representation_levels(decision_levels, representation_levels)


def max_lloyd_mse(hist: np.ndarray, decision_levels: np.ndarray, representation_levels: np.ndarray) -> float:
    mse = 0

    for i in range(len(decision_levels) - 1):
        for j in range(int(decision_levels[i]), int(decision_levels[i+1])):
            mse += hist[j] * (j - representation_levels[i]) ** 2
    
    return mse


def max_lloyd_step(hist: np.ndarray, decision_levels: np.ndarray) -> tuple:
    representation_levels = np.zeros(len(decision_levels) - 1)
    for i in range(len(representation_levels)):
        num = 0
        den = 0
        for j in range(int(decision_levels[i]), int(decision_levels[i+1])):
            num += j * hist[j]
            den += hist[j]

        if num == 0:
            representation_levels[i] = 0
        else:
            representation_levels[i] = num / den

    # Calculate the new optimal decision levels for the given representation levels
    decision_levels = np.zeros(len(representation_levels) + 1)
    for i in range(1, len(representation_levels)):
        decision_levels[i] = (representation_levels[i - 1] + representation_levels[i]) / 2
    decision_levels[0] = 0
    decision_levels[-1] = 256

    return decision_levels, representation_levels


def max_lloyd_algorithm(hist: np.ndarray, decision_levels: np.ndarray, epsilon: float) -> np.ndarray:
    mse = float('inf')
    prev_mse = 0
    representation_levels = np.zeros(len(decision_levels) - 1)

    while abs(mse - prev_mse) > epsilon:
        decision_levels, representation_levels = max_lloyd_step(hist, decision_levels)

        # Calculate the new mse
        prev_mse = mse
        mse = max_lloyd_mse(hist, decision_levels, representation_levels)

    return decision_levels, representation_levels, mse


def plot_max_lloyd_mse(hist: np.ndarray, max_bits: int) -> None:
    mse = []
    epsilon = 0.001
    hist = normalize_histogram(hist)

    for i in range(1, max_bits + 1):
        decision_levels = get_decision_levels(i)
        mse.append(max_lloyd_algorithm(hist, decision_levels, epsilon)[2])

    plt.plot(range(1, max_bits + 1), mse)
    plt.title('MSE vs. Quantization Bits')
    plt.xlabel('Quantization Bits')
    plt.ylabel('MSE')
    plt.show()


def plot_max_lloyd_decision_and_representation_levels(hist: np.ndarray, max_bits: int) -> None:
    epsilon = 0.001
    hist = normalize_histogram(hist)

    for i in range(1, max_bits + 1):
        decision_levels = get_decision_levels(i)
        decision_levels, representation_levels, _ = max_lloyd_algorithm(hist, decision_levels, epsilon)
        plot_decision_and_representation_levels(decision_levels, representation_levels)


def plot_mse_compare(image: np.ndarray, max_bits: int) -> None:
    # this is not the most efficient way to do this, but it's the easiest to get this done
    mse = []
    for i in range(1, max_bits + 1):
        quan_image = quantize_image(image, i)
        mse.append(get_mse(image, quan_image))

    hist = get_image_histogram(image)
    epsilon = 0.001
    hist = normalize_histogram(hist)

    mse_lloyd = []
    for i in range(1, max_bits + 1):
        decision_levels = get_decision_levels(i)
        mse_lloyd.append(max_lloyd_algorithm(hist, decision_levels, epsilon)[2])

    plt.plot(range(1, max_bits + 1), mse, label='Uniform Quantization')
    plt.plot(range(1, max_bits + 1), mse_lloyd, label='Lloyd-Max Quantization')
    plt.title('MSE vs. Quantization Bits')
    plt.xlabel('Quantization Bits')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()


def resize_image(image: np.ndarray) -> np.ndarray:
    new_size_bits = int(min(np.log2(image.shape[0]), np.log2(image.shape[1])))
    new_size = (2**new_size_bits, 2**new_size_bits)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def get_subsampled_mse_image(image: np.ndarray, D: int) -> np.ndarray:
    # As we know from the lecture and question 1, the mean is the optimal for MSE
    subsampled_image = np.zeros((D,D))
    for i in range(D):
        for j in range(D):
            subsampled_image[i][j] = np.mean(image[i*image.shape[0]//D:(i+1)*image.shape[0]//D,
                                                   j*image.shape[1]//D:(j+1)*image.shape[1]//D])
    
    return subsampled_image


def get_subsampled_mad_image(image: np.ndarray, D: int) -> np.ndarray:
    # As we know from the lecture and question 1, the median is the optimal for mAD
    subsampled_image = np.zeros((D,D))
    for i in range(D):
        for j in range(D):
            subsampled_image[i][j] = np.median(image[i*image.shape[0]//D:(i+1)*image.shape[0]//D,
                                                   j*image.shape[1]//D:(j+1)*image.shape[1]//D])
    
    return subsampled_image


def reconstruct_image(subsampled_image: np.ndarray, shape: tuple) -> np.ndarray:
    reconstructed_image = np.zeros(shape)
    D = subsampled_image.shape[0]
    for i in range(D):
        for j in range(D):
            reconstructed_image[i*shape[0]//D:(i+1)*shape[0]//D, j*shape[1]//D:(j+1)*shape[1]//D] = subsampled_image[i][j]
    
    return reconstructed_image


def get_subsampled_and_reconstructed_image(image: np.ndarray, sense: callable) -> tuple:
    J, K, mses = [], [], []
    for i in range(1, 9):
        D = 2 ** i
        subsampled_image = sense(image, D)
        reconstructed_image = reconstruct_image(subsampled_image, image.shape)
        mse = get_mse(image, reconstructed_image)
        J.append(subsampled_image)
        K.append(reconstructed_image)
        mses.append(mse)
    
    return J, K, mses


def plot_err_with_d(err: list, title: str) -> None:
    plt.plot(range(1, len(err)+1), err)
    plt.title(f'{title} vs. D')
    plt.xlabel('D')
    plt.ylabel(f'{title}')
    plt.show()


def plot_images_with_d(images: list, title: str) -> None:
    for i, j in enumerate(images):
        plt.imshow(j, cmap='gray')
        plt.title(f'Reconstructed Image - D={2**(i+1)} - {title}')
        plt.show()

if __name__ == '__main__':
    # Question 1
    # Prepare the images
    orig_image = load_image(ORIGINAL_IMAGE)
    gray_image = get_gray_image(orig_image)
    #print(gray_image.shape)
    #store_image(gray_image, GRAY_IMAGE)

    # part 1
    # hist = get_image_histogram(gray_image)
    # plot_histogram(hist)

    # part 2
    # plot_mse(gray_image, 8)
    # for i in range(1, 9):
    #    plot_decision_and_representation_levels_bits(i)

    # part 3+4
    # plot_max_lloyd_mse(hist, 8)
    # plot_max_lloyd_decision_and_representation_levels(hist, 8)
    # plot_mse_compare(gray_image, 8)

    # Question 2
    # part 1
    resized_image = resize_image(gray_image)
    #print(gray_image.shape, resized_image.shape)
    #plt.imshow(resized_image, cmap='gray')
    #plt.show()

    J, K, mses = get_subsampled_and_reconstructed_image(resized_image, get_subsampled_mse_image)
    #plot_err_with_d(mses, 'MSE')
    #plot_images_with_d(J, 'MSE')
    plot_images_with_d(K, 'MSE')

    J, K, mads = get_subsampled_and_reconstructed_image(resized_image, get_subsampled_mad_image)
    #plot_err_with_d(mads, 'MAD')
    #plot_images_with_d(J, 'MAD')
    plot_images_with_d(K, 'MAD')






