"""
preprocessing images or augmenting to enhance training data

steps to achive:
normal distribution of samples with steering angles evenly distributed

resize images
shear
shift
rotate
brightness
shadow patch - draw random polygon
mirror image


"""
import os
import csv
# %matplotlib inline
import cv2
import math

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline


def plot_img(image):
    plt.figure()
    plt.imshow(image, cmap="gray")


def _img_specs(image, fake=False):
    if fake:
        return (image['h'], image['w'])
    return image.shape[:2]


def crop_img(image, steer, y_st=66, y_end=135, x_st=0, x_end=None, fake=False):
    if fake:
        image["h"] = y_end - y_st
        image["w"] = x_end if x_end else image["w"] - x_st
        return (image, steer)
    if x_end is None:
        x_end = image.shape[1]
    return (image[y_st:y_end, x_st:x_end], steer)


def resize(image, steer, fake=False):
    if fake:
        image["h"] //= 2
        image["w"] //= 2
        return (image, steer)
    scale = tuple(reversed(list(map(lambda x: x // 2, _img_specs(image, fake)))))
    return (cv2.resize(image, scale, interpolation=cv2.INTER_AREA), steer)


def rand_warp_shear(image, steer, s_range=[0, 75], shear_right=False, fake=False):
    """changes steering angle d_steer"""
    mi, mx = s_range
    h, w = _img_specs(image, fake)
    # shear = int(np.random.uniform() * (mx - mi) + mi)
    shear = min(abs(np.random.normal(s_range[0], s_range[1] / 2)), s_range[1])
    if shear_right:
        ori_pts = np.float32([[0, 0], [0, h], [w // 2, h // 2]])
        new_pts = np.float32([[0, 0], [0, h], [w // 2 + shear, h // 2]])
        # TODO mark shear correction coefficient
        steer = min(1, steer + math.atan(shear / (h / 2) / 20))
    else:
        ori_pts = np.float32([[w, 0], [w, h], [w // 2, h // 2]])
        new_pts = np.float32([[w, 0], [w, h], [w // 2 - shear, h // 2]])
        steer = max(-1, steer + math.atan(-shear / (h / 2) / 20))
    if fake:
        return (image, steer)
    matrix = cv2.getAffineTransform(ori_pts, new_pts)

    return (cv2.warpAffine(image, matrix, (w, h), borderMode=1), steer)


def rand_warp_shift2D(image, steer, r_tx=[0, 0], r_ty=[20, 30], fake=False):
    """deprecated. use shear/rotate for better results"""
    h, w = _img_specs(image, fake)
    tx = int(np.random.uniform() * (r_tx[1] - r_tx[0]) + r_tx[0])
    ty = int(np.random.uniform() * (r_ty[1] - r_ty[0]) + r_ty[0])
    d_steer = math.atan(tx / (h / 2))
    if fake:
        return (image, steer + d_steer)
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return (cv2.warpAffine(image, matrix, (w, h), borderMode=1), steer + d_steer)


def rand_warp_rotate(image, steer, range=[-10, 10], left_pivot=False, fake=False):
    if fake:
        return (image, steer)
    h, w = _img_specs(image, fake)
    rotate = np.random.normal(sum(range) // 2, range[1] / 2)
    rotate = -10 if rotate < -10 else (10 if rotate > 10 else rotate)
    if left_pivot:
        pivot = (0, h)
    else:
        pivot = (w, h)
    M = cv2.getRotationMatrix2D(pivot, -rotate, 1)
    return (cv2.warpAffine(image, M, (w, h), borderMode=1), steer)


def flip(image, steer, p_flip=0.5, fake=False):
    """changes steering angle cahnges sign only"""
    if np.random.uniform() >= p_flip:
        steer = -steer
        if fake:
            return (image, steer)
        image = cv2.flip(image, 1)
    return (image, steer)


def rand_brightness(image, steer, fake=False):
    if fake:
        return (image, steer)
    image_hsv = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2HSV), dtype=np.float64)
    rand_bright = .5 + np.random.uniform()
    image_hsv[:, :, 2] = image_hsv[:, :, 2] * rand_bright
    image_hsv[:, :, 2][image_hsv[:, :, 2] > 255] = 255
    image_hsv = np.array(image_hsv, dtype=np.uint8)
    return (cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB), steer)


def create_transform_pipeline(*args, **kwargs):
    params = kwargs.get("params", {})

    def transform_image(image, steer, plot_pipeline=False, fake=False):
        for func in args:
            param = {} if func not in params else params[func]
            if type(param) is dict:
                param['fake'] = fake
                image, steer = func(image, steer, **param)
            else:
                image, steer = func(image, steer, *param, **{'fake': fake})
            if plot_pipeline:
                print ("Applied transformation : {} with params: {}".format(func.__name__, param))
                plot_img(image)
                print ("Adjusted steering angle : ", steer)
        return image, steer
    return transform_image


# def create_process_pipelien()
# def process_image(image, steer):


def create_aug_img_pipeline(plot=False, fake=False, accepted_angles=None):
    # normal_dis=False, sample_size=None):
    left_turn_params = {
        rand_warp_rotate: {"left_pivot": False},
        rand_warp_shear: {"shear_right": False},
    }
    left_turn_trans_img_pipeline = create_transform_pipeline(
        crop_img, resize, rand_brightness, rand_warp_shear, rand_warp_rotate, flip,
        params=left_turn_params
    )
    right_turn_params = {
        rand_warp_rotate: {"left_pivot": True},
        rand_warp_shear: {"shear_right": True},
    }
    right_turn_trans_img_pipeline = create_transform_pipeline(
        crop_img, resize, rand_brightness, rand_warp_shear, rand_warp_rotate, flip,
        params=right_turn_params
    )
    args = [plot, fake]

    def augment_image(image, steer):
        if accepted_angles:
            if steer not in accepted_angles:
                return (image, steer)
        # TODO generalize coin tosses
        # Monkey patch normalized distri
        if steer == 0.0:
            if np.random.uniform() <= 0.2:
                return (image, steer)

        if plot:
            plot_img(image)
            print ("original steering : ", steer)
        if steer > 0 or ((not steer) and np.random.uniform() <= 0.5):
            image, steer = right_turn_trans_img_pipeline(image, steer, *args)
        else:
            image, steer = left_turn_trans_img_pipeline(image, steer, *args)
        return (image, steer)

    return augment_image


LOG_HEADERS = ["center", "left", "right", "steering", "throttle"]

PATH_TO_DATA_FOLDER = 'data'
# data heirachy should be one more dir deep as this may contain multiple dirs
#   which contain different training data

LOG_FILE = 'driving_log.csv'

# for each dir entry, training data is fetched from corresponding subdirectory
# TODO add all folders
PATHS_TO_IMG_FOLDERS = ['data_ori', ]
# 'data_sides', 'data_lap', 'data_reverse']
# 'data_ori']#, 'data_sides', 'data_lap', 'data_reverse']

PATH_TO_IMG = 'IMG'


def parse_logs(limit=20000):
    samples = []
    sample_count = 0
    quit = False
    for folder in PATHS_TO_IMG_FOLDERS:
        log_file_path = os.path.join(PATH_TO_DATA_FOLDER, folder, LOG_FILE)
        folder_count = 0
        with open(log_file_path) as logs:
            reader = csv.reader(logs)
            for row in reader:
                if row[0] in LOG_HEADERS:
                    continue
                # including source folder as extra col to row
                row.append(folder)
                samples.append(row)
                sample_count += 1
                folder_count += 1
                if sample_count >= limit:
                    quit = True
                    break
        print ("extracted {} samples from {}.".format(folder_count, folder))
        if quit:
            break
    print ("extracted {} samples total.".format(sample_count))
    return samples


def get_img(img_entry, folder, fake=False):
    if fake:
        return {'h': 160, 'w': 320}
    path = os.path.join(PATH_TO_DATA_FOLDER, folder, PATH_TO_IMG,
                        img_entry.split('/')[-1])
    return cv2.imread(path)


def extract_samples_from_rows(data, img_index=0, angle_offset=0, fake=False):
    if type(img_index) == int:
        img_index = [img_index, ]
    if type(angle_offset) == int:
        angle_offset = [angle_offset, ]
    images = []
    angles = []
    for row in data:
        images.extend([get_img(row[i], row[-1], fake) for i in img_index])
        angles.extend([float(row[3]) + offset for offset in angle_offset])
    return (images, angles)

STEERING_CORRECTION_FACTORS = [0, 1.0 / 10, -1.0 / 10]
# TODO set to True
INCLUDE_SIDES = False
# set to false to also augment side images
AUGMENT_CENTER_ONLY = True
def generate_batch(data, batch_size=128):
    while True:
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            # adjusting for all camera images

            images, steers = extract_samples_from_rows(batch, [0, 1, 2], [0, -0.15, 0.15])
            aug_image = create_aug_img_pipeline(accepted_angles=[0, 0.15, -0.15])
            images_a, steers_a = zip(*list(map(aug_image, images, steers)))
            yield images_a, steers_a


def get_data():
    data = parse_logs()
    np.random.shuffle(data)
    # TODO generalize
    sample_size = len(data) * 3
    return (data, sample_size)

# testing utilities


def convert_img_to_fakes(images):
    return list(map(lambda x: {'h': x.shape[0], 'w': x.shape[1]}, images))


def plot_data(data):
    plt.hist(data, bins=100)
    plt.title("Gaussian Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.gcf()
    # return fig


def test():
    data = parse_logs()
    images, steers = extract_samples_from_rows(data, [0, 1, 2], [0, -0.15, 0.15], fake=True)
    plot_data(steers)
    aug_image = create_aug_img_pipeline(fake=True, accepted_angles=[0, 0.15, -0.15])
    images_a, steers_a = zip(*list(map(aug_image, images, steers)))
    plot_data(steers_a)
    print (len(steers))


if __name__ == "__main__":
    test()