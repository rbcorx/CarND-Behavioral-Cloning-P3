import os
import csv
import cv2
import math
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
NOTE: almost all augmentation/helper functions have support for 'fake' data, i.e. augment a data item without
loading the real image to memory. A fake image is just represented by it's height and weight.

The purpose of this is to visualize effects of different params on the final augmented data
faster by only observing how the steering angles and the image dimensions are changed due to augmentation.
"""


LOG_HEADERS = ["center", "left", "right", "steering", "throttle"]

PATH_TO_DATA_FOLDER = 'data'
# data heirachy should be one more dir deep as this may contain multiple dirs
#   which contain different training data

LOG_FILE = 'driving_log.csv'

# for each dir entry, training data is fetched from corresponding subdirectory
# TODO add all folders
PATHS_TO_IMG_FOLDERS = ['track1', "data_ori"]
# "recovery"]  # 'recovery', 'drive' 'data_sides', 'data_lap', 'data_reverse']


PATH_TO_IMG = 'IMG'

IMAGE_INDEX = [0, 1, 2]
STEERING_CORRECTION_COEFF = [0, 0.15, -0.15]

INCLUDE_SIDES = True
# set to false to also augment side images

AUGMENT_IMAGES = True
# augment images
DO_SHEAR_PROB = 0.45
SHEAR_ATAN_CORRECTION = 5
SHEAR_RANGE = [0, 20]
SHEAR_MU = SHEAR_RANGE[0]
SHEAR_SIGMA = SHEAR_RANGE[1]

# adjust inputs
CHOOSE_ALL_CAMERAS = True
SHUFFLE = True

# filters
P_DROP_ZERO = 0.97
DO_FILTER = True
P_FILTER = 0.95
FILTER_BIN_SIZE = 0.01
FILTER_BIN_TAKE_ABS = False
DROP_ZERO_HISTORY_LIM = 1000


DO_FLIP = True

P_KEEP_ZERO = 0.01

P_FLIP = 0.5

CROP_IMG_Y_START = 70
CROP_IMG_Y_END = 136
CROP_IMG_X_START = 0
CROP_IMG_X_END = None

RESIZE_DOWN_SCALE = 2

SAMPLE_SIZE_CORRECTION = 1  # (3 if CHOOSE_ALL_CAMERAS else 1) * (2 if DO_FLIP else 1)

settings = {
    "IMAGE_INDEX": IMAGE_INDEX,
    "STEERING_CORRECTION_COEFF": STEERING_CORRECTION_COEFF,
    # TODO set to True
    "INCLUDE_SIDES": INCLUDE_SIDES,
    # set to false to also augment side images

    "AUGMENT_IMAGES": AUGMENT_IMAGES,
    # augment images
    "DO_SHEAR_PROB": DO_SHEAR_PROB,
    "SHEAR_ATAN_CORRECTION": SHEAR_ATAN_CORRECTION,
    "SHEAR_RANGE": SHEAR_RANGE,
    "SHEAR_MU": SHEAR_MU,
    "SHEAR_SIGMA": SHEAR_SIGMA,

    # adjust inputs
    "CHOOSE_ALL_CAMERAS": CHOOSE_ALL_CAMERAS,
    "DO_FLIP": DO_FLIP,
    "SHUFFLE": SHUFFLE,

    # filters
    "P_DROP_ZERO": P_DROP_ZERO,
    "DROP_ZERO_HISTORY_LIM": DROP_ZERO_HISTORY_LIM,
    "DO_FILTER": DO_FILTER,
    "P_FILTER": P_FILTER,
    "FILTER_BIN_SIZE": FILTER_BIN_SIZE,
    "FILTER_BIN_TAKE_ABS": FILTER_BIN_TAKE_ABS,

    "P_FLIP": P_FLIP,

    "CROP_IMG_Y_START": CROP_IMG_Y_START,
    "CROP_IMG_Y_END": CROP_IMG_Y_END,
    "CROP_IMG_X_START": CROP_IMG_X_START,
    "CROP_IMG_X_END": CROP_IMG_X_END,

    "RESIZE_DOWN_SCALE": RESIZE_DOWN_SCALE,

    "SAMPLE_SIZE_CORRECTION": SAMPLE_SIZE_CORRECTION,
    "P_KEEP_ZERO": P_KEEP_ZERO,
}


def change_settings(**kwargs):
    # to dynamically change settings on the go
    settings.update(kwargs)


def plot_img(image):
    plt.figure()
    plt.imshow(image, cmap="gray")


def _img_specs(image, fake=False):
    if fake:
        return (image['h'], image['w'])
    return image.shape[:2]


def crop_img(image, steer, y_st=None, y_end=None,
             x_st=None, x_end=None, fake=False):
    if not y_st:
        y_st=settings["CROP_IMG_Y_START"]
    if not y_end:
        y_end=settings["CROP_IMG_Y_END"]
    if not x_st:
        x_st=settings["CROP_IMG_X_START"]

    if fake:
        image["h"] = y_end - y_st
        image["w"] = x_end - x_st if x_end else image["w"] - x_st
        return (image, steer)
    if x_end is None:
        x_end = image.shape[1]
    return (image[y_st:y_end, x_st:x_end], steer)


def resize(image, steer, fake=False):
    if fake:
        image["h"] //= settings["RESIZE_DOWN_SCALE"]
        image["w"] //= settings["RESIZE_DOWN_SCALE"]
        return (image, steer)
    scale = tuple(reversed(list(map(lambda x: x // settings["RESIZE_DOWN_SCALE"], _img_specs(image, fake)))))
    # (64, 64)
    return (cv2.resize(image, scale, interpolation=cv2.INTER_AREA), steer)


def rand_warp_shear(image, steer, shear_right=False, fake=False):
    """Shears the image to right/left. Changes steering angle d_steer"""
    # mi, mx = s_range
    h, w = _img_specs(image, fake)
    # if random.random() > settings["DO_SHEAR_PROB"]:
    #     return (image, steer)
    # shear = int(np.random.uniform() * (mx - mi) + mi)
    shear = abs(np.random.normal(settings["SHEAR_MU"], settings["SHEAR_SIGMA"]))  # min(, s_range[1])
    # shear = abs(random.randint(settings["SHEAR_RANGE"][0], settings["SHEAR_RANGE"][1]))
    if shear_right:
        ori_pts = np.float32([[0, 0], [0, h], [w // 2, h // 2]])
        new_pts = np.float32([[0, 0], [0, h], [w // 2 + shear, h // 2]])
        # TODO mark shear correction coefficient
        # steer = min(1, steer + math.atan(shear / (h / 2) / settings["SHEAR_ATAN_CORRECTION"]))
        steer += shear / (h / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    else:
        ori_pts = np.float32([[w, 0], [w, h], [w // 2, h // 2]])
        new_pts = np.float32([[w, 0], [w, h], [w // 2 - shear, h // 2]])
        #steer = max(-1, steer + math.atan(-shear / (h / 2) / settings["SHEAR_ATAN_CORRECTION"]))
        steer -= shear / (h / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    if fake:
        return (image, steer)
    matrix = cv2.getAffineTransform(ori_pts, new_pts)

    return (cv2.warpAffine(image, matrix, (w, h), borderMode=1), steer)


def rand_warp_shift2D(image, steer, r_tx=[0, 0], r_ty=[20, 30], fake=False):
    """shifts image in 2D. deprecated. use shear/rotate for better results"""
    h, w = _img_specs(image, fake)
    tx = int(np.random.uniform() * (r_tx[1] - r_tx[0]) + r_tx[0])
    ty = int(np.random.uniform() * (r_ty[1] - r_ty[0]) + r_ty[0])
    d_steer = math.atan(tx / (h / 2))
    if fake:
        return (image, steer + d_steer)
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return (cv2.warpAffine(image, matrix, (w, h), borderMode=1), steer + d_steer)


def rand_warp_rotate(image, steer, range=[-10, 10], left_pivot=False, fake=False):
    """rotates images while keeping steering angle constant"""
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


def flip(image, steer, p_flip=None, fake=False):
    """flips image horizontally. changes steering angle, sign only"""
    if p_flip is None:
        p_flip = settings["P_FLIP"]
    if np.random.uniform() <= p_flip:
        steer = -steer
        if fake:
            return (image, steer)
        image = cv2.flip(image, 1)
    return (image, steer)


def rand_brightness(image, steer, fake=False):
    """changes image brightness randomly"""
    if fake:
        return (image, steer)
    image_hsv = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2HSV), dtype=np.float64)
    rand_bright = .5 + np.random.uniform()
    image_hsv[:, :, 2] = image_hsv[:, :, 2] * rand_bright
    image_hsv[:, :, 2][image_hsv[:, :, 2] > 255] = 255
    image_hsv = np.array(image_hsv, dtype=np.uint8)
    return (cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB), steer)


def create_transform_pipeline(*args, **kwargs):
    """creates transform pipeline to encapsulate transformations within a single func"""
    params = kwargs.get("params", {})

    def transform_image(image, steer, plot_pipeline=False, fake=False):
        """ARGS:
        plot pipeline (bool): use to make transfromation verbose and compare before after results

        """
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


def create_aug_img_pipeline(plot=False, fake=False, accepted_angles=None, augment=None,):
    """Image augmentation pipeline. encapsulates logic of applying series
        of augmentation on a data entry"""

    # normal_dis=False, sample_size=None):
    if not augment:
        augment = settings["AUGMENT_IMAGES"]

    left_turn_params = {
        rand_warp_rotate: {"left_pivot": False},
        rand_warp_shear: {"shear_right": False},
    }
    # augmentations for left turns
    left_turn_trans_img_pipeline = create_transform_pipeline(
        rand_warp_shear, rand_warp_rotate, rand_brightness,
        params=left_turn_params  # crop_img, resize,
    )
    right_turn_params = {
        rand_warp_rotate: {"left_pivot": True},
        rand_warp_shear: {"shear_right": True},
    }
    # augmentations for right turns
    right_turn_trans_img_pipeline = create_transform_pipeline(
        rand_warp_shear, rand_warp_rotate, rand_brightness,
        params=right_turn_params  # crop_img, resize,
    )
    # no augmentation pipeline, only crops and resizes image
    not_augmented_img_pipeline = create_transform_pipeline(
        # crop_img, resize,
        params={}
    )
    args = [plot, fake]

    def augment_image(image, steer, flipit=False):

        # deprecated flip implementation
        # if flipit:
        #     return flip(image, steer, 1.0, fake=fake)

        if not augment:
            return not_augmented_img_pipeline(image, steer, *args)

        # if accepted_angles:
        #    if steer not in accepted_angles:
        #        return not_augmented_img_pipeline(image, steer, *args)

        # TODO generalize coin tosses
        # Monkey patch normalized distribution
        # using the probability of not augmenting a zero angle steer, return image as is
        if steer == 0.0:
            if np.random.uniform() <= settings["P_KEEP_ZERO"]:
                return not_augmented_img_pipeline(image, steer, *args)

        if plot:
            plot_img(image)
            print ("original steering : ", steer)
        if steer > 0 or ((not steer) and np.random.uniform() <= 0.5):
            image, steer = right_turn_trans_img_pipeline(image, steer, *args)
        else:
            image, steer = left_turn_trans_img_pipeline(image, steer, *args)
        return (image, steer)

    return augment_image


def parse_logs(limit=200000, p_drop_zero=None,
               drop_zero_history_lim=None, replicate_from_folders=[], flipit=None):
    """parses csv logs to return data entries while also applying:
            flipping,
            camera view selection (random or all),
            gaussian probabilistic filters on dynamic steering angle bins for data normalization"""

    if not p_drop_zero:
        p_drop_zero = settings["P_DROP_ZERO"]
    if not drop_zero_history_lim:
        drop_zero_history_lim = settings["DROP_ZERO_HISTORY_LIM"]
    samples = []
    sample_count = 0
    quit = False
    PATHS_TO_IMG_FOLDERS.extend(replicate_from_folders)
    dropped = 0
    print("flipping is set to {}".format(settings["DO_FLIP"]))
    for folder in PATHS_TO_IMG_FOLDERS:
        log_file_path = os.path.join(PATH_TO_DATA_FOLDER, folder, LOG_FILE)
        folder_count = 0

        with open(log_file_path) as logs:
            history = []
            reader = csv.reader(logs)
            for row in reader:
                if row[0] in LOG_HEADERS:
                    continue
                # including source folder as extra col to row
                angle = float(row[3])

                # dropping consecutive zero angles over a limit
                if angle == 0.0 and drop_zero_history_lim is not None:
                    if len(history) >= drop_zero_history_lim:
                        # print (history)
                        # print("dropping row for limit")
                        dropped += 1 * (3 if settings["CHOOSE_ALL_CAMERAS"]
                                        else 1) * (2 if flipit else 1)
                        continue
                    history.append(angle)
                    # print ("appending to history")
                elif drop_zero_history_lim is not None:
                    history = []

                # dropping zero angles with probability
                if p_drop_zero:
                    if angle == 0.0 and np.random.uniform() <= p_drop_zero:
                        dropped += 1 * (3 if settings["CHOOSE_ALL_CAMERAS"]
                                        else 1) * (2 if flipit else 1)
                        continue

                row.append(folder)
                row.append("norm")

                if settings["CHOOSE_ALL_CAMERAS"]:
                    corr = settings["STEERING_CORRECTION_COEFF"]
                    indexes = settings["IMAGE_INDEX"]
                    items = [[row[indexes[i]], float(row[3]) + corr[i], row[-2], row[-1]]
                             for i in range(len(indexes))]
                    #print ("choosing all cameras: len items = {}".format(len(items)))
                else:
                    items = [[row[0], float(row[3]), row[-2], row[-1]], ]
                # samples.append(row)

                # flipping image if settings is set, doing here to randomize data spread
                if flipit:
                    flipped = []
                    for item in items:
                        item = list(item)
                        item[-1] = "flip"
                        flipped.append(item)
                        # samples.append(row)
                    # print ("choosing all cameras: flipping items len = {}".format(len(flipped)))
                    items.extend(flipped)
                # print ("choosing all cameras: final item count append = {}".format(len(items)))
                samples.extend(items)
                sample_count += len(items)
                folder_count += len(items)

                # sample_count += 1
                # folder_count += 1
                if sample_count >= limit:
                    quit = True
                    break
        print ("extracted {} samples from {}.".format(folder_count, folder))
        if quit:
            break
    print ("{} samples dropped based on 00 rules".format(dropped))

    """
    appying custom filters by value and count
    """
    bin_dict = {}
    # deals with absolute values
    take_abs = settings["FILTER_BIN_TAKE_ABS"]
    mi_value = None
    bin_size = settings["FILTER_BIN_SIZE"]

    def binify(sorted_list, bin_len=None):
        """creates bins based on the sorted list provided"""
        if not bin_len:
            bin_len = bin_size
        cur = 1
        if cur not in bin_dict:
            bin_dict[cur] = 0
        # print ("\n\nfinding bin ... \n\n")
        for el, count in sorted_list:
            # print(el, count)
            if take_abs:
                el = abs(el)
            # print ("comparing value el : {} >> {}".format(el, cur * bin_len + mi_value))
            while el > cur * bin_len + mi_value:
                cur += 1
                if cur not in bin_dict:
                    bin_dict[cur] = 0
            # print ("found value el : {} < {} @ CUr = {}\n\n".format(el, cur * bin_len + mi_value, cur))
            bin_dict[cur] += count
        # print ("binified with bin width: {}. \n {}".format(bin_len, bin_dict))

    def get_bin(stat, bin_len=None):
        """get bin for provided statistic"""
        if not bin_len:
            bin_len = bin_size
        cur = 1
        el, count = stat
        if take_abs:
            el = abs(el)
        while el > cur * bin_len + mi_value:
            cur += 1
        return bin_dict[cur]

    if settings["DO_FILTER"] is True:
        steers = list(map(lambda x: x[1], samples))
        stats = list(zip(*np.unique(steers, return_counts=True)))
        mx_count = max(stats, key=lambda x: x[1])[1]
        mx_value = abs(max(stats, key=lambda x: abs(x[0]))[0])
        if take_abs:
            mi_value = abs(min(stats, key=lambda x: abs(x[0]))[0])
        else:
            mi_value = min(stats, key=lambda x: x[0])[0]
        if take_abs:
            sorted_stats = sorted(stats, key=lambda x: abs(x[0]))
        else:
            sorted_stats = sorted(stats, key=lambda x: x[0])
        print ("\n\n")
        print ("MIN VALUE :: ", mi_value)
        binify(sorted_stats)
        mx_count = max(bin_dict.values())
        print ("MAX COUNT :: ", mx_count)

        count_dict = dict(stats)
        filtered_samples = []
        for sample in samples:
            angle = sample[1]
            p_filter_val = (1 - abs(angle) / mx_value) * settings["P_FILTER"]
            p_filter_count = get_bin((angle, count_dict[angle])) / (mx_count) * settings["P_FILTER"]
            p_filter_avg = p_filter_val * p_filter_count
            if np.random.uniform() < p_filter_avg:
                continue
            filtered_samples.append(sample)
        print ("@Filter ON. {} samples were filtered.".format(sample_count - len(filtered_samples)))
        samples = filtered_samples
        sample_count = len(filtered_samples)
        dropped += sample_count - len(filtered_samples)

    print ("extracted {} samples total.".format(sample_count))
    return samples


def get_img(img_entry, folder, flipit, fake=False):
    """get image from data entry"""
    if fake:
        return {'h': 160, 'w': 320}
    path = os.path.join(PATH_TO_DATA_FOLDER, folder, PATH_TO_IMG,
                        img_entry.split('/')[-1])
    img = mpimg.imread(path)
    if flipit == "flip":
        img, _ = flip(img, 0, 2.0, fake=fake)
    return img


def extract_samples_from_rows(data, img_index=0, angle_offset=0, fake=False, rand=False, augment=False
                              ):
    """Extract image data, steering angles from csv rows"""
    if not img_index:
        img_index = settings["IMAGE_INDEX"]
    if not angle_offset:
        angle_offset = settings["STEERING_CORRECTION_COEFF"]

    if type(img_index) == int:
        img_index = [img_index, ]
    if type(angle_offset) == int:
        angle_offset = [angle_offset, ]
    images = []
    angles = []
    if augment:
        aug_image = create_aug_img_pipeline(fake=fake,
                                        accepted_angles=settings["STEERING_CORRECTION_COEFF"],
                                        augment=augment)

    for row in data:
        angle = float(row[1])
        if row[-1] == "flip":
            angle = -angle
        # deprecated random choice
        # if rand:
        #     index = random.randint(0, len(img_index) - 1)
        img = get_img(row[0], row[-2], row[-1], fake)

        images.append(img)
        angles.append(angle)

        if augment:
            img_aug, angle_aug = aug_image(img, angle)
            images.append(img_aug)
            angles.append(angle_aug)

        # deprecated. mutate row functionality centralized to parse_logs
        # else:
        #     images.extend([get_img(row[i], row[-2], row[-1], fake) for i in img_index])
        #     angles.extend([angle + offset for offset in angle_offset])
    return (images, angles)


def generate_batch(data, augment=True, batch_size=128, fake=False, flipit=None,
                   rand_camera=None):
    """generates batches of images for keras model"""

    if flipit is None:
        flipit = settings["DO_FLIP"]
    if rand_camera is None:
        rand_camera = (not settings["CHOOSE_ALL_CAMERAS"])

    aug_image = create_aug_img_pipeline(fake=fake,
                                        accepted_angles=settings["STEERING_CORRECTION_COEFF"],
                                        augment=augment)
    while True:
        np.random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            # adjusting for all camera images

            images, steers = extract_samples_from_rows(batch, settings["IMAGE_INDEX"],
                                                       settings["STEERING_CORRECTION_COEFF"],
                                                       fake=fake, rand=rand_camera)

            images_a, steers_a = zip(*list(map(aug_image, images, steers)))
            images_a, steers_a = list(images_a), list(steers_a)

            # deprecated. putting flipit functionality to log parser to randomize starting data
            # if flipit:
            #     flipped_images = []
            #     flipped_steers = []
            #     for i in range(len(images_a)):
            #         _img, _steer = aug_image(images_a[i], steers_a[i], flipit=True)
            #         flipped_images.append(_img)
            #         flipped_steers.append(_steer)
            #     images_a.extend(flipped_images)
            #     steers_a.extend(flipped_steers)

            if settings["SHUFFLE"]:
                np.random.shuffle(images_a)
                np.random.shuffle(steers_a)
            yield (np.array(images_a), np.array(steers_a))


def get_data(limit=None, replicate=[], split_valid=None):
    if limit:
        data = parse_logs(limit, replicate_from_folders=replicate, flipit=settings["DO_FLIP"])
    else:
        data = parse_logs(replicate_from_folders=replicate, flipit=settings["DO_FLIP"])
    if settings["SHUFFLE"]:
        np.random.shuffle(data)
    # TODO generalize
    correct = settings["SAMPLE_SIZE_CORRECTION"]
    if split_valid:
        bound = int(len(data) * split_valid)
        valid, train = data[:bound], data[bound:]
        return (train, len(train) * correct, valid, len(valid) * correct)
    sample_size = len(data) * correct
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


def visialize_data_chunk(data):
    i_st = random.randint(0 + len(data) // 30, len(data) - 1)

    data_pl = data[i_st:i_st + 10]

    images, steers = extract_samples_from_rows(data_pl)
    print (steers)
    for img, angle in zip(images, steers):
        print(angle)
        plot_img(img)


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
