from numpy import array

try:
    import os
    import json
    import glob
    import argparse
    import cv2
    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage import maximum_filter
    import typing
    from PIL import Image, ImageDraw
    import pandas as pd
    import tables
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

kernel = np.asarray(
    [[-69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58,
      -69 / 58],
     [-69 / 58, -69 / 58, -69 / 58, 1, 1, 1, 1, 1, 1, -69 / 58, -69 / 58, -69 / 58],
     [-69 / 58, -69 / 58, 1, 1, 1, 1, 1, 1, 1, 1, -69 / 58, -69 / 58],
     [-69 / 58, -69 / 58, 1, 1, 1, 1, 1, 1, 1, 1, -69 / 58, -69 / 58],
     [-69 / 58, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -69 / 58],
     [-69 / 58, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -69 / 58],
     [-69 / 58, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -69 / 58],
     [-69 / 58, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -69 / 58],
     [-69 / 58, -69 / 58, 1, 1, 1, 1, 1, 1, 1, 1, -69 / 58, -69 / 58],
     [-69 / 58, -69 / 58, 1, 1, 1, 1, 1, 1, 1, 1, -69 / 58, -69 / 58],
     [-69 / 58, -69 / 58, -69 / 58, 1, 1, 1, 1, 1, 1, -69 / 58, -69 / 58, -69 / 58],
     [-69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58,
      -69 / 58]])

data = {'seq':[],'is_true':[],'is_ignore':[],'path':[],'x0':[],'x1':[],'y0':[],'y1':[],'col':[]}
df=pd.DataFrame(data)

# def find_tfl_lights(c_image: np.ndarray, **kwargs):
#     """
#     Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
#     :param c_image: The image itself as np.uint8, shape of (H, W, 3)
#     :param kwargs: Whatever config you want to pass in here
#     :return: 4-tuple of x_red, y_red, x_green, y_green
#     """
#     return [500, 510, 520], [500, 500, 500], [700, 710], [500, 500]

def read_table():
    data = pd.read_hdf("../attention_results.h5")
    pd.set_option('display.max_rows', None)
    print(data)
    return data


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def search_by_color(image: np.ndarray, color: str) -> (array, typing.List):
    """
    Highlights the relevant color, convolutes the image and filter the list of attentions from it.
    :param image: The image that transferred for processing.
    :param color: The color of the attentions that should be found.
    :return: The processed image and the list of attentions.
    """
    if color == "red":
        img_color = image[:, :, 0]
        con = sg.convolve(img_color, kernel, mode='same')
        lst = np.argwhere(maximum_filter(con, 5) > 7000)
    # else:
    #     img_color = image[:, :, 1]
    #     con = sg.convolve(img_color, kernel, mode='same')
    #     lst = np.argwhere(maximum_filter(con, 30) > 7000)
    return con, lst


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    read_table()
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    # show_image_and_gt(image, objects, fig_num)
    convolved_image, lst_red = search_by_color(image, "red")
    # convolved_image, lst_green = search_by_color(image, "green")

    plt.figure(57)
    plt.clf()
    h = plt.subplot(111)
    plt.imshow(convolved_image)

    plt.figure(52)
    plt.clf()
    plt.subplot(111, sharex=h, sharey=h)
    plt.imshow(Image.open(image_path))
    for i in lst_red:
        red_y, red_x = i
        plt.plot(red_x, red_y, 'ro', color='r', markersize=1)
    # for i in lst_green:
    #     green_y, green_x = i
    #     plt.plot(green_x, green_y, 'ro', color='g', markersize=1)

    plt.figure(55)
    plt.clf()
    plt.subplot(111, sharex=h, sharey=h)
    plt.imshow(convolved_image > 7000)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = "../images"

    if args.dir is None:
        args.dir = default_base
    # flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    flist = [r"C:\Users\IMOE001\images\aachen_000010_000019_leftImg8bit.png"]
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


def crop_image(image, x_left, y_up, w, h):
    image_crop = image.crop((x_left, y_up, x_left + w, y_up + h))
    # image_crop.show()
    return image_crop, x_left, y_up, w, h


def cal_size(path, x, y, zoom, color):
    x_100 = 20.0
    x_zoom = x_100 * (1.0 - zoom)
    y_zoom = 0.0
    if color == 'r':
        y_zoom = x_zoom
    elif color == 'g':
        y_100 = 100.0
        y_zoom = y_100 * (1.0 - zoom)

    x_left = x - x_zoom
    y_up = y - y_zoom
    image = Image.open(path)
    if zoom == 0.0000:
        return crop_image(image, x_left, y_up, 40, 120)
    elif zoom == 0.0625:
        return crop_image(image, x_left, y_up, 37.5, 112.5)
    elif zoom == 0.1250:
        return crop_image(image, x_left, y_up, 35, 105)
    elif zoom == 0.2500:
        return crop_image(image, x_left, y_up, 30, 90)
    elif zoom == 0.5000:
        return crop_image(image, x_left, y_up, 20, 60)


def cropping():
    table = read_table()
    # for i in range(table.shape[0]):
    for i in range(len(table)-5909):
        path = table.iloc[4, :]['path']
        x = table.iloc[4, :]['x']
        y = table.iloc[4, :]['y']
        zoom = table.iloc[4, :]['zoom']
        col = table.iloc[4, :]['col']
        lib = path.split("_")[0]
        path = "..\\leftImg8bit_trainvaltest\\leftImg8bit\\train\\" + lib + "\\" + path
        crop, x_left_top, y_left_top, w, h = cal_size(path, x, y, zoom, col)
        x_right_bottom = x_left_top + w
        y_right_bottom = y_left_top + h
        index=table.index(i)
        color_image_path = path.replace("_leftImg8bit.png", "_gtFine_color.png")
        color_image = Image.open(color_image_path)

        color=np.array(color_image)[int(y)][int(x)]
        if color.all != [250, 170, 30, 255]:
            flag = "False"
            # TODO: insert into table "insert_to_table"
            continue

        color_image = color_image.crop((x_left_top, y_left_top, x_right_bottom, y_right_bottom))
        color_image_array = np.array(color_image)
        color_precent = (len(np.argwhere(color_image_array == [250, 170, 30, 255])) / (color_image_array.size)) * 100

        if color_precent > 80:
            flag = "True"
            # TODO: insert into table
            print(flag)
        else:
            flag = "Ignore"
            # TODO: insert into table

        plt.figure(57)
        plt.clf()
        plt.subplot(111)
        plt.imshow(crop)

        plt.figure(59)
        plt.clf()
        plt.subplot(111)
        plt.imshow(color_image)

        plt.show(block=True)


if __name__ == '__main__':
    # main()
    # read_table()
    cropping()
    print(df)
