import sentinelhub
import pandas as pd
from sentinelhub import SHConfig
from datetime import timedelta
from datetime import datetime
from ast import literal_eval
import time
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataSource, bbox_to_dimensions, DownloadRequest
import glob
from ref_data_setup import string_fixer

owd = os.getcwd()

# In case you put the credentials into the configuration file you can leave this unchanged

#CLIENT_ID = 'd081cc69-29cb-4d6c-a240-f0c34f412245'
CLIENT_ID = '9a0eaca6-c18f-4fff-a73f-98722c7a080e'
CLIENT_SECRET = 'afGv}BMM8(^1y5Y^]w2Vz?9e%4#w.o5c49}V,HL9'
#CLIENT_SECRET = 'kDu&@Xb8cvq9#8%wq$>#FUysBZ>QdLK!@s8@O1-v'

config = SHConfig()

if CLIENT_ID and CLIENT_SECRET:
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

if config.sh_client_id == '' or config.sh_client_secret == '':
    print("Warning! To use Sentinel Hub services, please provide the credentials (client ID and client secret).")


# +- 0.005, +- 0.005 on each coordinate for boundaries


def plot_image(image, factor=1.0, clip_range=None, **kwargs):
    """
    Utility function for plotting RGB images.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])


def sentinel_request(coords, date_range):
    resolution = 30
    fire_bbox = BBox(bbox=coords, crs=CRS.WGS84)
    fire_box_size = bbox_to_dimensions(fire_bbox, resolution=resolution)

    print(f'Image shape at {resolution} m resolution: {fire_box_size} pixels')

    evalscript_true_color = """
        //VERSION=3

        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"]
                }],
                output: {
                    bands: 3
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
    """

    request_true_color = SentinelHubRequest(
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_source=DataSource.LANDSAT8_L1C,
                time_interval=date_range,
                mosaicking_order='leastCC'
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.PNG)
        ],
        bbox=fire_bbox,
        size=fire_box_size,
        config=config
    )

    true_color_imgs = request_true_color.get_data()

    print(f'Returned data is of type = {type(true_color_imgs)} and length {len(true_color_imgs)}.')
    print(
        f'Single element in the list is of type {type(true_color_imgs[-1])} and has shape {true_color_imgs[-1].shape}')

    image = true_color_imgs[0]
    print(f'Image type: {image.dtype}')
    return image


def reshaper(array):
    array_dim1 = array.shape[0]
    array_dim2 = array.shape[1]
    edgesA = (array_dim1 - 60) / 2
    edgesB = (array_dim2 - 60) / 2
    boundA1 = int(edgesA - 1)
    boundA2 = int(array_dim1 - edgesA - 1)
    boundB1 = int(edgesB - 1)
    boundB2 = int(array_dim2 - edgesB - 1)
    new_array = array[boundA1:boundA2, boundB1:boundB2, 0:3]
    return new_array


def request_batch(folder_name, type):
    os.chdir('Data/' + str(folder_name) + '/' + str(type))
    file = glob.glob("*.csv")[0]
    ref_df = pd.read_csv(file)

    image_list = []
    for i in range(len(ref_df)):
        coords = literal_eval(ref_df['BBOX'][i])
        date = ref_df['Date'][i]
        date_range = (str(datetime.datetime.strptime(date, '%Y-%m-%d').date() - timedelta(days=180)), str(date))
        image = sentinel_request(coords, date_range)
        # plot_image(image, factor=3.5 / 255, clip_range=(0, 1))
        # plt.show()
        reshaped_image = reshaper(image)
        if reshaped_image.shape == (60, 60, 3):
            print('No shape issue')
            image_list.append(reshaped_image)
            last_image = reshaped_image
        else:
            print('SHAPE_ISSUE')
            image_list.append(last_image)
            print(reshaped_image.shape)
        print('new image array shape is ' + str(reshaped_image.shape))
        time.sleep(.1)
        print('~~~~~~ DOWNLOADED ' + str(folder_name) + ' ' + str(type) + ' IMAGE ' + str(i) + ' ~~~~~~')

    array_file = str(folder_name) + '_' + str(type)
    np.save(array_file, image_list)
    os.chdir(owd)


def main_function():
    owd = os.getcwd()
    # Input location of pre-processed dataset
    file_path = '2014-2015_raw.csv'
    df = pd.read_csv(file_path)

    list_of_causes = list(set(df['STAT_CAUSE_DESCR']))
    folders_list = [string_fixer(n) for n in list_of_causes]

    folders_list = ['Lightning']

    for item in folders_list:
        for type in ['train', 'test']:
            image_list = request_batch(item, type)
            print('images (numpy arrays) saved for ' + str(item) + ' ' + str(type))

    return image_list


if __name__ == '__main__':
    main_function()







