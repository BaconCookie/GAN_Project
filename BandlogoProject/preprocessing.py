import json
import glob
from PIL import Image
import string
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def read_band_data(letter):
    with open("./half_clean_info/{}.json".format(letter), encoding='utf-8-sig') as json_file:
        json_data = json.load(json_file)

    return json_data


def read_all_genres():
    with open('all_genres_listed.json', 'r') as json_file:
        json_data = json.load(json_file)
    return json_data


def get_all_genres():
    genres = []
    letters = list(string.ascii_lowercase)
    for letter in letters:
        json_bands = read_band_data(letter)
        for i in range(len(json_bands)):
            band = json_bands[i]
            genres.append(band["genre"])
    counted = Counter(genres)
    sorted_genres = sorted(counted.items(), key=lambda kv: kv[1])
    sorted_genres = sorted_genres[::-1]
    sorted_dict = dict(sorted_genres)
    return sorted_dict


def plot_genres():
    fig, ax = plt.subplots()

    # Data
    genres = get_all_genres()

    bands = list(genres.keys())
    values = list(genres.values())
    y_pos = np.arange(len(bands))
    x_pos = np.arange(max(values))

    # Plotting
    ax.barh(y_pos, values, align='center', color='blue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bands)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xticks(x_pos)
    ax.set_xlabel('Frequency of listed genres')

    # plt.show()
    plt.savefig('all_the_genres.png')

    # with sns.axes_style('white'):
    #     g = sns.catplot(y = counted.values(), data=counted, aspect=2,
    #                        kind="count", color='steelblue')
    #     g.set_xticklabels(step=5)


def band_data_to_dict():
    bands = {}
    letters = list(string.ascii_lowercase)
    for letter in letters:
        json_bands = read_band_data(letter)
        for i in range(len(json_bands)):
            band_info = json_bands[i]
            # genre = band_info["genre"]
            url = band_info["url"]
            band_id = url.rsplit('/', 1)[-1]
            bands[band_id] = band_info
    return bands


#im_modes = []
#im_sizes = []


def prepoces_imgs():
    bands = band_data_to_dict()
    image_list = []
    images = {}
    letters = list(string.ascii_lowercase)
    for letter in letters:
        for filename in glob.glob('./img/{}/*.jpg'.format(letter)):
            try:
                im_id = filename.rsplit('/', 1)[-1].rsplit('.', 1)[0]
                im = Image.open(filename)
                im = im.resize([128, 64])
                im = im.convert('RGB')  # convert to RGB
                # im = im.convert('1')  # convert to black and white

                # Todo add method getting data for this specific band # Look for IMG id in band info
                try:
                    band_info = bands[im_id]
                    genre_of_band = band_info['genre']
                    #print(genre_of_band)
                    images[im_id] = im

                    #im_modes.append(im.mode)
                    #im_sizes.append(im.size)
                    # Todo add method getting data for this specific band # Look for IMG id in band info
                    #  band_logo_url = 'https://www.metal-archives.com/images/3/5/4/0/'
                    # Todo add method deciding genre
                    genre = 'x'
                    # Todo save img in appropriate folder
                    im.save('./processed_img/{}/{}.jpg'.format(genre, im_id), 'JPEG')
                except KeyError:
                    print('Band with id {} throws KeyError'.format(im_id))
            except OSError:
                print('OSError caused by: ', filename)


    # with open('all_genres_listed.json', 'w') as f:
    #     f.write(json.dumps(images))
    #print(images)


pre = prepoces_imgs()

# ----------------------------------------------------------------------
# Put all genres in a dictionary, sorted from high to low use frequency
#
# genres = get_all_genres()
#
# with open('all_genres_listed.json', 'w') as f:
#     f.write(json.dumps(genres))
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Mini-script to get some data insight in numbers concerning genres
#
# g = read_all_genres()
# number_of_bands = sum(value >= 1000 for value in g.values())
# print(number_of_bands)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Mini-script to get some data insight in numbers concerning image modes
#
# In order to run, uncomment code concerning im_modes in the code above!
#
# pre = prepoces_imgs()
# print(len(im_modes))
# print('L', im_modes.count('L'))
# print('RGB', im_modes.count('RGB'))
# number_of_modes = Counter(im_modes)
# print(number_of_modes)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Mini-script to get some data insight in numbers concerning image sizes
#
# In order to run, uncomment code concerning im_sizes in the code above!
#
# pre = prepoces_imgs()
#
# number_of_sizes = Counter(im_sizes)
# #print(number_of_sizes)
#
# keys = number_of_sizes.keys()
# print(len(keys))
# h = np.mean([x[0] for x in keys])
# w = np.mean([x[1] for x in keys])
# print(h, w)
# hv = np.var([x[0] for x in keys])
# wv = np.var([x[1] for x in keys])
# print(hv, wv)
# ----------------------------------------------------------------------



# plot_genres()

# genres = read_all_genres()

# b = band_data_to_dict()
# print(b)
# print(b['3739']['url'])
'''band = json_bands[last_band_index]
print(band)

print(band["genre"])
url = band["url"]
band_id = url.rsplit('/', 1)[-1]
print(band_id)'''
