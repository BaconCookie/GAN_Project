import json
import glob
from PIL import Image
import string
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# im_modes = []
# im_sizes = []


undecided = []


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
    #ax.set_yticklabels(bands)
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


def decide_genre(original):
    original = original.lower()
    if 'core' in original or 'grind' in original or 'nu' in original:
        genre = 'core'
    elif 'gothic' in original or 'avant-garde' in original:
        genre = 'gothic'
    elif 'pagan' in original or 'viking' in original or 'folk' in original or 'celtic' in original:
        genre = 'pagan'
    elif 'black' in original:
        genre = 'black'
    elif 'death' in original:
        genre = 'death'
    elif 'thrash' in original or 'groove' in original:
        genre = 'thrash'
    elif 'heavy' in original or 'nwobhm' in original:
        genre = 'heavy'
    elif 'power' in original or 'speed' in original:
        genre = 'power'
    elif 'doom' in original or 'stoner' in original or 'sludge' in original or 'depressive' in original or 'dark' in original:
        genre = 'doom'
    elif 'prog' in original or 'experimental' in original or 'shred' in original:
        genre = 'progressive'
    elif 'symphonic' in original:
        genre = 'gothic'
    elif 'post' in original or 'shoegaze' in original:
        genre = 'black'
    else:
        genre = 'undecided'
        undecided.append(original)
    return genre


def prepoces_imgs():
    bands_dict = band_data_to_dict()
    image_list = []
    genres = []
    letters = list(string.ascii_lowercase)
    i = 0
    for letter in letters:
        for filename in glob.glob('./img/{}/*.jpg'.format(letter)):
            try:
                # Preprocess image
                im_id = filename.rsplit('/', 1)[-1].rsplit('.', 1)[0]
                im = Image.open(filename)
                im = im.resize([128, 64])
                im = im.convert('RGB')  # convert to RGB
                # im = im.convert('1')  # convert to black and white

                # im_modes.append(im.mode)
                # im_sizes.append(im.size)
                try:
                    band_info = bands_dict[im_id]
                    original_genre = band_info['genre']
                    genre = decide_genre(original_genre)
                    if genre is not 'undecided':
                        i += 1
                        # name = band_info['name']
                        # name = name.translate({ord(c): " " for c in "!@#$%^&*()[]{};:.,/<>?\|`~-=_+"})# Remove special chars
                        # genres.append(name)
                        # im.save('./preprocessed_img/{}/{}_{}.jpg'.format(genre, name, im_id), 'JPEG')
                        im.save('./preprocessed_imgs_all/{}_{}.jpg'.format(genre, i), 'JPEG')

                    else:
                        undecided.append(band_info)
                except KeyError:
                    print('Band with id {} throws KeyError. Band info: '.format(im_id), band_info)
            except OSError:
                print('OSError caused by: ', filename, band_info)
    # print(Counter(genres))
    # print(Counter(undecided))


# pre = prepoces_imgs()
plot_genres()

# ----------------------------------------------------------------------
# Put all band info in a dictionary, save as file
#
# d = band_data_to_dict()
#
# with open('bandinfo_dict.json', 'w') as f:
#      f.write(json.dumps(d))
# ----------------------------------------------------------------------

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
# prepoces_imgs()
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
# prepoces_imgs()
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
