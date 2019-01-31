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


def get_all_genres():
    genres = []
    letters = list(string.ascii_lowercase)
    for letter in letters:
        json_bands = read_band_data(letter)
        last_band_index = len(json_bands) - 1
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

    #plt.show()
    plt.savefig('all_the_genres.png')


    # with sns.axes_style('white'):
    #     g = sns.catplot(y = counted.values(), data=counted, aspect=2,
    #                        kind="count", color='steelblue')
    #     g.set_xticklabels(step=5)



def prepoces_imgs(letter):
    image_list = []
    images = {}
    for filename in glob.glob('./img/{}/*.jpg'.format(letter)):
        id = filename.rsplit('/', 1)[-1].rsplit('.', 1)[0]
        im = Image.open(filename)
        im = im.resize([128, 128]) #Todo resize properly with padding
        #im = im.convert('1')  # convert to black and white
        images[id] = im

    print(images)


# genres = get_all_genres()
#
# with open('all_genres_listed.txt', 'w') as f:
#     f.write(json.dumps(genres))

plot_genres()


# print(len(json_data))
# print(json_data)

'''last_band_index = len(json_bands) - 1
band = json_bands[last_band_index]
print(band)

print(band["genre"])
url = band["url"]
band_id = url.rsplit('/', 1)[-1]
print(band_id)'''
