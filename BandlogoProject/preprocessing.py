import json
import glob
from PIL import Image
import string


genres = []


def read_band_data(letter):
    with open("./half_clean_info/{}.json".format(letter), encoding='utf-8-sig') as json_file:
        json_data = json.load(json_file)

    return json_data


def get_all_genres():
    letters = list(string.ascii_lowercase)
    for letter in letters:
        json_bands = read_band_data(letter)
        last_band_index = len(json_bands) - 1
        print(band)
        for i in range(last_band_index):
            band = json_bands[i]
            genres.append(band["genre"])


#print(images)

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

#print(img.info)



# print(len(json_data))
# print(json_data)

'''last_band_index = len(json_bands) - 1
band = json_bands[last_band_index]
print(band)

print(band["genre"])
url = band["url"]
band_id = url.rsplit('/', 1)[-1]
print(band_id)'''
