import glob
from PIL import Image
import numpy as np

def get_label(genre):
    if genre == 'black':
        label = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif genre == 'core':
        label = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif genre == 'death':
        label = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif genre == 'doom':
        label = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif genre == 'gothic':
        label = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif genre == 'heavy':
        label = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif genre == 'pagan':
        label = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif genre == 'power':
        label = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif genre == 'progressive':
        label = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    elif genre == 'thrash':
        label = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    else:
        return KeyError
    return label


def load_data(n_images):
    n_images = n_images - 1
    data = np.empty((n_images, 64, 128, 3), dtype='float32')  # number of images, n channels (3 = RGB), w, h
    label = np.empty((n_images, 10), dtype='uint8')
    i = 0
    # categorical_labels = to_categorical(int_labels, num_classes=None)
    for filename in glob.glob('./preprocessed_imgs_all/*.jpg'):
        if i == n_images:
            break
        try:
            img = Image.open(filename)
            arr = np.asarray(img, dtype='float32')
            genre = filename.rsplit('/', 1)[-1].rsplit('_', 1)[0]
            # band_nr = filename.rsplit('/', 1)[-1].rsplit('_', 1)[1].rsplit('.', 1)[0]
            print(arr.shape)
            data[i, :, :, :] = arr
            label[i] = get_label(genre)
            i += 1
        except OSError:
            print('OSError caused by: ', img)
        except KeyError:
            print('OSError caused by: ', img, genre)
    return data, label


dat = load_data(43511)
# get_label('black')

print(dat)
