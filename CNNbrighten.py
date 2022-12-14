
import pandas as pd
from tqdm import tqdm

import os
import shutil
import pathlib


import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.activations import softmax, relu
from tensorflow.keras.initializers import zeros
from tensorflow.keras.layers import Flatten, Dropout, Input, MaxPool2D, Dense, BatchNormalization, Conv2D






# Hyperparameters
VAL_BATCH_SIZE = 32
CLASS_NAMES = list(range(43))
N_CLASSES = 43
train_path = "./Train"
val_path = "./Validation"
IMG_WIDTH = 30
IMG_HEIGHT = 30
N_CHANNELS = 3
BATCH_SIZE = 32
N_EPOCHS = 200


# Split the data into train val
data_root_train = pathlib.Path(train_path)
data_root_val = pathlib.Path(val_path)

all_image_paths_train = list(data_root_train.glob('*/*'))
all_image_paths_train = [str(path) for path in all_image_paths_train]

all_image_paths_val = list(data_root_val.glob('*/*'))
all_image_paths_val = [str(path) for path in all_image_paths_val]

# Generating secondary info
image_count_train = len(all_image_paths_train)
image_count_val = len(all_image_paths_val)

label_names_train = sorted(int(item.name) for item in data_root_train.glob('*/') if item.is_dir())
label_names_val = sorted(int(item.name) for item in data_root_val.glob('*/') if item.is_dir())
label_to_index_train = dict((name, index) for index,name in enumerate(label_names_train))
label_to_index_val = dict((name, index) for index,name in enumerate(label_names_val))
all_image_labels_train = [label_to_index_train[int(pathlib.Path(path).parent.name)] for path in all_image_paths_train]
all_image_labels_val = [label_to_index_val[int(pathlib.Path(path).parent.name)] for path in all_image_paths_val]

# load data now
df_train = pd.read_csv("Train.csv")

# Fix rounding errors
for idx, row in df_train.iterrows() :
    w = row['Width']
    h = row['Height']
    if w > IMG_WIDTH :
        diff = w-IMG_WIDTH
        df_train.iloc[idx, 4] = df_train.iloc[idx]['Roi.X2'] - diff
    else :
        diff = IMG_WIDTH-w
        df_train.iloc[idx, 4] = df_train.iloc[idx]['Roi.X2'] + diff
    if h > IMG_HEIGHT :
        diff = h - IMG_HEIGHT
        df_train.iloc[idx, 5] = df_train.iloc[idx]['Roi.Y2'] - diff
    else :
        diff = IMG_HEIGHT - h
        df_train.iloc[idx, 5] = df_train.iloc[idx]['Roi.Y2'] + diff

train_idx_list = []
val_idx_list = []

for path_tr in tqdm(all_image_paths_train) :
    train_idx_list.append(df_train[df_train['Path'] == path_tr[14 : ]].index[0])
for path_val in tqdm(all_image_paths_val) :
    path_val = "Train/" + path_val[25:]
    val_idx_list.append(df_train[df_train['Path'] == path_val].index[0])

new_df_train = pd.DataFrame()
new_df_val = pd.DataFrame()

new_df_train = new_df_train.append(df_train.iloc[train_idx_list], ignore_index = True)
new_df_val = new_df_val.append(df_train.iloc[val_idx_list], ignore_index = True)

new_df_train = new_df_train.drop(['Height', 'Width', 'ClassId', 'Path'], axis = 1)
new_df_val = new_df_val.drop(['Height', 'Width', 'ClassId', 'Path'], axis = 1)

# Generate tf data, add additional images
def tfdata_generator(images, labels, df, is_training, batch_size=32):
    def parse_function(filename, labels, df):

        image_string = tf.io.read_file(filename)

        image = tf.image.decode_png(image_string, channels=N_CHANNELS)
        # Flatten
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Brighten
        if tf.math.reduce_mean(image) < 0.3 :
            image = tf.image.adjust_contrast(image, 5)
            image = tf.image.adjust_brightness(image, 0.2)
        # Resize
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], method="nearest", preserve_aspect_ratio=False)
        image = image/255.0

        return image, {"classification" : labels, "regression" : df}

    dataset = tf.data.Dataset.from_tensor_slices((images, labels, df))
    if is_training:
        dataset = dataset.shuffle(25000)

    dataset = dataset.map(parse_function, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


# Data generators :
tf_image_generator_train = tfdata_generator(all_image_paths_train, all_image_labels_train, new_df_train, is_training=True, batch_size=32)
tf_image_generator_val = tfdata_generator(all_image_paths_val, all_image_labels_val, new_df_val, is_training=False, batch_size=32)