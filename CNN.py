import csv
import cv2 as cv
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from imgaug import augmenters as img

def preview(images, labels):
    plt.figure(figsize=(16, 16))
    for c in range(len(np.unique(labels))):
        i = random.choice(np.where(labels == c)[0])
        plt.subplot(10, 10, c+1)
        plt.axis('off')
        plt.title('class: {}'.format(c))
        plt.imshow(images[i])
    plt.tight_layout()
    plt.show()

def count_images_in_classes(lbls):
    dct = {}
    for i in lbls:
        if i in dct:
            dct[i] += 1
        else:
            dct[i] = 1
    return dct

def distribution_diagram(dct):
    plt.title("Images per Class")
    plt.bar(range(len(dct)), list(dct.values()), align='center')
    plt.xticks(range(len(dct)), list(dct.keys()), rotation=90, fontsize=7)
    plt.show()


def augment_imgs(imgs, p):

    from imgaug import augmenters as iaa
    augs = iaa.SomeOf((2, 4),
                      [
                          iaa.Crop(px=(0, 4)),
                          iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                          iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                          iaa.Affine(rotate=(-45, 45)),
                          iaa.Affine(shear=(-10, 10))
                      ])

    seq = iaa.Sequential([iaa.Sometimes(p, augs)])
    res = seq.augment_images(imgs)
    return res


def augmentation(imgs, lbls):
    classes = count_images_in_classes(lbls)
    for i in range(len(classes)):
        if (classes[i] < MIN_IMGS_IN_CLASS):

            add_num = MIN_IMGS_IN_CLASS - classes[i]
            imgs_for_augm = []
            lbls_for_augm = []
            for j in range(add_num):
                im_index = random.choice(np.where(lbls == i)[0])
                imgs_for_augm.append(imgs[im_index])
                lbls_for_augm.append(lbls[im_index])
            augmented_class = augment_imgs(imgs_for_augm, 1)
            augmented_class_np = np.array(augmented_class)
            augmented_lbls_np = np.array(lbls_for_augm)
            imgs = np.concatenate((imgs, augmented_class_np), axis=0)
            lbls = np.concatenate((lbls, augmented_lbls_np), axis=0)
    return (imgs, lbls)

# Hyperparameters
EPOCHS = 10
INIT_LR = 0.001
BATCH_SIZE = 256
SET_DECAY = True
MIN_IMGS_IN_CLASS = 500


images = []
labels = []

# Setup
gtFile = open('./Train.csv')
gtReader = csv.reader(gtFile, delimiter=',')
next(gtReader)

for row in gtReader:
    # need this for missing images
    try:
        img = cv.imread('./' + row[7])
        images.append(cv.resize(img, (28, 28)))
        labels.append(row[6])
    except:
        pass
gtFile.close()

train_X = np.asarray(images)
train_X = train_X / 255
train_X = np.asarray(train_X, dtype = "float32")
train_Y = np.asarray(labels, dtype= "float32")
train_X, train_Y = augmentation(train_X, train_Y)

train_X = rgb2gray(train_X)



# More hyperparameters again
height = 28
width = 28
depth = 1
classes = 43

# Model
model = keras.Sequential()
inputShape = (height, width, depth)
chanDim = -1

# CONV init
model.add(Conv2D(8, (5, 5), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Conv layer 1
model.add(Conv2D(16, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(16, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Conv layer 2
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten 1
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Flatten 2
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Softmax activation
model.add(Dense(classes))
model.add(Activation("softmax"))

opt = Adam(learning_rate=INIT_LR, weight_decay=INIT_LR / (EPOCHS * 0.5))
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# Test
test_images = []
test_labels = []

# loop over all 43 classes
gtFile = open('./Test.csv')
gtReader = csv.reader(gtFile, delimiter=',')
next(gtReader)

for row in gtReader:
    img = cv.imread('./' + row[7])
    test_images.append(cv.resize(img, (28, 28)))
    test_labels.append(row[6]) # the 6th column is the label
gtFile.close()

test_X = np.asarray(test_images)
test_X = test_X / 255
test_X = np.asarray(test_X, dtype = "float32")
test_X = rgb2gray(test_X)

test_Y = np.asarray(test_labels, dtype = "float32")

train_X_ext = np.expand_dims(train_X, axis=3)
test_X_ext = np.expand_dims(test_X, axis=3)

H = model.fit(train_X_ext, train_Y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_X, test_Y))

test_loss, test_acc = model.evaluate(test_X_ext, test_Y, verbose=1)

plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()