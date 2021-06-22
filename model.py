import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
def get_my_EFFmodel(img_height, img_width, class_nums, checkpoint = None):
    # load the EFF_model

    EFF_model = tf.keras.applications.EfficientNetB4(
        include_top=False, weights='imagenet', input_tensor=None,
        input_shape=(img_height, img_width, 3), pooling=None,
        classifier_activation=None,
    )

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
            layers.experimental.preprocessing.RandomRotation(0.3),
            layers.experimental.preprocessing.RandomZoom(0.1)
        ]
    )

    # classifier
    classifier = keras.Sequential([
        layers.Dropout(0.2),
        layers.Dense(class_nums, activation='softmax')]
    )

    # normalization
    norm_layer = keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3))

    # create my own model and compile
    inputs = keras.Input(shape=(img_width, img_height, 3))

    x = data_augmentation(inputs)
    x = norm_layer(x)
    x = EFF_model(x, training=True)
    x = keras.layers.GlobalAveragePooling2D()(x)

    outputs = classifier(x)
    model = keras.Model(inputs, outputs)
    if checkpoint is not None:
        model.load_weights(checkpoint)
    model.trainable = True

    return model
