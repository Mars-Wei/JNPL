import numpy as np
import pandas as pd
import tensorflow as tf
from JNPL.model import get_my_EFFmodel
from tensorflow import keras
from JNPL.loss import CustomLoss
'''
basic parameter config
'''
img_width = 480
img_height = 480

def train( save_dir, csv_dir, data_dir, best_path, epochs, batch_size, portion,log_dir,begin_epoch='no'):
    """
    call backs
    """
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
    logger = keras.callbacks.CSVLogger(csv_dir, separator=',', append=False)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_dir,
        verbose=1,
        save_weights_only=True,
        period=1)
    # learning rate schedule
    def decay(epoch):
      if epoch < 3:
        return 1e-4
      elif epoch >= 3 and epoch < 7:
        return 1e-5
      else:
        return 1e-6

    '''
    # load the data and define the save dir and callback function
    '''

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=1 - portion,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=1 - portion,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

    class_nums = len(train_ds.class_names)
    print('the data has been loaded')

    '''
    load the EFF model and to create my own model
    '''
    model = get_my_EFFmodel(img_height, img_width, class_nums)
    my_loss = CustomLoss(class_num=class_nums)
    model.compile(optimizer=keras.optimizers.Adam(), loss=my_loss, metrics=['accuracy'])

    print(model.summary())

    '''
    model training
    '''
    # prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # train
    history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[cp_callback,
                        early_stop,
                        logger,
                        tf.keras.callbacks.LearningRateScheduler(decay)]
        )
    model.save_weights(best_path)
    print("training finish==================================================================================")

    names_df = pd.DataFrame(class_names).set_index(0)
    names_df.to_csv(log_dir)
