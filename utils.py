import tensorflow as tf
import tensorflow_hub as tfhub
import os
import pandas as pd


def sort_labels(df, col_id, imgs_path):
    # the Keras utils function 'image_dataset_from_directory' takes as labels a list in the same order as returned by 'os.walk'
    # therefore the goal of this function is to ensure that the order the quantitative information on each picture is according
    # to the result of 'os.walk' on the respective directory.
    img_names = [fn.split(".")[0] for fn in list(os.walk(imgs_path))[0][2]]
    df["Id_for_ordering"] = pd.Categorical(
        df[col_id], categories=img_names, ordered=True
    )
    return df


def build_interm_models(
    interm_label_var, img_path, df, model_handle, img_size, hub_config
):
    # first step: create the image dataset with the appropriate label
    img_ds = tf.keras.utils.image_dataset_from_directory(
        directory=img_path,
        # label_mode='binary',
        labels=df[interm_label_var].tolist(),
        subset="training",
        image_size=img_size,
        validation_split=0.3,
        seed=42,
    )
    # second step: build the model

    print(
        "Building model with", model_handle, "for variable", interm_label_var, img_size
    )
    model = tf.keras.Sequential(
        [
            # Explicitly define the input shape so the model can be properly
            # loaded by the TFLiteConverter
            tf.keras.layers.InputLayer(input_shape=img_size + (3,)),
            tfhub.KerasLayer(model_handle, trainable=hub_config["do_fine_tuning"]),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            ),
        ]
    )
    model.build((None,) + img_size + (3,))
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1),
        metrics=["accuracy"],
    )

    # third step: train the model with the dataset built on the first step
    # steps_per_epoch = len(img_ds) // BATCH_SIZE
    # validation_steps = valid_size // BATCH_SIZE
    hist = model.fit(
        img_ds,
        epochs=hub_config["epochs"],
        #    steps_per_epoch=steps_per_epoch,
    ).history

    # all set - return the trained model
    return model
