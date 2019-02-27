import keras
import keras.backend as K

IMG_SIZE = 160
DENSE_SHAPE = 4096

def oneshot_bsm():
    """
    pre-trained baseline model
    :return: a keras model object
    """
    pre_model = keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_shape=[IMG_SIZE, IMG_SIZE, 3])
    flat = keras.layers.Flatten()(pre_model.output)
    dens = keras.layers.Dense(DENSE_SHAPE, activation="sigmoid", kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer=keras.initializers.truncated_normal(0, 0.01))(flat)
    model = keras.Model(inputs=pre_model.input, outputs=dens)
    return model

def oneshot_sim():
    """
    top layer of Siamese net
    :return: a keras model object
    """
    input_1 = keras.Input(shape=[DENSE_SHAPE,])
    input_2 = keras.Input(shape=[DENSE_SHAPE,])
    ### calculate distance
    diff_1 = keras.layers.Lambda(lambda x: x[0] * x[1])([input_1, input_2])
    diff_2 = keras.layers.Lambda(lambda x: x[0] + x[1])([input_1, input_2])
    diff_3 = keras.layers.Lambda(lambda x: K.abs(x[0] - x[1]))([input_1, input_2])
    diff_4 = keras.layers.Lambda(lambda x: K.square(x))(diff_3)
    diff_5 = keras.layers.Lambda(lambda x: K.abs(K.log(x[0]/x[1])))([input_1, input_2])
    diff = keras.layers.Concatenate()([diff_1, diff_2, diff_3, diff_4, diff_5])
    diff = keras.layers.Reshape([5, DENSE_SHAPE, 1])(diff)
    diff = keras.layers.Conv2D(128, (5, 1), activation='linear', padding='valid')(diff)
    diff = keras.layers.Conv2D(1, 1, activation='linear', padding='valid')(diff)
    diff = keras.layers.Flatten()(diff)
    output = keras.layers.Dense(1, activation="sigmoid")(diff)
    model = keras.Model(inputs=[input_1, input_2], outputs=output)
    return model

def oneshot_inference(base_model, sim):
    """
    final model
    :param base_model: pre-trained baseline model
    :param sim: top layer of Siamese net
    :return: a keras model object
    """
    input_1 = keras.Input(shape=[IMG_SIZE, IMG_SIZE, 3])
    input_2 = keras.Input(shape=[IMG_SIZE, IMG_SIZE, 3])
    flat_1 = base_model(input_1)
    flat_2 = base_model(input_2)
    y = sim([flat_1, flat_2])
    model = keras.Model(inputs=[input_1, input_2], outputs=y)
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam())
    return model

def nw_inference():

    """
    build model

    :return: a keras model object
    """

    base_model = keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_shape=[IMG_SIZE, IMG_SIZE, 3])
    x = keras.layers.Flatten()(base_model.output)
    x = keras.layers.Dense(1024, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    #x = keras.layers.Dense(256, activation="relu")(x)
    #x = keras.layers.Dropout(0.5)(x)
    #x = keras.layers.Dense(128, activation="relu")(x)
    y = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=base_model.input, outputs=y)
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam())
    return model