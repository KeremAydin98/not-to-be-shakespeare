import tensorflow as tf

def create_model(max_id):

    model = tf.keras.Sequential([
        tf.keras.layers.GRU(512, return_sequences=True,
                            input_shape=[None, max_id], dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.GRU(512, return_sequences=True,
                            dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(max_id, activation="softmax"))
        # This wrapper allows to apply a layer to every temporal slice of an input.
    ])

    """
    TimeDistributed:

      Consider a batch of 32 video samples, where each sample is a 128x128 RGB image with channels_last data format, across 10 timesteps. The batch input shape is (32, 10, 128, 128, 3).

      You can then use TimeDistributed to apply the same Conv2D layer to each of the 10 timesteps, independently

      Because TimeDistributed applies the same instance of Conv2D to each of the timestamps, the same set of weights are used at each timestamp.
    """

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  # even though input is one hot encoded, target is still tokenized, so we must use sparse categorical cross entropy
                  optimizer=tf.keras.optimizers.Adam())

    return model

