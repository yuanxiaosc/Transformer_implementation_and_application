# coding=utf-8
#  the implementation description see https://medium.com/tensorflow/a-transformer-chatbot-tutorial-with-tensorflow-2-0-88bf59e66fe2,
#  the paper see https://arxiv.org/abs/1706.03762
"""300 lines of code complete implementation of Transformer"""
import tensorflow as tf


def scaled_dot_product_attention(query, key, value, mask):
    """
    Calculate the attention weights.
    :param query: [batch_size, num_heads, sequence_length, depth]
    :param key: [batch_size, num_heads, sequence_length, depth]
    :param value: [batch_size, num_heads, sequence_length, depth]
    :param mask: [batch_size, 1, 1, sequence_length]
    :return: output [batch_size, num_heads, sequence_length, depth]
    """
    # [batch_size, num_heads, sequence_length, sequence_length]
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    # [batch_size, num_heads, sequence_length, depth]
    output = tf.matmul(attention_weights, value)
    return output


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        """
        :param inputs: [batch_size, sequence_length, d_model]
        :param batch_size:
        :return: [batch_size, num_heads, sequence_length, depth]
        """
        # [batch_size, sequence_length, num_heads, depth]
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        # [batch_size, num_heads, sequence_length, depth]
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], \
                                  inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers [batch_size, sequence_length, d_model]
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads [batch_size, num_heads, sequence_length, depth]
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention [batch_size, num_heads, sequence_length, depth]
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        # [batch_size, sequence_length, num_heads, depth]
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # concatenation of heads [batch_size, sequence_length, d_model]
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # final linear layer [batch_size, sequence_length, d_model]
        outputs = self.dense(concat_attention)
        return outputs


def create_padding_mask(x):
    """
    :param x: [batch_size, sequence_length]
    :return: padding_mask [batch_size, 1, 1, sequence_length]
    """
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    """
    :param x: [batch_size, sequence_length]
    :return: look_ahead_mask [batch_size, 1, sequence_length, sequence_length]
    """
    seq_len = tf.shape(x)[1]
    # [sequence_length, sequence_length]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    # [batch_size, 1, 1, sequence_length]
    padding_mask = create_padding_mask(x)
    # [batch_size, 1, sequence_length, sequence_length]
    return tf.maximum(look_ahead_mask, padding_mask)


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        """
        :param position: [position_length, 1]
        :param i: [1, d_model]
        :param d_model:
        :return:  [position_length, d_model]
        """
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        # [position_length, d_model]
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)

        # apply sin to even index in the array [position_length, d_model / 2]
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array [position_length, d_model / 2]
        cosines = tf.math.cos(angle_rads[:, 1::2])

        # [position_length, d_model]
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        # [1, position_length, d_model]
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        """
        :param inputs: [batch_size, sequence_length, d_model]
        :return: [batch_size, sequence_length, d_model]
        """
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    # [batch_size, sequence_length, d_model]
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    # [batch_size, 1, 1, sequence_length]
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # [batch_size, sequence_length, d_model]
    attention = MultiHeadAttention(d_model, num_heads, name="attention") \
        ({'query': inputs, 'key': inputs, 'value': inputs, 'mask': padding_mask})
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    # [batch_size, sequence_length, units]
    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    # [batch_size, sequence_length, d_model]
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    # [batch_size, sequence_length, d_model]
    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name="encoder"):
    # [batch_size, sequence_length]
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    # [batch_size, 1, 1, sequence_length]
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
    # [batch_size, sequence_length, d_model]
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    # [batch_size, sequence_length, d_model]
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    # [batch_size, sequence_length, d_model]
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    # [batch_size, sequence_length, d_model]
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        # [batch_size, sequence_length, d_model]
        outputs = encoder_layer(
            units=units, d_model=d_model, num_heads=num_heads, dropout=dropout,
            name="encoder_layer_{}".format(i))([outputs, padding_mask])
    # [batch_size, sequence_length, d_model]
    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    # [batch_size, sequence_length, d_model]
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    # [batch_size, sequence_length, d_model]
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    # [batch_size, 1, sequence_length, sequence_length]
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    # [batch_size, 1, 1, sequence_length]
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # [batch_size, sequence_length, d_model]
    attention1 = MultiHeadAttention(d_model, num_heads, name="attention_1") \
        (inputs={'query': inputs, 'key': inputs, 'value': inputs, 'mask': look_ahead_mask})
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    # [batch_size, sequence_length, d_model]
    attention2 = MultiHeadAttention(d_model, num_heads, name="attention_2") \
        (inputs={'query': attention1, 'key': enc_outputs, 'value': enc_outputs, 'mask': padding_mask})
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    # [batch_size, sequence_length, units]
    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    # [batch_size, sequence_length, d_model]
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    # [batch_size, sequence_length, d_model]
    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs, name=name)


def decoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name='decoder'):
    # [batch_size, sequence_length]
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    # [batch_size, sequence_length, d_model]
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    # [batch_size, 1, sequence_length, sequence_length]
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    # [batch_size, 1, 1, sequence_length]
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # [batch_size, sequence_length, d_model]
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    # [batch_size, sequence_length, d_model]
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    # [batch_size, sequence_length, d_model]
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    # [batch_size, sequence_length, d_model]
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        # [batch_size, sequence_length, d_model]
        outputs = decoder_layer(units=units, d_model=d_model, num_heads=num_heads,
                                dropout=dropout, name='decoder_layer_{}'.format(i), ) \
            (inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    # [batch_size, sequence_length, d_model]
    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs, name=name)


def transformer(vocab_size, num_layers, units, d_model, num_heads, dropout, name="transformer"):
    # [batch_size, sequence_length]
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    # [batch_size, sequence_length]
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    # [batch_size, 1, 1, sequence_length]
    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)

    # mask the future tokens for decoder inputs at the 1st attention block
    # [batch_size, 1, sequence_length, sequence_length]
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask,
        output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)

    # mask the encoder outputs for the 2nd attention block
    # [batch_size, 1, 1, sequence_length]
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)

    # [batch_size, sequence_length, d_model]
    enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, units=units,
                          d_model=d_model, num_heads=num_heads, dropout=dropout, ) \
        (inputs=[inputs, enc_padding_mask])

    # [batch_size, sequence_length, d_model]
    dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, units=units,
                          d_model=d_model, num_heads=num_heads, dropout=dropout, ) \
        (inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    # [batch_size, sequence_length, vocab_size]
    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)
    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


if __name__ == "__main__":
    VOCAB_SIZE = 8192
    # Hyper-parameters
    NUM_LAYERS = 2
    D_MODEL = 256
    NUM_HEADS = 8
    UNITS = 512
    DROPOUT = 0.1

    transformer_model = transformer(vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, units=UNITS,
                                    d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT)
    transformer_model.summary()
    # plot_model
    tf.keras.utils.plot_model(transformer_model, to_file='transformer.png', show_shapes=True)

    temp_input = tf.random.uniform((64, 62))
    temp_target = tf.random.uniform((64, 26))
    print(f"temp_input shape:\t{temp_input.shape}")
    print(f"temp_target shape:\t{temp_target.shape}")
    outputs = transformer_model([temp_input, temp_target])
    print("Pass by---------------------transformer_model-------------------")
    print(f"outputs.shape:\t {outputs.shape}")
