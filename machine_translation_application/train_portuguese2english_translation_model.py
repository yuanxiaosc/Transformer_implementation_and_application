from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../transformer_implement")))
from transformer import Transformer, create_masks, CustomSchedule, loss_function
import time


def portuguese2english_translation_data_and_tokenizer():
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)
    tokenizer_en.save_to_file("tokenizer_en")

    tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (pt.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)
    tokenizer_pt.save_to_file("tokenizer_pt")

    return train_examples, val_examples, tokenizer_en, tokenizer_pt


def get_dataset(train_examples, val_examples, BATCH_SIZE, MAX_LENGTH, BUFFER_SIZE):
    tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file("tokenizer_en")
    tokenizer_pt = tfds.features.text.SubwordTextEncoder.load_from_file("tokenizer_pt")

    def encode(lang1, lang2): #tokenizer_pt.vocab_size -> <start>, tokenizer_pt.vocab_size + 1 -> <end>
        lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
            lang1.numpy()) + [tokenizer_pt.vocab_size + 1]
        lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
            lang2.numpy()) + [tokenizer_en.vocab_size + 1]
        return lang1, lang2

    def filter_max_length(x, y, max_length=MAX_LENGTH):
        return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

    def tf_encode(pt, en):
        return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])

    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = val_examples.map(tf_encode)
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
    return train_dataset, val_dataset



def main(BUFFER_SIZE, BATCH_SIZE, MAX_LENGTH, num_layers, d_model, dff, num_heads, dropout_rate, EPOCHS, checkpoint_path):
    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    # prepare data and produce tokenizer
    train_examples, val_examples, tokenizer_en, tokenizer_pt = portuguese2english_translation_data_and_tokenizer()
    # prepare dataset
    train_dataset, val_dataset = get_dataset(train_examples, val_examples,
                                             BATCH_SIZE, MAX_LENGTH, BUFFER_SIZE)
    print("check a batch data:")
    pt_batch, en_batch = next(iter(val_dataset))
    print("pt_batch:\n", pt_batch)
    print("en_batch:\n", en_batch)

    input_vocab_size = tokenizer_pt.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2

    # create model
    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size, dropout_rate)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # checkpoint
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    # train
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            if batch % 500 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


if __name__ == "__main__":
    # hyper parameter
    BUFFER_SIZE = 20000
    BATCH_SIZE = 64
    MAX_LENGTH = 40
    # model structure
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    # train parameter
    EPOCHS = 20
    checkpoint_path = "./checkpoints/train"

    main(BUFFER_SIZE, BATCH_SIZE, MAX_LENGTH,
         num_layers, d_model, dff, num_heads, dropout_rate,
         EPOCHS, checkpoint_path)

