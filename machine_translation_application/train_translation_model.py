from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
import os
import time

from transformer_A_model import Transformer, create_masks, CustomSchedule, loss_function
from dataset_ultis import get_ted_hrlr_translate_dataset, spilt_task_name


def get_vocab_size(task_name_prefix, task_name):
    complete_task_name = task_name_prefix + '/' + task_name
    tokenizer_languageA_path = os.path.join(complete_task_name, spilt_task_name(task_name)[0])
    tokenizer_languageB_path = os.path.join(complete_task_name, spilt_task_name(task_name)[1])
    tokenizer_languageA = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_languageA_path)
    tokenizer_languageB = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_languageB_path)
    input_vocab_size = tokenizer_languageA.vocab_size + 2
    target_vocab_size = tokenizer_languageB.vocab_size + 2
    return input_vocab_size, target_vocab_size


def main(task_name_prefix='ted_hrlr_translate', task_name='pt_to_en',
         BUFFER_SIZE=20000, BATCH_SIZE=64, MAX_LENGTH=40,
         num_layers=4, d_model=128, dff=512, num_heads=8, dropout_rate=0.1, EPOCHS=20):
    checkpoint_path = f"./checkpoints/train/{task_name_prefix}/{task_name}"

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

    # prepare dataset
    train_dataset, val_dataset = get_ted_hrlr_translate_dataset(task_name, BATCH_SIZE, MAX_LENGTH, BUFFER_SIZE,
                                                                languageA_target_vocab_size=2 ** 13,
                                                                languageB_target_vocab_size=2 ** 13)
    print("check a batch data:")
    pt_batch, en_batch = next(iter(val_dataset))
    print("pt_batch:\n", pt_batch)
    print("en_batch:\n", en_batch)

    input_vocab_size, target_vocab_size = get_vocab_size(task_name_prefix, task_name)

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
    start_epoch = 0
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(f'start_epoch is {start_epoch}. Latest checkpoint restored!!')

    # train
    for epoch in range(start_epoch, EPOCHS):
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
    task_name_prefix = 'ted_hrlr_translate'
    task_name = 'pt_to_en'
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
    EPOCHS = 200

    main(task_name_prefix, task_name, BUFFER_SIZE, BATCH_SIZE, MAX_LENGTH,
         num_layers, d_model, dff, num_heads, dropout_rate, EPOCHS)
