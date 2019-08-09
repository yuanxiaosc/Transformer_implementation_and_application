import tensorflow as tf
import tensorflow_datasets as tfds
from transformer_A_model import Transformer, create_masks, CustomSchedule
import matplotlib.pyplot as plt
import os

from dataset_ultis import get_ted_hrlr_translate_dataset, spilt_task_name


class PredictManager(object):
    def __init__(self, task_name_prefix='ted_hrlr_translate', task_name='pt_to_en',
                 MAX_LENGTH=40,
                 num_layers=4, d_model=128, num_heads=8, dff=512, dropout_rate=0.1):
        self.MAX_LENGTH = MAX_LENGTH
        self.checkpoint_path = f"./checkpoints/train/{task_name_prefix}/{task_name}"
        self.resotre_tokenizer(task_name_prefix, task_name)
        self.restore_model(num_layers, d_model, num_heads, dff, dropout_rate)

    def resotre_tokenizer(self, task_name_prefix, task_name):
        complete_task_name = task_name_prefix + '/' + task_name
        tokenizer_languageA_path = os.path.join(complete_task_name, spilt_task_name(task_name)[0])
        tokenizer_languageB_path = os.path.join(complete_task_name, spilt_task_name(task_name)[1])
        self.tokenizer_languageA = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_languageA_path)
        self.tokenizer_languageB = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_languageB_path)
        self.input_vocab_size = self.tokenizer_languageA.vocab_size + 2
        self.target_vocab_size = self.tokenizer_languageB.vocab_size + 2
        print("self.input_vocab_size", self.input_vocab_size)
        print("self.target_vocab_size", self.target_vocab_size)

    def restore_model(self, num_layers, d_model, num_heads, dff, dropout_rate):
        # create model tructure
        self.translate_transformer = Transformer(num_layers, d_model, num_heads, dff,
                                                 self.input_vocab_size, self.target_vocab_size, dropout_rate)
        # restore model weight
        ckpt = tf.train.Checkpoint(transformer=self.translate_transformer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!')
        else:
            raise ValueError('Not found checkpoint file!')

    def evaluate(self, inp_sentence):
        start_token = [self.tokenizer_languageA.vocab_size]
        end_token = [self.tokenizer_languageA.vocab_size + 1]

        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = start_token + self.tokenizer_languageA.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [self.tokenizer_languageB.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(self.MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.translate_transformer(encoder_input,
                                                                        output,
                                                                        False,
                                                                        enc_padding_mask,
                                                                        combined_mask,
                                                                        dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, self.tokenizer_languageB.vocab_size + 1):
                return tf.squeeze(output, axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def translate(self, sentence, plot=''):
        result, attention_weights = self.evaluate(sentence)

        predicted_sentence = self.tokenizer_languageB.decode([i for i in result
                                                              if i < self.tokenizer_languageB.vocab_size])

        print('Input: {}'.format(sentence))
        print('Predicted translation: {}'.format(predicted_sentence))

        if plot:
            self.plot_attention_weights(attention_weights, sentence, result, plot)

    def plot_attention_weights(self, attention, sentence, result, layer):
        fig = plt.figure(figsize=(16, 8))

        sentence = self.tokenizer_languageA.encode(sentence)

        attention = tf.squeeze(attention[layer], axis=0)

        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head + 1)

            # plot the attention weights
            ax.matshow(attention[head][:-1, :], cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(sentence) + 2))
            ax.set_yticks(range(len(result)))

            ax.set_ylim(len(result) - 1.5, -0.5)

            ax.set_xticklabels(
                ['<start>'] + [self.tokenizer_languageA.decode([i]) for i in sentence] + ['<end>'],
                fontdict=fontdict, rotation=90)

            ax.set_yticklabels([self.tokenizer_languageB.decode([i]) for i in result
                                if i < self.tokenizer_languageB.vocab_size],
                               fontdict=fontdict)

            ax.set_xlabel('Head {}'.format(head + 1))

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    task_name_prefix = 'ted_hrlr_translate'
    task_name = 'pt_to_en'

    MAX_LENGTH = 40
    # model structure
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    translate_manager = PredictManager(
        task_name_prefix, task_name, MAX_LENGTH,
        num_layers, d_model, num_heads, dff, dropout_rate)

    # translate_manager.translate("este é o primeiro livro que eu fiz.", plot='decoder_layer4_block2')
    translate_manager.translate("este é o primeiro livro que eu fiz.")
    print("Real translation: this is the first book i've ever done.\n")
