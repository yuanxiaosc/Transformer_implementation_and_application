import tensorflow_datasets as tfds
import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../transformer_implement")))
from transformer import Transformer, create_masks, CustomSchedule

import matplotlib.pyplot as plt


class PredictManager(object):
    def __init__(self, tokenizer_en_file="tokenizer_en", tokenizer_pt_file="tokenizer_pt",
                 checkpoint_path="./checkpoints/train", MAX_LENGTH=40,
                 num_layers=4, d_model=128, num_heads=8, dff=512, dropout_rate=0.1):
        self.MAX_LENGTH = MAX_LENGTH
        self.resotre_tokenizer(tokenizer_en_file, tokenizer_pt_file)
        self.restore_model(checkpoint_path, num_layers, d_model, num_heads, dff, dropout_rate)

    def resotre_tokenizer(self, tokenizer_en_file, tokenizer_pt_file):
        self.tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_en_file)
        self.tokenizer_pt = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_pt_file)
        self.input_vocab_size = self.tokenizer_pt.vocab_size + 2
        self.target_vocab_size = self.tokenizer_en.vocab_size + 2

    def restore_model(self, checkpoint_path, num_layers, d_model, num_heads, dff, dropout_rate):
        # create model tructure
        self.translate_transformer = Transformer(num_layers, d_model, num_heads, dff,
                                                 self.input_vocab_size, self.target_vocab_size, dropout_rate)
        # restore model weight
        learning_rate = CustomSchedule(d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        ckpt = tf.train.Checkpoint(transformer=self.translate_transformer, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!')

    def evaluate(self, inp_sentence):
        start_token = [self.tokenizer_pt.vocab_size]
        end_token = [self.tokenizer_pt.vocab_size + 1]

        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = start_token + self.tokenizer_pt.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [self.tokenizer_en.vocab_size]
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
            if tf.equal(predicted_id, self.tokenizer_en.vocab_size + 1):
                return tf.squeeze(output, axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def translate(self, sentence, plot=''):
        result, attention_weights = self.evaluate(sentence)

        predicted_sentence = self.tokenizer_en.decode([i for i in result
                                                  if i < self.tokenizer_en.vocab_size])

        print('Input: {}'.format(sentence))
        print('Predicted translation: {}'.format(predicted_sentence))

        if plot:
            self.plot_attention_weights(attention_weights, sentence, result, plot)

    def plot_attention_weights(self, attention, sentence, result, layer):
        fig = plt.figure(figsize=(16, 8))

        sentence = self.tokenizer_pt.encode(sentence)

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
                ['<start>'] + [self.tokenizer_pt.decode([i]) for i in sentence] + ['<end>'],
                fontdict=fontdict, rotation=90)

            ax.set_yticklabels([self.tokenizer_en.decode([i]) for i in result
                                if i < self.tokenizer_en.vocab_size],
                               fontdict=fontdict)

            ax.set_xlabel('Head {}'.format(head + 1))

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":


    translate_manager = PredictManager(tokenizer_en_file="tokenizer_en", tokenizer_pt_file="tokenizer_pt",
                 checkpoint_path="./checkpoints/train", MAX_LENGTH=40,
                 num_layers=4, d_model=128, num_heads=8, dff=512, dropout_rate=0.1)

    # translate_manager.translate("este é o primeiro livro que eu fiz.", plot='decoder_layer4_block2')
    translate_manager.translate("este é o primeiro livro que eu fiz.")
    print("Real translation: this is the first book i've ever done.\n")
