import tensorflow as tf
import tensorflow_datasets as tfds

from transformer_B_model import transformer
from train_chatbot import preprocess_sentence

def evaluate(sentence, model):
  sentence = preprocess_sentence(sentence)

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # concatenated the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)
  return tf.squeeze(output, axis=0)


def predict(sentence, model):
  prediction = evaluate(sentence, model)
  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])
  print('Input: {}'.format(sentence))
  print('Output: {}'.format(predicted_sentence))
  return predicted_sentence

vocab_filename = "vocab_file"
check_dir = "save_model"
# Hyper-parameters
MAX_LENGTH = 40
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_filename)
VOCAB_SIZE = tokenizer.vocab_size
START_TOKEN, END_TOKEN = [VOCAB_SIZE], [VOCAB_SIZE + 1]
VOCAB_SIZE = VOCAB_SIZE + 2

latest_checkpoint = tf.train.latest_checkpoint(check_dir)
chatbot_model = transformer(vocab_size=VOCAB_SIZE,num_layers=NUM_LAYERS,units=UNITS,d_model=D_MODEL,num_heads=NUM_HEADS,dropout=DROPOUT)
chatbot_model.load_weights(latest_checkpoint)



# feed the model with its previous output
sentence = 'I am not crazy, my mother had me tested.'
for _ in range(5):
  sentence = predict(sentence, model=chatbot_model)
  print('')

while True:
    sentence = input("Your Input:")
    if sentence == "exit()":
        break
    else:
        sentence = predict(sentence, model=chatbot_model)
        print('')

"""
Input: I am not crazy, my mother had me tested.
Output: you re not going to be around to be a doctor .

Input: you re not going to be around to be a doctor .
Output: i m not . i m a little tired . and i m going to go to bed . i m going to be here all night .

Input: i m not . i m a little tired . and i m going to go to bed . i m going to be here all night .
Output: you re not going to be any better .

Input: you re not going to be any better .
Output: i m not . i m just a little tired , and i m going to be a little bit of a good money . i ll go to the hotel .

Input: i m not . i m just a little tired , and i m going to be a little bit of a good money . i ll go to the hotel .
Output: i ll be back in a minute .
"""