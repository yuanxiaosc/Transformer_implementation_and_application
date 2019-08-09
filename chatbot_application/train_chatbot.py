import tensorflow as tf
import tensorflow_datasets as tfds
import os
import re
import time

from transformer_B_model import transformer

def cornell_movie_dialogs_file_path():
    path_to_zip = tf.keras.utils.get_file(
        'cornell_movie_dialogs.zip',
        origin='http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
        extract=True)

    path_to_dataset = os.path.join(
        os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")

    path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
    path_to_movie_conversations = os.path.join(path_to_dataset, 'movie_conversations.txt')
    print(f"cornell_movie_dialogs_data_file save to '{path_to_dataset}'")
    return path_to_movie_lines, path_to_movie_conversations


def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    # adding a start and an end token to the sentence
    return sentence


def load_conversations(MAX_SAMPLES):
    path_to_movie_lines, path_to_movie_conversations = cornell_movie_dialogs_file_path()
    # dictionary of line id to text
    id2line = {}
    with open(path_to_movie_lines, errors='ignore') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        id2line[parts[0]] = parts[4]

    inputs, outputs = [], []
    with open(path_to_movie_conversations, 'r') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        # get conversation in a list of line ID
        conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
        for i in range(len(conversation) - 1):
            inputs.append(preprocess_sentence(id2line[conversation[i]]))
            outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))
            if len(inputs) >= MAX_SAMPLES:
                return inputs, outputs
    return inputs, outputs


# Tokenize, filter and pad sentences
def tokenize_and_filter(tokenizer, inputs, outputs, START_TOKEN, END_TOKEN, MAX_LENGTH):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        # check tokenized sentence max length
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs


def get_dataset(MAX_SAMPLES, MAX_LENGTH, BATCH_SIZE, BUFFER_SIZE, vocab_filename="vocab_file"):
    questions, answers = load_conversations(MAX_SAMPLES)

    # Build tokenizer using tfds for both questions and answers
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        questions + answers, target_vocab_size=2**13)
    tokenizer.save_to_file(vocab_filename)
    # Define start and end token to indicate the start and end of a sentence
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    # Vocabulary size plus start and end token
    VOCAB_SIZE = tokenizer.vocab_size + 2


    questions, answers = tokenize_and_filter(tokenizer, questions, answers, START_TOKEN, END_TOKEN, MAX_LENGTH)
    print('Vocab size: {}'.format(VOCAB_SIZE))
    print('Number of samples: {}'.format(len(questions)))

    # decoder inputs use the previous target as input
    # remove START_TOKEN from targets
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions,
            'dec_inputs': answers[:, :-1]
        },
        {
            'outputs': answers[:, 1:]
        },
    ))

    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    print("dataset info:\t", dataset)
    return dataset, tokenizer, VOCAB_SIZE, START_TOKEN, END_TOKEN


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def accuracy(y_true, y_pred):
  # ensure labels have shape (batch_size, MAX_LENGTH - 1)
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  accuracy = tf.metrics.SparseCategoricalAccuracy()(y_true, y_pred)
  return accuracy

def get_saved_model_path(save_model_dir="save_model", model_name="chatbot"):
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    model_name += "_" + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    saved_model_path = os.path.join(save_model_dir, model_name)
    return saved_model_path

if __name__=="__main__":
    # Hyper-parameters
    MAX_LENGTH = 50
    # Dataset hyper-parameters
    VOCAB_SIZE = None
    MAX_SAMPLES = 500000
    BATCH_SIZE = 64
    BUFFER_SIZE = 20000
    # Model hyper-parameters
    EPOCHS = 100
    NUM_LAYERS = 2
    D_MODEL = 256
    NUM_HEADS = 8
    UNITS = 512
    DROPOUT = 0.1

    dataset, tokenizer, VOCAB_SIZE, START_TOKEN, END_TOKEN = get_dataset(MAX_SAMPLES, MAX_LENGTH, BATCH_SIZE, BUFFER_SIZE)

    tf.keras.backend.clear_session()

    chatbot_model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)

    learning_rate = CustomSchedule(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    chatbot_model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
    chatbot_model.summary()

    # Creating Keras callbacks
    os.makedirs('train_tensorboard', exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="train_tensorboard", histogram_freq=1)
    early_stopping_checkpoint = tf.keras.callbacks.EarlyStopping(patience=10)

    check_path = get_saved_model_path() + ".ckpt"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(check_path,
                                                     save_weights_only=True,
                                                     save_freq=5)
    check_dir = "save_model"
    latest_checkpoint = tf.train.latest_checkpoint(check_dir)

    chatbot_model.load_weights(latest_checkpoint)
    chatbot_model.fit(dataset, epochs=EPOCHS,
                      callbacks=[tensorboard_callback,
                         model_checkpoint_callback,
                         early_stopping_checkpoint])