import tensorflow as tf
import tensorflow_datasets as tfds
import os

def spilt_task_name(task_name):
    task_name_split = task_name.split('_to_')
    return task_name_split[0], task_name_split[1]

def get_ted_hrlr_translate_dataset(task_name='az_to_en', BATCH_SIZE=64, MAX_LENGTH=40, BUFFER_SIZE=20000,
                                   languageA_target_vocab_size=2 ** 13, languageB_target_vocab_size=2 ** 13):
    """
    tensorflow dataset url: https://www.tensorflow.org/datasets/datasets#ted_hrlr_translate
    Data sets derived from TED talk transcripts for comparing similar language pairs where one is high resource and the other is low resource.
    raw data URL: https://github.com/neulab/word-embeddings-for-nmt
    :param task_name: languageA_to_languageB, you can choose form task_name_list
    :param MAX_LENGTH:
    :param BATCH_SIZE:
    :param BUFFER_SIZE:
    :return:
    """
    task_name_prefix = 'ted_hrlr_translate'
    task_name_list = ['az_to_en', 'az_tr_to_en', 'be_to_en',
                      'be_ru_to_en', 'es_to_pt', 'fr_to_pt',
                      'gl_to_en', 'gl_pt_to_en', 'he_to_pt',
                      'it_to_pt', 'pt_to_en', 'ru_to_en',
                      'ru_to_pt', 'tr_to_en']
    if task_name not in task_name_list:
        raise ValueError(f'choose task_name form {task_name_list}')


    complete_task_name = task_name_prefix + '/' + task_name
    examples, metadata = tfds.load(complete_task_name, with_info=True, as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    if not os.path.exists(complete_task_name):
        os.makedirs(complete_task_name)

    tokenizer_languageA_path = os.path.join(complete_task_name, spilt_task_name(task_name)[0])
    tokenizer_languageA_complete_path = tokenizer_languageA_path + ".subwords"
    if not os.path.exists(tokenizer_languageA_complete_path):
        tokenizer_languageA = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (languageA.numpy() for languageA, languageB in train_examples),
            target_vocab_size=languageA_target_vocab_size)
        tokenizer_languageA.save_to_file(os.path.join(complete_task_name, spilt_task_name(task_name)[0]))
    else:
        tokenizer_languageA = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_languageA_path)

    tokenizer_languageB_path = os.path.join(complete_task_name, spilt_task_name(task_name)[1])
    tokenizer_languageB_complete_path = tokenizer_languageB_path + ".subwords"
    if not os.path.exists(tokenizer_languageB_complete_path):
        tokenizer_languageB = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (languageB.numpy() for languageA, languageB in train_examples),
            target_vocab_size=languageB_target_vocab_size)
        tokenizer_languageB.save_to_file(os.path.join(complete_task_name, spilt_task_name(task_name)[1]))
    else:
        tokenizer_languageB = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_languageB_path)

    def encode(lang1, lang2):  # tokenizer_pt.vocab_size -> <start>, tokenizer_pt.vocab_size + 1 -> <end>
        lang1 = [tokenizer_languageA.vocab_size] + tokenizer_languageA.encode(
            lang1.numpy()) + [tokenizer_languageA.vocab_size + 1]
        lang2 = [tokenizer_languageB.vocab_size] + tokenizer_languageB.encode(
            lang2.numpy()) + [tokenizer_languageB.vocab_size + 1]
        return lang1, lang2

    def filter_max_length(x, y, max_length=MAX_LENGTH):
        return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

    def tf_encode(pt, en):
        return tf.py_function(encode, [pt, en], [tf.int32, tf.int32])

    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = val_examples.map(tf_encode)
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
    return train_dataset, val_dataset


if __name__ == "__main__":
    # hyper parameter
    task_name = 'pt_to_en'
    BUFFER_SIZE = 20000
    BATCH_SIZE = 64
    MAX_LENGTH = 40

    # prepare dataset
    train_dataset, val_dataset = get_ted_hrlr_translate_dataset(task_name, BATCH_SIZE, MAX_LENGTH, BUFFER_SIZE,
                                                                languageA_target_vocab_size=2 ** 13,
                                                                languageB_target_vocab_size=2 ** 13)

    for batch_sample in train_dataset.take(3):
        print(batch_sample)
