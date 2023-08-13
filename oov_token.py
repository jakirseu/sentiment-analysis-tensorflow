import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'Stay hungry, stay foolish.',
    'The future belongs to those who prepare for it today.',
    'Life is like a box of chocolates.'
]

tokenizer = Tokenizer(num_words=100, oov_token="&lt;OOV")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

print(word_index)
print(sequences)

test_data = [
    'Stay hungry, stay foolish.',
    'A great man is always willing to be little.'
]
test_seq = tokenizer.texts_to_sequences(test_data)

print("\nTest Sequence:", test_seq)