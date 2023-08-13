import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# import data
df = pd.read_csv('Train.csv')
df.head()

x = df['text']
y = df['label']

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=50)

# preprocess
tokenizer = Tokenizer(num_words=100, oov_token="&lt;OOV&gt;")
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
# print word index aka dictionary
print(word_index)

vocab_size = 5000
embedding_dim = 32
max_length = 75
trunc_type = 'post'
pad_type = 'post'
oov_tok = "&lt;OOV&gt;"

train_seq = tokenizer.texts_to_sequences(X_train)
train_pad_seq = pad_sequences(train_seq, maxlen=max_length, truncating=trunc_type, padding=pad_type)

valid_seq = tokenizer.texts_to_sequences(X_test)
valid_pad_seq = pad_sequences(valid_seq, maxlen=max_length)

training_labels_final = np.array(Y_train)
validation_labels_final = np.array(Y_test)

# creating model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# train the model

num_epochs = 10
history = model.fit(train_pad_seq, training_labels_final, epochs=num_epochs,
                    validation_data=(valid_pad_seq, validation_labels_final))

test_reviews = ["This movie is not good at all. I did not enjoyed much",
                "One of the best movie I have ever seen. Recommend everyone to watch this movie"]

# test the model
# create the sequences from test data

test_sequences = tokenizer.texts_to_sequences(test_reviews)
test_padded = pad_sequences(test_sequences, padding=pad_type, maxlen=max_length)

classes = model.predict(test_padded)

# Closer to 1 means positive
for x in range(len(test_reviews)):
    print(test_reviews[x])
    print(classes[x])
    print('\n')