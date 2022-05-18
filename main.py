import tensorflow as tf
import numpy as np
from models import create_model
from evaluate import complete_text
import pandas as pd
import matplotlib.pyplot as plt

url = "https://homl.info/shakespeare"
# File at the origin url is downloaded to the cache dir, final location of the file is placed on the fname in our case it is "shakespeare.txt"
filepath = tf.keras.utils.get_file("shakespeare.txt", url)
# Open the file with "with" command so that we do not need to close it afterwards
with open(filepath) as f:
  text = f.read()


# Create the character level tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True) #  char_level: if True, every character will be treated as a token.

# Fit it on the text
# fit_on_texts: This method creates the vocabulary index based on word frequency. 0 is reserved for padding. So lower integer means more frequent word.
tokenizer.fit_on_texts(text)

# Number of distinct characters
max_id = len(tokenizer.word_index)

# Total number of characters
dataset_size = tokenizer.document_count

# We subtract 1 to get IDs from 0 to 38, rather than from 1 to 39
[encoded] = np.array(tokenizer.texts_to_sequences([text])) - 1

# Train and validation data split
split_size = int(dataset_size * 0.7)
dataset = tf.data.Dataset.from_tensor_slices(encoded[:split_size])

n_steps = 100
window_length = n_steps + 1 # target = input shifted 1 character ahead


"""
Input:
[[1,2,3,4,5,6,7,8]]
Output:
[[1,2,3,4,5],
[2,3,4,5,6],
[3,4,5,6,7],
[4,5,6,7,8]]
"""
dataset = dataset.window(window_length, shift=1, drop_remainder=True)


"""
    map: It returns a new RDD by applying given function to each element of the RDD. Function in map returns only one item.

    flatMap: Similar to map, it returns a new RDD by applying a function to each element of the RDD, but output is flattened.
"""
dataset = dataset.flat_map(lambda window: window.batch(window_length))

batch_size = 8
# We seperate the data into 8 batches and then shuffle it, in the end drop the remained data
dataset = dataset.shuffle(10000).batch(batch_size,drop_remainder=True)

# At this one we seperate target and input from the dataset
"""
Input:
[[1,2,3,4,5],
[2,3,4,5,6],
[3,4,5,6,7],
[4,5,6,7,8]]
Output:

Input: [[1,2,3,4], Target: [2,3,4,5]]
       [[2,3,4,5], [3,4,5,6]]
"""
dataset = dataset.map(lambda windows: (windows[:,:-1], windows[:,1:]))

# Then we do a one hot encoding on the input data so that loss function would make sense
dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))

dataset = dataset.prefetch(1)

# Create the model
model = create_model(max_id=max_id)

# Fit the model
history = model.fit(dataset,steps_per_epoch=500, epochs=10)

pd.DataFrame(history.history).plot(figsize=(10,7))
plt.show()

print(complete_text("r",model=model, tokenizer=tokenizer, max_id=max_id, temperature=1))


