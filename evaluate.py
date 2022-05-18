import tensorflow as tf

def preprocess(texts, tokenizer, max_id):
  # Preprocessing the text by first tokenizing and then one hot encoding the input
  x = np.array(tokenizer.texts_to_sequences(texts)) - 1
  return tf.one_hot(x, max_id)


def next_char(text, model, tokenizer, max_id, temperature=1):
  # First preprocess the text input
  X_new = preprocess([text], tokenizer, max_id)

  y_probs = model.predict(X_new)[0, -1:, :]

  rescaled_logits = tf.math.log(y_probs) / temperature

  char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1

  return tokenizer.sequences_to_texts(char_id.numpy())[0]

def complete_text(text, model, tokenizer, max_id, n_chars=1000, temperature=1):
  for _ in range(n_chars):
    text += next_char(text, model, tokenizer, max_id, temperature)
  return text