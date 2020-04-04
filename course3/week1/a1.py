import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

sentences = [
			'I love my dog',
			'I love my cat',
			'You love my dog!',
			'Do you think my dog is amazing?'
			]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>") # num_words: vocab size (dictionary size) , oov: out of vocabulary 
tokenizer.fit_on_texts(sentences) # sentences to form the dictionary 
word_index = tokenizer.word_index # the dictionary (vocab that is created)


sequences = tokenizer.texts_to_sequences(sentences) # the sentences in tokenized form


padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=5) # makes all the sentences of equal length

print("--"*40)
print("\nword_index: ")
print(word_index)
print("\nsequences: ")
print(sequences)
print("\npadded: ")
print(padded)



# test_data = [
# 			'i really love my dog',
# 			'my dog loves my manatee'
# 			]

# test_seq = tokenizer.texts_to_sequences(test_data)

# print(test_seq)