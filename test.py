# import random
# import json
# import pickle
# import numpy as np
# import nltk
# from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.models import load_model

# # Load the original words and classes
# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# ignore_letters=['?','!','.',',']
# lemmatizer = WordNetLemmatizer()

# with open('test_intent.json') as json_file:
#     test_intents = json.load(json_file)

# testing = []
# output_empty = [0] * len(classes)
# # print(classes)
# # Process test intents to create the test data
# for intent in test_intents['intents']:
#     for pattern in intent['patterns']:
#         bag = []
#         word_list = nltk.word_tokenize(pattern)
#         word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list]
#         for word in words:
#             bag.append(1) if word in word_list else bag.append(0)

#         output_row = list(output_empty)
#         output_row[classes.index(intent['tag'])] = 1

#         testing.append([bag, output_row])

# # Convert test data into a numpy array for consistency
# testing = np.array(testing, dtype=object)

# # Split the data into X (our patterns) and Y (our intents)
# test_x = np.array(list(testing[:, 0]))
# test_y = np.array(list(testing[:, 1]))

# # Load the trained model
# model = load_model('chatbotmodel.h5')

# # Evaluate the model with the test data
# eval_result = model.evaluate(test_x, test_y)
# print("Test Loss:", eval_result[0])
# print("Test Accuracy:", eval_result[1])
# # Test individual predictions
# for i in range(len(test_x)):
#     pred = model.predict(np.array([test_x[i]]))
#     pred_class = np.argmax(pred)
#     true_class = np.argmax(test_y[i])
#     print(f"Test example {i}: True class: {true_class}, Predicted class: {pred_class}")


import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('chatbotmodel.h5')

# Load words, classes, and lemmatizer
lemmatizer = WordNetLemmatizer()
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# # Display classes
# print("Classes loaded:", classes)

ignore_letters = ['?', '!', '.', ',']

def clean_up_sentence(sentence):
    # Tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize each word and remove ignored letters
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words if word not in ignore_letters]
    return sentence_words

def bag_of_words(sentence, words):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # Create the bag of words array
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

# Load the test dataset
with open('test_intent.json') as file:
    test_intents = json.load(file)

correct = 0
wrong = 0

# Iterate through each sentence in our intents patterns
for intent in test_intents['intents']:
    for pattern in intent['patterns']:
        # Generate a bag of words for the pattern
        bow = bag_of_words(pattern, words)
        # Use the model to predict the tag
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        # Sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        if results:
            predicted_tag = classes[results[0][0]]
            true_tag = intent['tag']
            # Adding print to show word, expected tag, and predicted tag
            print(f"Word '{pattern}' - Expected tag: {true_tag}, Predicted tag: {predicted_tag}")
            if predicted_tag == true_tag:
                correct += 1
            else:
                wrong += 1
        else:
            print(f"Word '{pattern}' - No prediction due to low probability")
            wrong += 1

total = correct + wrong
accuracy = (correct / total) if total > 0 else 0

# Final summary
print(f"Total tests: {total}, Correct predictions: {correct}, Wrong predictions: {wrong}")
print(f"Accuracy: {accuracy:.2f}")