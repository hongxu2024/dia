import random
import json
import pickle
import numpy as np
import pandas as pd
import pdb
import nltk
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import recall_score
from tensorflow.keras.metrics import Recall
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer=WordNetLemmatizer()

with open('intents.json') as json_file:
    intents = json.load(json_file)

#print(intents)

words=[]
classes=[]
documents=[]
ignore_letters=['?','!','.',',']

for intent in intents['intents']:
  for pattern in intent['patterns']:
    word_list=nltk.word_tokenize(pattern)
    words.extend(word_list)
    documents.append((word_list,intent['tag']))
    if intent['tag'] not in classes:
      classes.append(intent['tag'])
# i=0
# for document in documents:
#    print(document)
#    i=i+1
#    if(i == 7):break

words =[lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes=sorted(set(classes))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))



training=[]
output_empty=[0]*len(classes)

for document in documents:
  bag=[]
  word_patterns=document[0]
  words = [lemmatizer.lemmatize(word) for word in words if word and word not in ignore_letters]
  for word in words:
    bag.append(1) if word in word_patterns else bag.append(0)

  output_row=list(output_empty)
  output_row[classes.index(document[1])]=1
  training.append([bag,output_row])
# print(bag)
# print(output_row)
random.shuffle(training)
# training=np.array(training)
# pdb.set_trace()
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

# Building a neural network
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))  # 新增的隐藏层
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# compile the model
sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', Recall()])

# train model
history = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=10, verbose=1)  # 增加训练次数和批量大小
model.save('chatbotmodel.h5', history)
print('Training finished')



# # calculate accuracy
# accuracy = history.history['accuracy']

# recall = history.history['recall']
# epochs = range(1, len(accuracy) + 1)

# # plot the chart

# # plt.plot(epochs, accuracy, 'b', label='Training accuracy')
# plt.plot(epochs, recall, 'r', label='Training recall')
# plt.title('Training recall over epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Metric Value')
# plt.legend()
# plt.show()