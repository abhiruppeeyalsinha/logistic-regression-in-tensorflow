## Logistic Regression for classification in tensorflow.

import tensorflow as tf
from keras import Model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pickle



# loading iris dataset
iris = datasets.load_iris()

x = iris.data
y = iris.target

# print(x.shape)
# # print(x)
# print(y.shape)

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=.25,random_state=42,shuffle=True)
x_val,x_test, y_val,y_test = train_test_split(x_test,y_test,test_size=.5,random_state=42,shuffle=True)

# print(x_train,x_train.shape)
callback = EarlyStopping(
    monitor='val_loss', patience=20, verbose=1, mode='auto', baseline=None, restore_best_weights=True)


class LogisticReg(Model):
    def __init__(self):
        super(LogisticReg, self).__init__()
        self.dense = Dense(3,activation="softmax")

    def call(self, inputs):
        x = self.dense(inputs)

        return x
    
model = LogisticReg()

# print(model.summary())

model.compile(optimizer="Adam",loss="sparse_categorical_crossentropy",metrics=['acc'])
train_model = model.fit(x_train,y_train,epochs=700,validation_data=(x_val,y_val),callbacks=[callback])
# save_model = train_model.model.save('LogisticReg_700.h5')
# filename = 'Logistic_Reg_700.sav'
# pickle.dump(train_model, open(filename, 'wb'))



test_loss,test_acc = model.evaluate(x_test,y_test)
print(f"Test Accuracy:- {test_acc}")

# plt.plot(train_model.history['acc'])
# plt.plot(train_model.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train','validation'],loc='upper right')


plt.plot(train_model.history['loss'])
plt.plot(train_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper left')
plt.show()

# loaded_model = pickle.load(open(r'E:\DICOM\medical dataset\Logistic_Reg_700.sav', 'rb'))
# result = loaded_model.score(x_test, y_test)
# print(result)


