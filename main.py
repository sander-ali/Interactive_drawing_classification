import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_data_format('channels_first')
from sklearn import metrics
import seaborn as sns
import cv2
import pickle
# load the data
butterfly = np.load('data/butterfly.npy')
cake = np.load('data/cake.npy')
camera = np.load('data/camera.npy')
cat = np.load('data/cat.npy')
star = np.load('data/star.npy')
octopus = np.load('data/octopus.npy')
#star = np.load('data/star.npy')

butterfly = np.c_[butterfly, np.zeros(len(butterfly))]
cake = np.c_[cake, np.ones(len(cake))]
camera = np.c_[camera, 2*np.ones(len(camera))]
cat = np.c_[cat, 3*np.ones(len(cat))]
star = np.c_[star, 4*np.ones(len(star))]
octopus = np.c_[octopus, 5*np.ones(len(octopus))]
#star = np.c_[star, 6*np.ones(len(star))]
print(butterfly.shape)

def plot_samples(input_array, rows=4, cols=5, title=''):
    fig, ax = plt.subplots(figsize=(cols,rows))
    ax.axis('off')
    plt.title(title)

    for i in list(range(0, min(len(input_array),(rows*cols)) )):      
        a = fig.add_subplot(rows,cols,i+1)
        imgplot = plt.imshow(input_array[i,:784].reshape((28,28)), cmap='gray_r', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])

plot_samples(butterfly, title='Sample face drawings\n')
plot_samples(octopus, title='Sample octopus drawings\n')
#plot_samples(star, title='Sample star drawings\n')
# Merging arrays and splitting the features and labels
X = np.concatenate((butterfly[:10000,:-1], cake[:10000,:-1], camera[:10000,:-1], cat[:10000,:-1], star[:10000, :-1], octopus[:10000, :-1]), axis=0).astype('float32') # all columns but the last
y = np.concatenate((butterfly[:10000,-1], cake[:10000,-1], camera[:10000,-1], cat[:10000,-1],  star[:10000,-1], octopus[:10000,-1]), axis=0).astype('float32') # the last column

# We than split data between train and test (80 - 20 usual ratio). Normalizing the value between 0 and 1
X_train, X_test, y_train, y_test = train_test_split(X/255.,y,test_size=0.2,random_state=0)

y_train_cnn = np_utils.to_categorical(y_train)
y_test_cnn = np_utils.to_categorical(y_test)
num_classes = y_test_cnn.shape[1]

X_train_cnn = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test_cnn = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

def cnn_model():
    # create model
    model = Sequential()
    model.add(Conv2D(56, (3,3), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.7))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

%%time
np.random.seed(0)
# build the model
model_cnn = cnn_model()
# Fit the model
model_cnn.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=20, batch_size=128)
# Final evaluation of the model
scores = model_cnn.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print('Final CNN accuracy: ', scores[1])

y_pred_cnn = model_cnn.predict_classes(X_test_cnn, verbose=0)


c_matrix = metrics.confusion_matrix(y_test, y_pred_cnn)


def confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class_names = ['butterfly', 'cake', 'camera', 'cat', 'star','octopus']
confusion_matrix(c_matrix, class_names, figsize = (10,7), fontsize=14)

#Misclassification when y_pred and y_test are different.
misclassified = X_test[y_pred_cnn == y_test]

plot_samples(misclassified, rows=10, cols=5, title='')



label_dict = {0:'Butterfly', 1:'Cake', 2:'Camera', 3:'Cat', 4:'star', 5:'Octopus'}
image = cv2.imread("test4.jpg")
#image = np.asarray(bytearray(image), dtype="uint8")
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)

vect = np.asarray(resized, dtype="uint8")
vect = vect.reshape(1, 1, 28, 28).astype('float32')
my_prediction = model_cnn.predict(vect)
index = np.argmax(my_prediction[0])
final_pred = label_dict[index]
final_pred