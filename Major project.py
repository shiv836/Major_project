#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, Dense
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from ann_visualizer.visualize import ann_viz;
from keras.callbacks import CSVLogger, ModelCheckpoint
import matplotlib.pyplot as plt


# In[2]:


import os, tqdm, re, time, itertools, sys
import warnings
warnings.filterwarnings('ignore')


# In[3]:


start = time.time()

data_train = pd.read_csv('mitbih_train.csv', header=None)
data_test = pd.read_csv('mitbih_test.csv', header=None)
abnormal = pd.read_csv('ptbdb_abnormal.csv', header=None)
normal = pd.read_csv('ptbdb_normal.csv', header=None)

end = time.time()
print('Time taken: %.3f seconds' % (end-start))

print('Data loaded........')


# In[4]:


data_train


# In[5]:


normal


# In[6]:


plt.plot(normal.values[0])


# In[7]:


abnormal


# In[8]:


data_test


# In[9]:


data_train.isnull().sum().to_numpy()


# In[10]:


normal.shape


# In[11]:


abnormal.shape


# In[12]:


normal = normal.drop([187], axis=1)
abnormal = abnormal.drop([187], axis=1)


# In[13]:


flatten_y = abnormal.values


# In[14]:


flatten_y = flatten_y[:, 5:70].flatten()


# In[15]:


flatten_y


# In[16]:


plt.figure(figsize=(15, 3))
plt.title('ECG Visualization of Abormal Persons')
plt.subplot(1, 5, 1)
plt.plot(abnormal.values[0][5:50])
plt.subplot(1, 5, 2)
plt.plot(abnormal.values[10][5:50])
plt.subplot(1, 5, 3)
plt.plot(abnormal.values[20][5:50])
plt.subplot(1, 5, 4)
plt.plot(abnormal.values[40][5:50])
plt.subplot(1, 5, 5)
plt.plot(abnormal.values[44][5:50])


# In[17]:


plt.figure(figsize=(15, 3))
plt.title('ECG Visualization of Normal Persons')
plt.subplot(1, 5, 1)
plt.plot(normal.values[0][5:50])
plt.subplot(1, 5, 2)
plt.plot(normal.values[10][5:50])
plt.subplot(1, 5, 3)
plt.plot(normal.values[20][5:50])
plt.subplot(1, 5, 4)
plt.plot(normal.values[40][5:50])
plt.subplot(1, 5, 5)
plt.plot(normal.values[77][5:50])


# In[18]:


fig, axs = plt.subplots(4)
fig.suptitle('Vertically stacked ECG plots of Normal People')
axs[0].plot(normal.values[10][1:70])
axs[1].plot(normal.values[55][1:70])
axs[2].plot(normal.values[87][1:70])
axs[3].plot(normal.values[98][1:70])


# In[19]:


target=data_train[187]


# In[20]:


plt.figure(figsize=(10, 8))
circle = plt.Circle((0, 0), 0.7, color='white')
plt.pie(data_train[187].value_counts(), labels=['Non-Ectopic Beats', 'Superventrical Ectopic', 'Ventricular Beats',
                                                'Unknown', 'Fusion Beats'], colors=['blue', 'magenta', 'cyan', 
                                                                                   'red', 'grey'])
p = plt.gcf()
p.gca().add_artist(circle)


# In[21]:


sns.set_style('darkgrid')
plt.figure(figsize=(10, 5))
plt.plot(data_train.iloc[0, 0:187])


# In[22]:


data_1 = data_train[data_train[187] == 1]
data_2 = data_train[data_train[187] == 2]
data_3 = data_train[data_train[187] == 3]
data_4 = data_train[data_train[187] == 4]


# In[23]:


sns.set_style('darkgrid')
plt.figure(figsize=(10, 5))
plt.plot(data_train.iloc[0, 0:187], color='green', label='Normal Heartbeats')
plt.plot(data_1.iloc[0, 0:187], color='blue', label='Supraventricular Heartbeats')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()


# In[24]:


sns.set_style('darkgrid')
plt.figure(figsize=(10, 5))
plt.plot(data_train.iloc[0, 0:187], color='red', label='Normal Heartbeats')
plt.plot(data_4.iloc[0, 0:187], color='grey', label='Unknown Heartbeats')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()


# In[25]:


y_abnormal = np.ones(abnormal.shape[0])
y_abnormal = pd.DataFrame(y_abnormal)


# In[26]:


y_normal = np.zeros(normal.shape[0])
y_normal = pd.DataFrame(y_normal)


# In[27]:


X = pd.concat([abnormal, normal], sort=True)
y = pd.concat([y_abnormal, y_normal], sort=True)


# In[28]:


from sklearn.utils import resample


# In[29]:


data_1_resample = resample(data_1, n_samples=20000, 
                           random_state=123, replace=True)
data_2_resample = resample(data_2, n_samples=20000, 
                           random_state=123, replace=True)
data_3_resample = resample(data_3, n_samples=20000, 
                           random_state=123, replace=True)
data_4_resample = resample(data_4, n_samples=20000, 
                           random_state=123, replace=True)
data_0 = data_train[data_train[187] == 0].sample(n=20000, random_state=123)


# In[30]:


data_1_resample.shape


# In[31]:


data_1.shape


# In[32]:


train_dataset = pd.concat([data_0, data_1_resample, data_2_resample, data_3_resample, 
                          data_4_resample])


# In[33]:


plt.figure(figsize=(10, 8))
circle = plt.Circle((0, 0), 0.7, color='white')
plt.pie(train_dataset[187].value_counts(), labels=['Non-Ectopic Beats', 'Superventrical Ectopic', 'Ventricular Beats',
                                                'Unknown', 'Fusion Beats'], colors=['blue', 'magenta', 'cyan', 
                                                                                   'red', 'grey'])
p = plt.gcf()
p.gca().add_artist(circle)


# In[34]:


target_train = train_dataset[187]
target_test = data_test[187]


# In[35]:


y_train = to_categorical(target_train)
y_test = to_categorical(target_test)
y_train[:4]


# In[44]:


X_train = train_dataset.iloc[:, :-1].values
X_test = data_test.iloc[:, :-1].values


# In[47]:





# In[48]:


X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)
X_train.shape, X_test.shape


# In[41]:


def model():
    model = Sequential()
    
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', padding='same', input_shape=(187, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', padding='same', input_shape=(187, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', padding='same', input_shape=(187, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[42]:


model = model()
model.summary()


# In[43]:


logger = CSVLogger('logs.csv', append=True)
his = model.fit(X_train, y_train, epochs=200
                , batch_size=32, 
          validation_data=(X_test, y_test), callbacks=[logger])


# In[49]:


model.evaluate(X_test, y_test)


# In[50]:


history = his.history
history.keys()


# In[51]:


history['val_accuracy']


# In[52]:


epochs = range(1, len(history['loss']) + 1)
acc = history['accuracy']
loss = history['loss']
val_acc = history['val_accuracy']
val_loss = history['val_loss']

plt.figure(figsize=(10, 5))
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(epochs, acc, label='accuracy')
plt.plot(epochs, val_acc, label='val_acc')
plt.legend()
plt.figure(figsize=(10, 5))
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epochs, loss, label='loss', color='g')
plt.plot(epochs, val_loss, label='val_loss', color='r')
plt.legend()


# In[53]:


y_pred = model.predict(X_test)
y_hat = np.argmax(y_pred, axis = 1)
confusion_matrix(np.argmax(y_test, axis = 1), y_hat)


# In[54]:


plt.figure(figsize=(10, 5))
sns.heatmap(confusion_matrix(np.argmax(y_test, axis = 1), y_hat), annot=True, fmt='0.0f', cmap='RdPu')


# In[55]:


print(classification_report(np.argmax(y_test, axis = 1), y_hat ,target_names=["0","1","2","3","4"]))


# In[58]:


from sklearn.metrics import roc_auc_score
y_test=np.argmax(y_test, axis = 1)


# In[59]:


roc_auc_score(y_test,y_pred,multi_class="ovr")


# In[60]:


from sklearn.metrics import roc_curve


# In[62]:


fpr = {}
tpr = {}
thresh ={}

n_class = 5

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred[:,i], pos_label=i)
    
# plotting    
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Normal vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label=' SV ectopic vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='V ectopic vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--',color='red', label='Fusion beats vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--',color='yellow', label='Unknown beats vs Rest')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC',dpi=300); 


# In[ ]:




