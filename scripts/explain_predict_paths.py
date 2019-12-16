#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


# In[2]:


iris = load_iris()
X = iris['data']
y = iris['target']
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y)


# In[3]:


left_nodes = clf.tree_.children_left[clf.tree_.children_left>0]
right_nodes = clf.tree_.children_right[clf.tree_.children_right>0]
node_indicator = clf.decision_path(X)
path_list = []
for i, j in enumerate(X):
    path_list.append(node_indicator.indices[node_indicator.indptr[i]:node_indicator.indptr[i+1]])

## Convert path to strings
path_column = np.array([])
for i, j in enumerate(X):
    path_as_string = []
    for node in path_list[i]:
        if node == 0:
            path_as_string.append('S')
        elif node in left_nodes:
            path_as_string.append('L')
        elif node in right_nodes:
            path_as_string.append('R')
            
    path_as_string.append('E')
    path_as_string = ' '.join(path_as_string)
    path_column = np.append(path_column, path_as_string)


# In[4]:


chars = ['S', 'L', 'R', 'E']
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

Xnew = np.hstack((X, path_column.reshape(-1,1)))
path_sequence = Xnew[:,4]
data = pd.DataFrame(Xnew)
data[5]=y
df = data.sample(frac=1).reset_index(drop=True)

# prepare dataset for training
get_path_lengths = lambda t: len(t.split())
paths_lengths = np.array([get_path_lengths(xi) for xi in path_sequence])


# In[5]:


vocab_size = 4
label_size = 3
feature_size = 4
# cut the text in semi-redundant sequences of maxlen characters
maxlen = np.max(paths_lengths)
sentences = []
next_chars = []
features = []
labels = []

for i in range(0, len(df)):
    # get the feature
    curr_feat = np.array([df.iloc[i,0:4]])
    curr_path = df.iloc[i,4].split()
    curr_path_len = len(curr_path)
    curr_label = y[i]
    for j in range(1,curr_path_len):
        features.append(curr_feat)
        labels.append(curr_label)
        sentences.append(curr_path[0:j])
        next_chars.append(curr_path[j])
print('Vectorization...')

x_sent = np.zeros((len(sentences), maxlen, vocab_size), dtype=np.bool)
x_feat = np.zeros((len(sentences), feature_size), dtype=np.float)
y_chars = np.zeros((len(sentences), vocab_size), dtype=np.bool)
y_feat = np.zeros((len(sentences), label_size), dtype=np.float)
#from keras.utils import to_categorical
#y_feat_tmp = to_categorical(df[5])


# In[6]:


for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x_sent[i, t, char_indices[char]] = 1
    y_chars[i, char_indices[next_chars[i]]] = 1
    x_feat[i,:] = features[i]
    y_feat[i,labels[i]]=1


# In[7]:


index = 10
print(y_chars[index],y_feat[index],x_sent[index],x_feat[index])
print(y_chars.shape,y_feat.shape,x_sent.shape,x_feat.shape)


# In[8]:


from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate, concatenate, Flatten

h1_size = 5
latent_dim = 5

input_x_features = Input(shape=(feature_size,),name='ip_x')
hidden_state_x = Dense(h1_size, activation='relu',name='hidden_x')(input_x_features)
output_labels = Dense(3, activation='softmax',name='op_x')(hidden_state_x)

input_sent_features = Input(shape=(maxlen,vocab_size),name='ip_sent')
decoder = LSTM(latent_dim,return_state=False,return_sequences=False,name='lstm_sent')
decoder_outputs = decoder(input_sent_features)

merge_layer = concatenate([hidden_state_x,decoder_outputs],name='cat')
output_chars = Dense(vocab_size, activation='softmax',name='op_sent')(merge_layer)
model = Model([input_x_features,input_sent_features], [output_labels,output_chars])
model.summary()


# In[9]:


def paths_joint_model(initialize=True, rnn_cell= 'gru'):
    from keras.models import Model
    from keras.layers import Input, LSTM, Dense, Concatenate, concatenate, Flatten, GRU
    h1_size = 5
    latent_dim = 5
    
    input_x_features = Input(shape=(feature_size,),name='ip_x')
    hidden_state_x = Dense(h1_size, activation='relu',name='hidden_x')(input_x_features)
    output_labels = Dense(3, activation='softmax',name='op_x')(hidden_state_x)
    
    input_sent_features = Input(shape=(maxlen,vocab_size),name='ip_sent')
    if rnn_cell == 'gru':
        RNN = GRU
    else:
        RNN = LSTM
            
    decoder = RNN(latent_dim,return_state=False,return_sequences=False,name='lstm_sent')
    if initialize:
        decoder_outputs = decoder(input_sent_features,initial_state=hidden_state_x)
    else:
        decoder_outputs = decoder(input_sent_features)
    
    merge_layer = concatenate([hidden_state_x,decoder_outputs],name='cat')
    output_chars = Dense(vocab_size, activation='softmax',name='op_sent')(merge_layer)
    model = Model([input_x_features,input_sent_features], [output_labels,output_chars])
    return model


# In[10]:


model = paths_joint_model()
model.compile(optimizer='adam', loss={'op_x':'categorical_crossentropy','op_sent':'categorical_crossentropy'},metrics=['accuracy'])
model.fit([x_feat,x_sent],[y_feat,y_chars],batch_size =20, epochs = 2,verbose=1)


# In[11]:


def sample(x):
    n = x.shape[0]
    x_f = x.reshape(1,feature_size)
    token = 'S'
    cont = True
    text = [token]
    x_sent = np.zeros((1,maxlen,vocab_size),dtype=np.bool)
    x_sent[0,0,char_indices[token]] = 1
    label = []
    index = 1
    while cont & (index <maxlen):
        pred = model.predict([x_f.reshape(1,feature_size),x_sent])
        char_index = np.argmax(pred[1])
        label.append(np.argmax(pred[0])) 
        x_sent[0,index,char_index] = 1
        next_char = indices_char[char_index]
        text.append(next_char)
        index += 1    
        if next_char == 'E':
            cont = False
    return [text,label]


# In[12]:


count = []
for i in range(10,20):
    curr_feat = np.array([df.iloc[i,0:4]])
    path,label= sample(curr_feat)
    print('actual vs predicted: ', df.iloc[i,4] ,' vs ', ' '.join(path), 'labels: ', df.iloc[i,5],label[0])
    count.append(df.iloc[i,5]==label[0])
np.mean(count)


# In[14]:


def paths_model(initialize=True, rnn_cell= 'gru',latent_dim = 5):
    from keras.models import Model
    from keras.layers import Input, LSTM, Dense, Concatenate, concatenate, Flatten, GRU
    latent_dim = latent_dim
    
    hidden_state_x = Input(shape=(latent_dim,),name='hidden_x')
    input_sent_features = Input(shape=(maxlen,vocab_size),name='ip_sent')
    if rnn_cell == 'gru':
        RNN = GRU
    else:
        RNN = LSTM
            
    decoder = RNN(latent_dim,return_state=False,return_sequences=False,name='gru_sent')
    if initialize:
        decoder_outputs = decoder(input_sent_features,initial_state=hidden_state_x)
    else:
        decoder_outputs = decoder(input_sent_features)
    
    merge_layer = concatenate([hidden_state_x,decoder_outputs],name='cat')
    output_chars = Dense(vocab_size, activation='softmax',name='op_sent')(merge_layer)
    model = Model([hidden_state_x,input_sent_features], output_chars)
    return model

def label_model(feature_size = 4, latent_dim = 5):
    from keras.models import Model
    from keras.layers import Input, LSTM, Dense, Concatenate, concatenate, Flatten, GRU
    feature_size = feature_size
    h1_size = latent_dim
    input_x_features = Input(shape=(feature_size,),name='ip_x')
    hidden_state_x1 = Dense(20, activation='relu',name='hidden_x1')(input_x_features)
    hidden_state_x = Dense(h1_size, activation='relu',name='hidden_x')(hidden_state_x1)
    output_labels = Dense(3, activation='softmax',name='op_x')(hidden_state_x)    
    model = Model(input_x_features,output_labels)
    return model

from keras import backend as K

def get_hidden_x(x,model=model,layer_num=2):
    def get_hidden_x_inner(model,layer_num=layer_num):
        return K.function([model.layers[0].input], [model.layers[layer_num].output])
    return get_hidden_x_inner(model,layer_num=layer_num)([x])[0]


# In[15]:


path_m = paths_model()
path_m.summary()
label_m = label_model()
label_m.summary()


# In[16]:


label_m.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
label_m.fit(x_feat,y_feat,batch_size =20, epochs = 500,verbose=1)

x_latent = get_hidden_x(x_feat,model=label_m)

path_m.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
path_m.fit([x_latent,x_sent],y_chars,batch_size =20, epochs = 500,verbose=0)


# In[17]:


def sample_paths(x,path_model=path_m,label_model=label_m,latent_dim=latent_dim,feature_size=feature_size):
    n = x.shape[0]
    x_f = x.reshape(1,feature_size)
    token = 'S'
    cont = True
    text = [token]
    x_sent = np.zeros((1,maxlen,vocab_size),dtype=np.bool)
    x_latent = get_hidden_x(x_f,model=label_model)
    x_latent = x_latent.reshape(1,latent_dim)
    x_sent[0,0,char_indices[token]] = 1
    pred = label_model.predict(x_f)
    label = [np.argmax(pred[0])]
    index = 1
    while cont & (index <maxlen):
        pred = path_model.predict([x_latent,x_sent])
        char_index = np.argmax(pred[0])
        x_sent[0,index,char_index] = 1
        next_char = indices_char[char_index]
        text.append(next_char)
        index += 1    
        if next_char == 'E':
            cont = False
    return [text,label]

count = []
for i in range(10):
    curr_feat = np.array([df.iloc[i,0:4]])
    path,label= sample_paths(curr_feat)
    print('actual vs predicted: ', df.iloc[i,4] ,' vs ', ' '.join(path), 'labels: ', df.iloc[i,5],label[0])
    count.append(df.iloc[i,5]==label[0])
np.mean(count)

