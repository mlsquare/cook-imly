from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate, concatenate, GRU

# Load data
iris = load_iris()
X = iris['data']
y = iris['target']
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
clf = DecisionTreeClassifier(random_state=0)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.6, shuffle=True)
clf.fit(X_train, y_train)

# Prepare data
left_nodes = clf.tree_.children_left[clf.tree_.children_left > 0]
right_nodes = clf.tree_.children_right[clf.tree_.children_right > 0]
node_indicator = clf.decision_path(X)
path_list = []
for i, j in enumerate(X):
    path_list.append(node_indicator.indices[node_indicator.indptr[i]:node_indicator.indptr[i+1]])

path_column = np.array([])
dec_feat_column = []
for i, j in enumerate(X):
    path_as_string = []
    dec_feat = []
    for node in path_list[i]:
        if node == 0:
            path_as_string.append('S')
            dec_feat.append(clf.tree_.feature[node])
        elif node in left_nodes:
            path_as_string.append('L')
            if clf.tree_.feature[node] > 0:
                dec_feat.append(clf.tree_.feature[node])
        elif node in right_nodes:
            path_as_string.append('R')
            if clf.tree_.feature[node] > 0:
                dec_feat.append(clf.tree_.feature[node])

    path_as_string.append('E')
    dec_feat = np.array(dec_feat)
    path_as_string = ' '.join(path_as_string)
    path_column = np.append(path_column, path_as_string)
    dec_feat_column.append(dec_feat)


chars = ['S', 'L', 'R', 'E']
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

Xnew = np.hstack((X, path_column.reshape(-1, 1)))
path_sequence = Xnew[:,4]
data = pd.DataFrame(Xnew)
data[5] = y
data[6] = np.array(dec_feat_column)
df = data.sample(frac=1).reset_index(drop=True)

# prepare dataset for training
get_path_lengths = lambda t: len(t.split())
paths_lengths = np.array([get_path_lengths(xi) for xi in path_sequence])

vocab_size_feat = 3
vocab_size = 4
label_size = 3
feature_size = 4
maxlen = np.max(paths_lengths)
maxlen_seq = maxlen-2
sentences = []
feat_seq = []
next_chars = []
features = []
labels = []
next_dec_feature = []

for i in range(0, len(df)):
    # get the feature
    curr_feat = np.array([df.iloc[i, 0:4]])
    curr_feat_seq_len = len(curr_feat)
    curr_path = df.iloc[i, 4].split()
    curr_path_len = len(curr_path)
    curr_label = y[i]
    curr_dec_feat = df.iloc[i, 6]
    for j in range(1,curr_path_len):
        features.append(curr_feat)
        labels.append(curr_label)
        sentences.append(curr_path[0:j])
        next_chars.append(curr_path[j])
    for k in range(1, len(curr_dec_feat)):
        next_dec_feature.append(curr_dec_feat[k])
        feat_seq.append(curr_dec_feat[0:k])
print('Vectorization...')

x_sent = np.zeros((len(sentences), maxlen, vocab_size), dtype=np.bool)
x_seq = np.zeros((len(feat_seq), maxlen_seq, 4), dtype=np.bool)

x_feat = np.zeros((len(sentences), feature_size), dtype=np.float)
x_feat_seq = np.zeros((len(feat_seq), feature_size), dtype=np.float)

y_chars = np.zeros((len(sentences), vocab_size), dtype=np.bool)
y_seq = np.zeros((len(feat_seq), 4), dtype=np.bool)

y_feat = np.zeros((len(sentences), label_size), dtype=np.float)
y_feat_seq = np.zeros((len(feat_seq), label_size), dtype=np.float)

# Populate inputs
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x_sent[i, t, char_indices[char]] = 1
    y_chars[i, char_indices[next_chars[i]]] = 1
    x_feat[i, :] = features[i]
    y_feat[i, labels[i]] = 1

for i, feat in enumerate(feat_seq):
    for t, val in enumerate(feat):
        x_seq[i, t, val] = 1
    y_seq[i, next_dec_feature[i]] = 1
    x_feat_seq[i, :] = features[i]
    y_feat_seq[i, labels[i]] = 1


def paths_model(initialize=True, rnn_cell= 'gru',latent_dim = 5):
    latent_dim = latent_dim

    hidden_state_x = Input(shape=(latent_dim,),name='hidden_x')
    input_sent_features = Input(shape=(maxlen,vocab_size),name='ip_sent')
    # input_path/cutpoint/feat_features --> triplet tuple
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
    # concat -- All three with hidden_state_x
    output_chars = Dense(vocab_size, activation='softmax',name='op_sent')(merge_layer)
    model = Model([hidden_state_x,input_sent_features], output_chars)
    return model

# Define all the three models
def features_model(initialize=True, rnn_cell= 'gru',latent_dim = 5):
    latent_dim = latent_dim

    hidden_state_x = Input(shape=(latent_dim,),name='hidden_x')
    input_sent_features = Input(shape=(maxlen_seq,vocab_size),name='ip_sent')
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
    feature_size = feature_size
    h1_size = latent_dim
    input_x_features = Input(shape=(feature_size,),name='ip_x')
    hidden_state_x1 = Dense(20, activation='tanh',name='hidden_x1')(input_x_features)
    hidden_state_x2 = Dense(20, activation='tanh',name='hidden_x2')(hidden_state_x1)
    hidden_state_x3 = Dense(h1_size, activation='tanh',name='hidden_x3')(hidden_state_x2)
    output_labels = Dense(3, activation='softmax',name='op_x')(hidden_state_x3)    
    model = Model(input_x_features,output_labels)
    return model


def get_hidden_x(x,model,layer_num=3):
    def get_hidden_x_inner(model,layer_num=layer_num):
        return K.function([model.layers[0].input], [model.layers[layer_num].output])
    return get_hidden_x_inner(model,layer_num=layer_num)([x])[0]

path_m = paths_model()
label_m = label_model()
features_m = features_model()

y_cat = to_categorical(y)

label_m.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
label_m_history = label_m.fit(X,y_cat,batch_size =20, epochs = 200,verbose=0, shuffle=True)

x_latent = get_hidden_x(x_feat,model=label_m)

path_m.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
path_m.fit([x_latent,x_sent],y_chars,batch_size =20, epochs = 200,verbose=0, shuffle=True)

x_latent = get_hidden_x(x_feat_seq,model=label_m)

features_m.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
features_m.fit([x_latent,x_seq],y_seq,batch_size =20, epochs = 200,verbose=0, shuffle=True)

from sklearn.metrics import accuracy_score
# import distance
# jaccard_score

latent_dim = 5

def jaccard_score_inconsistent(x,y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)
# def get_j_coeff(a,b):
#     if len(a) != len(b):
#         return jaccard_score_inconsistent(a, b)
#     return jaccard_score(a, b, average='micro')
        

def sample_paths(x,path_model=path_m,label_model=label_m,latent_dim=latent_dim,feature_size=feature_size):
    n = x.shape[0]
    x_f = x.reshape(1,feature_size)
    token = 'S'
    feat_tok = 3
    cont = True
    text = [token]
    seq = []
    feat = []
    x_sent = np.zeros((1,maxlen,vocab_size),dtype=np.bool)
    x_latent = get_hidden_x(x_f,model=label_model)
    x_latent = x_latent.reshape(1,latent_dim)
    x_sent[0,0,char_indices[token]] = 1
    # x_seq[0,0,feat_tok] = 1
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
    index = 0
    while (index <text.index('E')-1):
        pred = features_m.predict([x_latent,x_seq])
        pred_val = np.argmax(pred[0])
        x_seq[0,index,pred_val] = 1
        next_val = pred_val
        seq.append(next_val)
        index += 1 
    return [text,label,seq]

count = []
j_coeff = []
l_dist = []
pred_feat_list = []
pred_feat_accuracy = []
for i in range(6):
    curr_feat = np.array([df.iloc[i,0:4]])
    path,label,seq= sample_paths(curr_feat)
    print('actual vs predicted: ', df.iloc[i,4] ,' vs ', ' '.join(path), 'labels: ', df.iloc[i,5],label[0])
    print('Actual vs predicted features: ',df.iloc[i,6], 'vs', seq,'\n')
    count.append(df.iloc[i,5]==label[0])
    actual_path = df.iloc[i,4].split()
    actual_path_tok = [char_indices[char] for char in actual_path]
    pred_path_tok = [char_indices[char] for char in path]
    # j_coeff.append(get_j_coeff(actual_path_tok, pred_path_tok))
    # l_dist.append(distance.levenshtein(df.iloc[i,4].replace(' ', ''), ''.join(path)))
    

print('\nLabel accuracy - ',np.mean(count))
# print('Path metric (Jaccard) - ',np.mean(j_coeff)) # sklearn version issues
# print('Path metric (Levensthein) - ',np.mean(l_dist)) # pip install Distance package
