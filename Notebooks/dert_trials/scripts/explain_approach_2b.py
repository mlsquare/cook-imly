from explain_data_preprocessing import load_data
from keras import backend as K
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate, concatenate, Flatten, GRU
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from sklearn.metrics import jaccard_score
import distance
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

feature_size = 4
latent_dim = 5

data = load_data()
df = data['df']

chars = ['S', 'L', 'R', 'E']
trimmed_chars = ['L', 'R', 'E']

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

trimmed_char_indices = dict((c, i) for i, c in enumerate(trimmed_chars))
trimmed_indices_char = dict((i, c) for i, c in enumerate(trimmed_chars))


def paths_model(initialize=True, rnn_cell='gru', latent_dim=5):
    latent_dim = latent_dim

    hidden_state_x = Input(shape=(latent_dim,), name='hidden_x')
    input_sent_features = Input(
        shape=(data['maxlen'], data['vocab_size_sent']), name='ip_sent')
    if rnn_cell == 'gru':
        RNN = GRU
    else:
        RNN = LSTM

    decoder = RNN(latent_dim, return_state=False,
                  return_sequences=False, name='gru_sent')
    if initialize:
        decoder_outputs = decoder(
            input_sent_features, initial_state=hidden_state_x)
    else:
        decoder_outputs = decoder(input_sent_features)

    merge_layer = concatenate([hidden_state_x, decoder_outputs], name='cat')
    output_chars = Dense(
        data['vocab_size_sent'], activation='softmax', name='op_sent')(merge_layer)
    model = Model([hidden_state_x, input_sent_features], output_chars)
    return model


def features_model(initialize=True, rnn_cell='gru', latent_dim=5):
    latent_dim = latent_dim

    hidden_state_x = Input(shape=(latent_dim,), name='hidden_x')
    sent_decoder_input = Input(shape=(latent_dim,), name='dec_sent')
    input_seq_features = Input(
        shape=(data['maxlen_seq'], data['vocab_size_feat']), name='ip_seq')
    if rnn_cell == 'gru':
        RNN = GRU
    else:
        RNN = LSTM

    decoder = RNN(latent_dim, return_state=False,
                  return_sequences=False, name='gru_seq')
    if initialize:
        decoder_outputs = decoder(
            input_seq_features, initial_state=hidden_state_x)
    else:
        decoder_outputs = decoder(input_seq_features)

    merge_layer = concatenate(
        [hidden_state_x, sent_decoder_input, decoder_outputs], name='cat')
    output_chars = Dense(
        data['vocab_size_feat'], activation='softmax', name='op_sent')(merge_layer)
    model = Model([hidden_state_x, sent_decoder_input,
                   input_seq_features], output_chars)
    return model


def label_model(feature_size=4, latent_dim=5):
    feature_size = feature_size
    h1_size = latent_dim
    input_x_features = Input(shape=(feature_size,), name='ip_x')
    hidden_state_x1 = Dense(20, activation='tanh',
                            name='hidden_x1')(input_x_features)
    hidden_state_x2 = Dense(20, activation='tanh',
                            name='hidden_x2')(hidden_state_x1)
    hidden_state_x3 = Dense(h1_size, activation='tanh',
                            name='hidden_x3')(hidden_state_x2)
    output_labels = Dense(3, activation='softmax',
                          name='op_x')(hidden_state_x3)
    model = Model(input_x_features, output_labels)
    return model


def get_hidden_x(x, model, layer_num=3):
    def get_hidden_x_inner(model, layer_num=layer_num):
        return K.function([model.layers[0].input], [model.layers[layer_num].output])
    return get_hidden_x_inner(model, layer_num=layer_num)([x])[0]


def get_decoder_output(x1, x2, model, layer_num=2):
    temp_layer = K.function([model.layers[0].input, model.layers[1].input], [
                            model.layers[layer_num].output])
    return temp_layer([x1, x2])[0]


path_m = paths_model()
label_m = label_model()
features_m = features_model()

y_cat = to_categorical(data['y'])

label_m.compile(optimizer='adam',
                loss='categorical_crossentropy', metrics=['accuracy'])
label_m_history = label_m.fit(
    data['X'], y_cat, batch_size=20, epochs=200, verbose=1, shuffle=True, validation_split=0.2)

x_latent = get_hidden_x(data['x_feat'], model=label_m)

path_m.compile(optimizer='adam', loss='categorical_crossentropy',
               metrics=['accuracy'])
path_m.fit([x_latent, data['x_sent']], data['y_chars'], batch_size=20,
           epochs=250, verbose=1, shuffle=True, validation_split=0.2)

x_latent = get_hidden_x(data['x_feat_seq'], model=label_m)
sent_dec_output = get_decoder_output(x_latent, data['x_sent_dec_input'], path_m)

features_m.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
features_m.fit([x_latent, sent_dec_output, data['x_seq']], data['y_seq'],
               batch_size=20, epochs=60, verbose=1, shuffle=True, validation_split=0.2)


latent_dim = 5


def jaccard_score_inconsistent(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)


def get_j_coeff(a, b):
    if len(a) != len(b):
        return jaccard_score_inconsistent(a, b)
    return jaccard_score(a, b, average='micro')


def sample_paths(x, path_model=path_m, label_model=label_m, latent_dim=latent_dim, feature_size=feature_size):
    x_f = x.reshape(1, feature_size)
    token = 'S'
    # feat_tok = df.iloc[0, 6][0]  # Root node
    feat_tok = 0
    cont = True
    text = [token]
    seq = [feat_tok]
    x_sent = np.zeros((1, data['maxlen'], data['vocab_size_sent']), dtype=np.bool)
    x_seq = np.zeros((1, data['maxlen_seq'], data['vocab_size_feat']), dtype=np.bool)
    x_latent = get_hidden_x(x_f, model=label_model)
    x_latent = x_latent.reshape(1, latent_dim)
    x_sent[0, 0, char_indices[token]] = 1
    x_seq[0, 0, feat_tok] = 1
    pred = label_model.predict(x_f)
    label = [np.argmax(pred[0])]
    index = 1
    while cont & (index < data['maxlen']):
        pred = path_model.predict([x_latent, x_sent])
        char_index = np.argmax(pred[0])
        x_sent[0, index, char_index] = 1
        next_char = indices_char[char_index]
        text.append(next_char)
        index += 1
        if next_char == 'E':
            cont = False

    sent_decoder = get_decoder_output(x_latent, x_sent, path_m)
    index = 1
    while (index < len(text)-1):
        pred_feat = features_m.predict([x_latent, sent_decoder, x_seq])
        pred_val = np.argmax(pred_feat[0])
        x_seq[0, index, pred_val] = 1
        next_val = pred_val
        seq.append(next_val)
        index += 1
    return [text, label, seq]


count = []
j_coeff = []
j_coeff_feat = []
l_dist = []
bleu_score = []
pred_feat_list = []
pred_feat_accuracy = []
for i in range(150):
    curr_feat = np.array([df.iloc[i, 0:4]])
    path, label, seq = sample_paths(curr_feat)
    print('actual vs predicted: ', df.iloc[i, 4], ' vs ', ' '.join(
        path), 'labels: ', df.iloc[i, 5], label[0])
    count.append(df.iloc[i, 5] == label[0])
    actual_path = df.iloc[i, 4].split()
    bleu_score.append(sentence_bleu([actual_path], path))
    actual_path_tok = [char_indices[char] for char in actual_path]
    pred_path_tok = [char_indices[char] for char in path]
    j_coeff.append(get_j_coeff(actual_path_tok, pred_path_tok))
    j_coeff_feat.append(get_j_coeff(df.iloc[i, 6], seq))
    l_dist.append(distance.levenshtein(
        df.iloc[i, 4].replace(' ', ''), ''.join(path)))

    print('Actual vs predicted features: ', df.iloc[i, 6], 'vs', seq, '\n')

print('\nBlue score - ', np.mean(bleu_score))
print('\nLabel accuracy - ', np.mean(count))
print('Path metric (Jaccard) - ', np.mean(j_coeff))
print('Path metric (Levensthein) - ', np.mean(l_dist))
print('Decision feature metric (Jaccard) - ', np.mean(j_coeff_feat))


# def save_model(model):
#     model_json = model.to_json()
#     with open(model.alter_name+".json", "w") as json_file:
#         json_file.write(model_json)
#     # serialize weights to HDF5
#     model.save_weights(model.alter_name + ".h5")
#     print("Saved model to disk")


# label_m.alter_name = 'label_model'
# features_m.alter_name = 'features_model'
# path_m.alter_name = 'path_model'

# for m in [label_m, path_m, features_m]:
#     save_model(m)

## save new df as csv
## norm_features, exp(tuple) and label

# new_df = df.iloc[:,:4]
# new_df[4] = df.iloc[:,5]
# exp_list = []
# for i, val in df.iterrows():
#     temp_list = []
#     for j, char in enumerate(val[4].split()):
#         if char != 'S' and char != 'E':
#             temp_list.append((char, val[6][j], val[7][j-1]))
#     exp_list.append(temp_list)

# new_df[5] = exp_list
# new_df[0] = new_df[[0,1,2,3]].values.tolist()
# new_df.drop([1,2,3], axis=1, inplace=True)
# for i, val in new_df.iterrows():
#     for j, string in enumerate(val[0]):
#         new_df[0][i][j] = np.around(float(string),2)
# new_df.to_csv('pes_dataset.csv')


# # K fold validation
# def kfold_validation(X, Y, model_fn):
#     kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
#     cvscores = []
#     for train, test in kfold.split(X, Y):
#         model = model_fn()
#         model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
#         scores = model.evaluate(X[test], Y[test], verbose=0)
#         print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#         cvscores.append(scores[1] * 100)
#     print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


# y_cat = to_categorical(data['y'])
# kfold_validation(data['X'], y_cat, label_model)