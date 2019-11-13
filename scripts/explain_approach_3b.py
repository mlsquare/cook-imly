from explain_data_preprocessing import load_data
from keras import backend as K
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate, concatenate, Flatten, GRU
from keras.utils import to_categorical
from sklearn.metrics import jaccard_score
import distance
import numpy as np

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
    feat_decoder_input = Input(shape=(latent_dim,), name='dec_feat')
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

    merge_layer = concatenate(
        [hidden_state_x, feat_decoder_input, decoder_outputs], name='cat')
    output_chars = Dense(
        data['vocab_size_sent'], activation='softmax', name='op_sent')(merge_layer)
    model = Model([hidden_state_x, feat_decoder_input,
                   input_sent_features], output_chars)
    return model


def features_model(initialize=True, rnn_cell='gru', latent_dim=5):
    latent_dim = latent_dim

    hidden_state_x = Input(shape=(latent_dim,), name='hidden_x')
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
        [hidden_state_x, decoder_outputs], name='cat')
    output_chars = Dense(
        data['vocab_size_feat'], activation='softmax', name='op_sent')(merge_layer)
    model = Model([hidden_state_x, input_seq_features], output_chars)
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

x_latent = get_hidden_x(data['x_feat_seq'], model=label_m)

features_m.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
features_m.fit([x_latent, data['x_seq']], data['y_seq'],
               batch_size=20, epochs=50, verbose=1, shuffle=True, validation_split=0.2)

x_latent = get_hidden_x(data['x_feat'], model=label_m)
feat_dec_output = get_decoder_output(
    x_latent, data['x_ext_seq'], features_m)
print(feat_dec_output.shape)
path_m.compile(optimizer='adam', loss='categorical_crossentropy',
               metrics=['accuracy'])
path_m.fit([x_latent, feat_dec_output, data['x_sent']], data['y_chars'], batch_size=20,
           epochs=200, verbose=1, shuffle=True, validation_split=0.2)


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
    feat_tok = 0  # Root node
    cont = True
    text = [token]
    seq = [feat_tok]
    x_sent = np.zeros(
        (1, data['maxlen'], data['vocab_size_sent']), dtype=np.bool)
    x_seq = np.zeros(
        (1, data['maxlen_seq'], data['vocab_size_feat']), dtype=np.bool)
    x_ext_seq = np.zeros(
        (1, data['maxlen_seq'], data['vocab_size_feat']), dtype=np.bool)
    x_latent = get_hidden_x(x_f, model=label_model)
    x_latent = x_latent.reshape(1, latent_dim)
    x_sent[0, 0, char_indices[token]] = 1
    x_seq[0, 0, feat_tok] = 1
    x_ext_seq[0, 0, feat_tok] = 1
    pred = label_model.predict(x_f)
    label = [np.argmax(pred[0])]
    index = 1
    while (index < data['maxlen_seq']):
        pred_feat = features_m.predict([x_latent, x_seq])
        pred_val = np.argmax(pred_feat[0])
        x_seq[0, index, pred_val] = 1
        next_val = pred_val
        seq.append(next_val)
        x_ext_seq[0, index, seq[index]] = 1
        index += 1

    feat_decoder = get_decoder_output(x_latent, x_ext_seq, features_m)
    index = 1
    while cont & (index < data['maxlen']):
        pred = path_model.predict([x_latent, feat_decoder, x_sent])
        char_index = np.argmax(pred[0])
        x_sent[0, index, char_index] = 1
        next_char = indices_char[char_index]
        text.append(next_char)
        index += 1
        if next_char == 'E':
            cont = False

    return [text, label, seq]


count = []
j_coeff = []
j_coeff_feat = []
l_dist = []
pred_feat_list = []
pred_feat_accuracy = []
for i in range(150):
    curr_feat = np.array([df.iloc[i, 0:4]])
    path, label, seq = sample_paths(curr_feat)
    print('actual vs predicted: ', df.iloc[i, 4], ' vs ', ' '.join(
        path), 'labels: ', df.iloc[i, 5], label[0])
    count.append(df.iloc[i, 5] == label[0])
    actual_path = df.iloc[i, 4].split()
    actual_path_tok = [char_indices[char] for char in actual_path]
    pred_path_tok = [char_indices[char] for char in path]
    j_coeff.append(get_j_coeff(actual_path_tok, pred_path_tok))
    j_coeff_feat.append(get_j_coeff(df.iloc[i, 6], seq))
    l_dist.append(distance.levenshtein(
        df.iloc[i, 4].replace(' ', ''), ''.join(path)))

    print('Actual vs predicted features: ', df.iloc[i, 6], 'vs', seq, '\n')


print('\nLabel accuracy - ', np.mean(count))
print('Path metric (Jaccard) - ', np.mean(j_coeff))
print('Path metric (Levensthein) - ', np.mean(l_dist))
print('Decision feature metric (Jaccard) - ', np.mean(j_coeff_feat))
