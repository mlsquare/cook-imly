from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler


def load_data():
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    clf = DecisionTreeClassifier(random_state=0)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.6, shuffle=True)
    clf.fit(X_train, y_train)

    left_nodes = clf.tree_.children_left[clf.tree_.children_left > 0]
    right_nodes = clf.tree_.children_right[clf.tree_.children_right > 0]
    node_indicator = clf.decision_path(X)
    path_list = []
    for i, j in enumerate(X):
        path_list.append(
            node_indicator.indices[node_indicator.indptr[i]:node_indicator.indptr[i+1]])

    # Convert path to strings
    path_column = np.array([])
    dec_feat_column = []
    cutpoints_column = []
    for i, j in enumerate(X):
        path_as_string = []
        dec_feat = []
        cutpoints = []
        for node in path_list[i]:
            if node == 0:
                dec_feat.append(0)
                path_as_string.append('S')
                dec_feat.append(clf.tree_.feature[node]+1)
                cutpoints.append(np.around(clf.tree_.threshold[node], 1))
            elif node in left_nodes:
                path_as_string.append('L')
                if clf.tree_.feature[node] >= 0:
                    dec_feat.append(clf.tree_.feature[node]+1)
                    cutpoints.append(np.around(clf.tree_.threshold[node], 1))
                # else:
                    # dec_feat.append(0)
            elif node in right_nodes:
                path_as_string.append('R')
                if clf.tree_.feature[node] >= 0:
                    dec_feat.append(clf.tree_.feature[node]+1)
                    cutpoints.append(np.around(clf.tree_.threshold[node], 1))
                # else:
                    # dec_feat.append(0)

        path_as_string.append('E')
        # dec_feat.append(0)
        dec_feat = np.array(dec_feat)
        cutpoints = np.array(cutpoints)
        path_as_string = ' '.join(path_as_string)
        path_column = np.append(path_column, path_as_string)
        dec_feat_column.append(dec_feat)
        cutpoints_column.append(cutpoints)

    chars = ['S', 'L', 'R', 'E']
    trimmed_chars = ['L', 'R', 'E']

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    trimmed_char_indices = dict((c, i) for i, c in enumerate(trimmed_chars))
    trimmed_indices_char = dict((i, c) for i, c in enumerate(trimmed_chars))

    Xnew = np.hstack((X, path_column.reshape(-1, 1)))
    path_sequence = Xnew[:, 4]
    data = pd.DataFrame(Xnew)
    data[5] = y
    data[6] = np.array(dec_feat_column)
    data[7] = np.array(cutpoints_column)
    df = data.sample(frac=1).reset_index(drop=True)

    # prepare dataset for training
    def get_path_lengths(t): return len(t.split())
    paths_lengths = np.array([get_path_lengths(xi) for xi in path_sequence])

    # Modified for decision feature prediction

    vocab_size_feat = 5  # Or is it 4(should 0 be counted in)?
    vocab_size_sent = 4
    vocab_trimmed = 3
    label_size = 3
    feature_size = 4
    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = np.max(paths_lengths)
    maxlen_seq = maxlen-1

    sentences = []
    feat_seq = []
    ext_feat_seq = []
    trimmed_path_seq = []
    next_chars = []
    next_dec_feature = []
    next_trimmed_char = []
    features = []
    labels = []

    for i in range(0, len(df)):
        # get the feature
        curr_feat = np.array([df.iloc[i, 0:4]])
        # curr_feat_seq_len = len(curr_feat)
        curr_path = df.iloc[i, 4].split()
        curr_path_len = len(curr_path)
        curr_label = y[i]
        curr_dec_feat = df.iloc[i, 6]
        curr_trimmed_path = [n for n in curr_path if n != 'S']
        for j in range(1, curr_path_len):
            features.append(curr_feat)
            labels.append(curr_label)
            sentences.append(curr_path[0:j])
            next_chars.append(curr_path[j])
        for k in range(1, len(curr_dec_feat)):
            next_dec_feature.append(curr_dec_feat[k])
            feat_seq.append(curr_dec_feat[0:k])
            trimmed_path_seq.append(curr_trimmed_path[0:k])
            next_trimmed_char.append(curr_trimmed_path[k])
        for k in range(0, len(curr_dec_feat)):
            ext_feat_seq.append(curr_dec_feat[0:k+1])
    print('Vectorization...')

    x_sent = np.zeros((len(sentences), maxlen, vocab_size_sent), dtype=np.bool)
    x_seq = np.zeros((len(feat_seq), maxlen_seq,
                      vocab_size_feat), dtype=np.bool)
    x_trimmed_sent = np.zeros(
        (len(trimmed_path_seq), maxlen_seq, vocab_trimmed), dtype=np.bool)
    x_sent_dec_input = np.zeros(
        (len(trimmed_path_seq), maxlen, vocab_size_sent), dtype=np.bool)

    x_feat = np.zeros((len(sentences), feature_size), dtype=np.float)
    x_feat_seq = np.zeros((len(feat_seq), feature_size), dtype=np.float)

    # Verify maxlen in this case
    x_ext_seq = np.zeros((len(ext_feat_seq), maxlen_seq,
                          vocab_size_feat), dtype=np.bool)

    y_chars = np.zeros((len(sentences), vocab_size_sent), dtype=np.bool)
    y_seq = np.zeros((len(feat_seq), vocab_size_feat), dtype=np.bool)
    y_trimmed_chars = np.zeros((len(feat_seq), vocab_trimmed), dtype=np.bool)

    y_feat = np.zeros((len(sentences), label_size), dtype=np.float)
    y_feat_seq = np.zeros((len(feat_seq), label_size), dtype=np.float)

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

    for i, sentence in enumerate(trimmed_path_seq):
        for t, char in enumerate(sentence):
            x_trimmed_sent[i, t, trimmed_char_indices[char]] = 1
            x_sent_dec_input[i, t, char_indices[char]] = 1
        y_trimmed_chars[i, trimmed_char_indices[next_trimmed_char[i]]] = 1

    for i, feat in enumerate(ext_feat_seq):
        for t, val in enumerate(feat):
            x_ext_seq[i, t, val] = 1

    data = {
        'df': df,
        'x_sent': x_sent,
        'x_seq': x_seq,
        'x_trimmed_sent': x_trimmed_sent,
        'x_sent_dec_input': x_sent_dec_input,
        'x_feat': x_feat,
        'x_feat_seq': x_feat_seq,
        'x_ext_seq': x_ext_seq,
        'y_chars': y_chars,
        'y_seq': y_seq,
        'y_trimmed_chars': y_trimmed_chars,
        'y_feat': y_feat,
        'y_feat_seq': y_feat_seq,
        'maxlen': maxlen,
        'maxlen_seq': maxlen_seq,
        'vocab_size_feat': vocab_size_feat,
        'vocab_size_sent': vocab_size_sent,
        'vocab_trimmed': vocab_trimmed,
        'X': X,
        'y': y
    }
    return data
