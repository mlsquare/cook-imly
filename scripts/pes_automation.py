# from keras.models import model_from_json
# from keras import backend as K
# import numpy as np
# import pandas as pd
from flask import request, url_for
from flask_api import FlaskAPI, status, exceptions


# # Laod the three models
# def load_model(model_name):
#     json_file = open(model_name + '.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     loaded_model.load_weights(model_name + ".h5")
#     print("Loaded model from disk")

#     loaded_model.compile(
#         optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return loaded_model


# path_model = load_model('path_model')
# features_model = load_model('features_model')
# label_model = load_model('label_model')

# # Accept inputs

# # Predict the path

# path_model.summary()
# label_model.summary()
# features_model.summary()

# maxlen = 6
# vocab_size_sent = 4
# maxlen_seq = 5
# vocab_size_feat = 5
# latent_dim = 5
# feature_size = 4
# chars = ['S', 'L', 'R', 'E']
# char_indices = dict((c, i) for i, c in enumerate(chars))
# indices_char = dict((i, c) for i, c in enumerate(chars))


# def get_hidden_x(x, model, layer_num=3):
#     def get_hidden_x_inner(model, layer_num=layer_num):
#         return K.function([model.layers[0].input], [model.layers[layer_num].output])
#     return get_hidden_x_inner(model, layer_num=layer_num)([x])[0]


# def get_decoder_output(x1, x2, model, layer_num=2):
#     temp_layer = K.function([model.layers[0].input, model.layers[1].input], [
#                             model.layers[layer_num].output])
#     return temp_layer([x1, x2])[0]


# def generate_ouputs(x, initial_state=None):
#     x = np.array(x)
#     x_f = x.reshape(1, feature_size)
#     token = 'S'
#     feat_tok = 4  # Root node
#     cont = True
#     text = [token]
#     seq = [feat_tok]
#     x_sent = np.zeros((1, maxlen, vocab_size_sent), dtype=np.bool)
#     x_seq = np.zeros((1, maxlen_seq, vocab_size_feat), dtype=np.bool)
#     x_latent = get_hidden_x(x_f, model=label_model)
#     x_latent = x_latent.reshape(1, latent_dim)
#     x_sent[0, 0, char_indices[token]] = 1
#     x_seq[0, 0, feat_tok] = 1
#     pred = label_model.predict(x_f)
#     label = [np.argmax(pred[0])]
#     index = 1
#     while cont & (index < maxlen):
#         pred = path_model.predict([x_latent, x_sent])
#         char_index = np.argmax(pred[0])
#         x_sent[0, index, char_index] = 1
#         next_char = indices_char[char_index]
#         text.append(next_char)
#         index += 1
#         if next_char == 'E':
#             cont = False

#     sent_decoder = get_decoder_output(x_latent, x_sent, path_model)
#     index = 1
#     while (index < len(text)-1):
#         pred_feat = features_model.predict([x_latent, sent_decoder, x_seq])
#         pred_val = np.argmax(pred_feat[0])
#         x_seq[0, index, pred_val] = 1
#         next_val = pred_val
#         seq.append(next_val)
#         index += 1
#     outputs = []
#     for i, val in enumerate(text):
#         if val != 'S' and val != 'E':
#             outputs.append((val, seq[i]))

#     return outputs

# from ast import literal_eval
# df = pd.read_csv('pes_dataset.csv')
# initial_state = []
# for i, val in df.iterrows():
#     a = literal_eval(val[0])
#     # print(type(a))
#     x = np.array(a)
#     # print(x)
#     x = x.reshape(1, 4)
#     initial_state.append(get_hidden_x(x, model=label_model))
# initial_state = np.array(initial_state)
# initial_state = initial_state[:,0,:]
# # print(test[:,0,:])
# # print(test[:,0,:].shape)

# initial_state_df = pd.DataFrame(initial_state)
# initial_state_df[0] = initial_state_df[[0,1,2,3,4]].values.tolist()
# initial_state_df.drop([1,2,3,4], axis=1, inplace=True)
# for i, val in initial_state_df.iterrows():
#     for j, string in enumerate(val[0]):
#         initial_state_df[0][i][j] = np.around(float(string),2)
# print(initial_state_df.head())
# initial_state_df.to_csv('initial_states.csv')

# out = generate_ouputs([0.4, 0.5, 0.2, 0.1])
# print(out)
# Return path and states as outputs

app = FlaskAPI(__name__)


notes = {
    0: 'do the shopping',
    1: 'build the codez',
    2: 'paint the door',
}

def note_repr(key):
    return {
        'url': request.host_url.rstrip('/') + url_for('notes_detail', key=key),
        'text': notes[key]
    }


@app.route("/", methods=['GET', 'POST'])
def notes_list():
    """
    List or create notes.
    """
    if request.method == 'POST':
        note = str(request.data.get('text', ''))
        idx = max(notes.keys()) + 1
        notes[idx] = note
        return note_repr(idx), status.HTTP_201_CREATED

    # request.method == 'GET'
    return [note_repr(idx) for idx in sorted(notes.keys())]


@app.route("/<int:key>/", methods=['GET', 'PUT', 'DELETE'])
def notes_detail(key):
    """
    Retrieve, update or delete note instances.
    """
    if request.method == 'PUT':
        note = str(request.data.get('text', ''))
        notes[key] = note
        return note_repr(key)

    elif request.method == 'DELETE':
        notes.pop(key, None)
        return '', status.HTTP_204_NO_CONTENT

    # request.method == 'GET'
    if key not in notes:
        raise exceptions.NotFound()
    return note_repr(key)


if __name__ == "__main__":
    app.run(debug=True)