__author__ = 'tchaton'

'''
This code has been taken from https://github.com/VahidooX/DeepLDA.
It is written in python using Theano Library.
import theano.tensor as T
import theano
import numpy as np
from theano.compile.ops import as_op
@as_op(itypes=[theano.tensor.ivector],
       otypes=[theano.tensor.ivector])
def numpy_unique(a):
    return np.unique(a)
def lda_loss(n_components, margin):
    """
    The main loss function (inner_lda_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    """
    def inner_lda_objective(y_true, y_pred):
        """
        It is the loss function of LDA as introduced in the original paper.
        It is adopted from the the original implementation in the following link:
        https://github.com/CPJKU/deep_lda
        Note: it is implemented by Theano tensor operations, and does not work on Tensorflow backend
        """
        r = 1e-4
        # init groups
        yt = T.cast(y_true.flatten(), "int32")
        groups = numpy_unique(yt)
        def compute_cov(group, Xt, yt):
            Xgt = Xt[T.eq(yt, group).nonzero()[0], :]
            Xgt_bar = Xgt - T.mean(Xgt, axis=0)
            m = T.cast(Xgt_bar.shape[0], 'float32')
            return (1.0 / (m - 1)) * T.dot(Xgt_bar.T, Xgt_bar)
        # scan over groups
        covs_t, updates = theano.scan(fn=compute_cov, outputs_info=None,
                                      sequences=[groups], non_sequences=[y_pred, yt])
        # compute average covariance matrix (within scatter)
        Sw_t = T.mean(covs_t, axis=0)
        # compute total scatter
        Xt_bar = y_pred - T.mean(y_pred, axis=0)
        m = T.cast(Xt_bar.shape[0], 'float32')
        St_t = (1.0 / (m - 1)) * T.dot(Xt_bar.T, Xt_bar)
        # compute between scatter
        Sb_t = St_t - Sw_t
        # cope for numerical instability (regularize)
        Sw_t += T.identity_like(Sw_t) * r
        # return T.cast(T.neq(yt[0], -1), 'float32')*T.nlinalg.trace(T.dot(T.nlinalg.matrix_inverse(St_t), Sb_t))
        # compute eigenvalues
        evals_t = T.slinalg.eigvalsh(Sb_t, St_t)
        # get eigenvalues
        top_k_evals = evals_t[-n_components:]
        # maximize variance between classes
        # (k smallest eigenvalues below threshold)
        thresh = T.min(top_k_evals) + margin
        top_k_evals = top_k_evals[(top_k_evals <= thresh).nonzero()]
        costs = T.mean(top_k_evals)
        return -costs
    return inner_lda_objective
'''

import tensorflow as tf
from sklearn import svm
from sklearn.metrics import accuracy_score
from keras.utils import np_utils



def svm_classify(x_train, y_train, x_test, y_test, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(x_train, y_train.ravel())

    p = clf.predict(x_train)
    train_acc = accuracy_score(y_train, p)

    p = clf.predict(x_test)
    test_acc = accuracy_score(y_test, p)

    return [train_acc, test_acc]


def lda_loss(n_components, margin):
    """
    The main loss function (inner_lda_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    """

    def inner_lda_objective(y_true, y_pred):
        """
        It is the loss function of LDA as introduced in the original paper.
        It is adopted from the the original implementation in the following link:
        https://github.com/CPJKU/deep_lda
        Note: it is implemented by Theano tensor operations, and does not work on Tensorflow backend
        """
        r = 1e-4
        print("in lda inner loop")
        print(y_true)
        print(y_pred)

        # init groups
        # yt = tf.cast(tf.contrib.layers.flatten(y_true), tf.float32)
        # indexes = tf.argmax (y_true, axis=-1)
        locations = tf.where(tf.equal(y_true, 1))
        indices = locations[:, 1]
        y, idx = tf.unique(indices)

        print(locations)

        def fn(unique, indexes, preds):
            u_indexes = tf.where(tf.equal(unique, indexes))
            u_indexes = tf.reshape(u_indexes, (1, -1))
            X = tf.gather(preds, u_indexes)
            X_mean = X - tf.reduce_mean(X, axis=0)
            m = tf.cast(tf.shape(X_mean)[1], tf.float32)
            return (1 / (m - 1)) * tf.matmul(tf.transpose(X_mean[0]), X_mean[0])

        # scan over groups
        covs_t = tf.map_fn(lambda x: fn(x, indices, y_pred), y, dtype=tf.float32)

        # compute average covariance matrix (within scatter)
        Sw_t = tf.reduce_mean(covs_t, axis=0)

        # compute total scatter
        Xt_bar = y_pred - tf.reduce_mean(y_pred, axis=0)
        m = tf.cast(tf.shape(Xt_bar)[1], tf.float32)
        St_t = (1 / (m - 1)) * tf.matmul(tf.transpose(Xt_bar), Xt_bar)

        # compute between scatter
        dim = tf.shape(y)[0]
        Sb_t = St_t - Sw_t

        # cope for numerical instability (regularize)

        Sw_t += tf.eye(dim) * r

        ''' START : COMPLICATED PART WHERE TENSORFLOW HAS TROUBLE'''

        #cho = tf.eye(dim)
        # look at page 383
        # http://perso.ens-lyon.fr/patrick.flandrin/LedoitWolf_JMA2004.pdf
        
        r = 100
        cho = tf.cholesky(St_t + tf.eye(dim) * r)
        inv_cho = tf.matrix_inverse(cho)
        evals_t = tf.linalg.eigvalsh(inv_cho * Sb_t * tf.transpose(inv_cho + tf.eye(dim) * r))  # Sb_t, St_t # SIMPLIFICATION OF THE EQP USING cholesky
        #evals_t = tf.abs(tf.linalg.eigvalsh(tf.matrix_inverse(Sw_t) * Sb_t )) # INVERSED EQUATION
        top_k_evals = evals_t[-n_components:]

        ''' END : COMPLICATED PART WHERE TENSORFLOW HAS TROUBLE'''

        # index_max = tf.argmax(top_k_evals, 0)
        # thresh_max = top_k_evals[index_max] - margin

        index_min = tf.argmin(top_k_evals, 0)
        thresh_min = top_k_evals[index_min] + margin
        # thresh = tf.contrib.distributions.percentile(top_k_evals, 33) # TRY TO TAKE A PERCENTILE INSTEAD THAN A THRESOLD
        # mask_max = top_k_evals > thresh_max
        mask_min = top_k_evals < thresh_min

        # cost_max = tf.boolean_mask(top_k_evals, mask_max)
        cost_min = tf.boolean_mask(top_k_evals, mask_min)
        cost = -tf.reduce_mean(cost_min)
        
        cost = tf.math.divide(tf.linalg.trace(Sb_t),tf.linalg.trace(Sw_t))

        return cost

    return inner_lda_objective


def lda_trace_loss(n_components, margin):
    """
    The main loss function (inner_lda_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    """

    def inner_lda_trace_objective(y_true, y_pred):
        """
        It is the loss function of LDA as introduced in the original paper.
        It is adopted from the the original implementation in the following link:
        https://github.com/CPJKU/deep_lda
        Note: it is implemented by Theano tensor operations, and does not work on Tensorflow backend
        """
        r = 1e-4
        print("in lda inner loop")
        print(y_true)
        print(y_pred)

        # init groups
        # yt = tf.cast(tf.contrib.layers.flatten(y_true), tf.float32)
        # indexes = tf.argmax (y_true, axis=-1)
        locations = tf.where(tf.equal(y_true, 1))
        indices = locations[:, 1]
        y, idx = tf.unique(indices)

        print(locations)

        def fn(unique, indexes, preds):
            u_indexes = tf.where(tf.equal(unique, indexes))
            u_indexes = tf.reshape(u_indexes, (1, -1))
            X = tf.gather(preds, u_indexes)
            X_mean = X - tf.reduce_mean(X, axis=0)
            m = tf.cast(tf.shape(X_mean)[1], tf.float32)
            return (1 / (m)) * tf.matmul(tf.transpose(X_mean[0]), X_mean[0])

        # scan over groups
        covs_t = tf.map_fn(lambda x: fn(x, indices, y_pred), y, dtype=tf.float32)

        # compute average covariance matrix (within scatter)
        Sw_t = tf.reduce_mean(covs_t, axis=0)

        # compute total scatter
        Xt_bar = y_pred - tf.reduce_mean(y_pred, axis=0)
        m = tf.cast(tf.shape(Xt_bar)[1], tf.float32)
        St_t = (1 / (m)) * tf.matmul(tf.transpose(Xt_bar), Xt_bar)

        # compute between scatter
        dim = tf.shape(y)[0]
        Sb_t = St_t - Sw_t

        
        #cost = tf.math.divide(tf.linalg.trace(Sw_t),tf.linalg.trace(Sb_t))
        cost = tf.linalg.trace(St_t)

        return cost

    return inner_lda_trace_objective

import numpy as np
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import np_utils


from keras.models import Model



def create_model(input_dim, reg_par, outdim_size):
    """
    Builds the model
    The structure of the model can get easily substituted with a more efficient and powerful network like CNN
    """
    model = Sequential()

    model.add(Dense(1024, input_shape=(input_dim,), activation='sigmoid', kernel_regularizer=l2(reg_par)))
    #model.add(Dense(1024, activation='sigmoid', kernel_regularizer=l2(reg_par)))
    #model.add(Dense(1024, activation='sigmoid', kernel_regularizer=l2(reg_par)))
    model.add(Dense(outdim_size, activation='linear', kernel_regularizer=l2(reg_par)))

    return model

if __name__ == '__main__':
    ############
    # Parameters Section

    # the path to save the final learned features
    save_to = './new_features.gz'

    # the size of the new space learned by the model (number of the new features)
    outdim_size = 10

    # the parameters for training the network
    epoch_num = 10
    batch_size = 100

    # the regularization parameter of the network
    reg_par = 1e-5

    # The margin and n_components (number of components) parameter used in the loss function
    # n_components should be at most class_size-1
    margin = 1.0
    n_components = 9
    nr_classes = 10


    # Parameter C of SVM
    C = 1e-1
    # end of parameters section
    ############

    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.reshape(x_train, (len(x_train), -1))
    x_test = np.reshape(x_test, (len(x_test), -1))


    Y_train_svm = np_utils.to_categorical(y_train, nr_classes)
    Y_test_svm = np_utils.to_categorical(y_test, nr_classes)

    print(x_train.shape, y_train.shape)
    
    # Building, training, and producing the new features by Deep LDA
    input_dim = x_train.shape[-1]

    ip_layer = Input(shape=(input_dim,))
    lda_layer = Dense(outdim_size, name='lda', activation='linear', kernel_regularizer=l2(reg_par))(ip_layer) 
    #op_layer = Dense(nr_classes,name='output' ,activation='softmax')(lda_layer)
    op_layer = Dense(nr_classes,name='output' ,activation='softmax')(ip_layer)

    model = Model(inputs=ip_layer, outputs=[lda_layer,op_layer])

    losses = {'lda':lda_trace_loss(n_components, margin),'output':'categorical_crossentropy'}
    #losses = {'lda':lda_loss(n_components, margin)}
    #losses = {'output':'categorical_crossentropy'}

    model.compile(loss=losses, optimizer='rmsprop')

    model.summary()

    model.fit([x_train], [y_train,Y_train_svm], batch_size=batch_size, epochs=epoch_num, verbose=2)

    #model.fit([x_train], [y_train,Y_train_svm], batch_size=batch_size, epochs=epoch_num, shuffle=True, validation_data=(x_test, [y_test,Y_test_svm]),
    #          verbose=2)

    
    #x_train_new = model.predict(x_train)
    #x_test_new = model.predict(x_test)

    # Training and testing of SVM with linear kernel on the new features
    #[train_acc, test_acc] = svm_classify(x_train_new, y_train, x_test_new, y_test, C=C)
    #print("Accuracy on train data is:", train_acc * 100.0)
    #print("Accuracy on test data is:", test_acc * 100.0)