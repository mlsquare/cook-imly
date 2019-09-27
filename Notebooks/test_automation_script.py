import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def load_sheet():
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
    '../data/client_secret.json', scope)

    gc = gspread.authorize(credentials)

    sh = gc.open('Dataset details')

    worksheet = sh.get_worksheet(0)

    return worksheet


def get_url(dataset_name):
    worksheet = load_sheet()

    dataset_list = worksheet.col_values(worksheet.find("Name").col)
    url_list = worksheet.col_values(worksheet.find("Link").col)
    params_list = worksheet.row_values(worksheet.find("use_bias").row)
    params_list = params_list[params_list.index('use_bias'):]
    params_dict = {x:i+7 for i,x in enumerate(params_list)}
    activation_col_nb = worksheet.find("Algorithm").col
    activation_list = worksheet.col_values(activation_col_nb)

    for dataset in dataset_list:
        if dataset_name == dataset:
            row_nb = dataset_list.index(dataset)
            data_url = url_list[row_nb]

    n_col_nb = worksheet.find("N").col
    p_col_nb = worksheet.find("P").col
    c_col_nb = worksheet.find("Class distribution").col
    activation_function = activation_list[row_nb]
    data_url = url_list[row_nb]

    dataset_info = {
        "url": data_url,
        'params_dict': params_dict,
        'activation_function': activation_function,
        'n_col': n_col_nb,
        'p_col': p_col_nb,
        'c_col': c_col_nb,
        'row_nb': row_nb
    }
    return dataset_info


def get_keras_params(X,Y,predictions,data_info,config):
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    from keras.utils import np_utils

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)
    np.random.seed(7) # Why?

    # Defining model #
    if data_info['activation_function'] == "Logistic regression":
        activation = "sigmoid"
    input_dim = X.shape[1]
    epoch = config['epoch']
    batch_size = config['batch_size']
    verbose = config['verbose']


    model = Sequential()
    model.add(Dense(1,input_dim=input_dim,activation=activation))

    # Compile the model #

    model.compile(loss=config['model_info']['loss'], optimizer=config['model_info']['optimizer'],
     metrics=config['model_info']['metrics'])

    # Fit the model #

    model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size,verbose=verbose)

    # Evaluate the model #

    scores = model.evaluate(x_test, predictions)
    scores2 = model.evaluate(x_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores2[1]*100))

    # Prepare Keras configuration #
    keras_params = model.get_config()
    keras_params = keras_params['layers'][0]['config']
    keras_params['kernel_initializer'] = keras_params['kernel_initializer']['class_name']
    keras_params['bias_initializer'] = keras_params['bias_initializer']['class_name']

    return str(round(scores[1]*100,2)) + " %", keras_params


def get_scikit_params(X,Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)
    # Import and create an instance of your model(Logistic regression)

    logisticRegr = LogisticRegression()

    # Train your model using the training dataset

    logisticRegr.fit(x_train,y_train)

    # Predict the output 

    predictions = logisticRegr.predict(x_test)

    score = logisticRegr.score(x_test,y_test)
    print(score)
    scikit_params = logisticRegr.get_params(deep=True)
    return str(round(score*100,2)) + " %", scikit_params, predictions


def write_to_mastersheet(data,X,Y,accuracy_values):

    worksheet = load_sheet()
    params_dict = data['params_dict']
    scikit_params = data['scikit_params']
    keras_params = data['keras_params']
    row_nb = data['row_nb']
    n_col_nb = data['n_col']
    p_col_nb = data['p_col']
    c_col_nb = data['c_col']
    n = X.shape[0]
    p = X.shape[1]
    unique,count = np.unique(Y,return_counts=True)
    class1=count[0]/X.shape[0]*100
    class2=count[1]/X.shape[0]*100
    class_distribution = str(int(round(class1))) + " : " + str(int(round(class2)))

    for param,col_nb in params_dict.items():
        for s_param,value in scikit_params.items():
            if param == s_param:
                if value == None:
                    value = 'None'
                col_nb = worksheet.find(s_param).col
                worksheet.update_cell(row_nb+1, col_nb, value)
            

    for param,col_nb in params_dict.items():
        for k_param,value in keras_params.items():
            if param == k_param:
                if value == None:
                    value = 'None'
                col_nb = worksheet.find(k_param).col
                worksheet.update_cell(row_nb+1, col_nb, value)

    worksheet.update_cell(row_nb+1, n_col_nb, n)
    worksheet.update_cell(row_nb+1, p_col_nb, p)
    worksheet.update_cell(row_nb+1, c_col_nb, class_distribution)
    worksheet.update_cell(row_nb+1, worksheet.find("Keras acc").col, accuracy_values['keras'])
    worksheet.update_cell(row_nb+1, worksheet.find("Scikit acc").col, accuracy_values['scikit'])
    worksheet.update_cell(row_nb+1, worksheet.find("Kfold").col, accuracy_values['kfold'])
    worksheet.update_cell(row_nb+1, worksheet.find("Type").col, data['type'])



def get_kfold(X,Y,config):
    from keras.models import Sequential
    from keras.layers import Dense
    from sklearn.model_selection import StratifiedKFold

    seed = 7
    np.random.seed(seed)

    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=config['splits'], shuffle=True, random_state=seed)
    cvscores = []

    for train, test in kfold.split(X, Y):
        # create model
        model = Sequential()
        model.add(Dense(1,input_dim=X.shape[1],activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        model.fit(X.iloc[train], Y.iloc[train], epochs=config['epoch'], batch_size=config['batch_size'], verbose=0)
        # evaluate the model
        scores = model.evaluate(X.iloc[test], Y.iloc[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print('\n   -------------   \n')

    y_test, y_pred = get_roc(X,Y,model)
    print('\n   -------------   \n')

    target_names = ['Class 1', 'Class 2']
    (print(classification_report(y_test, y_pred, target_names=target_names)))

    return "%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))


def get_roc(X,Y,model):
    from sklearn import metrics
    import matplotlib.pyplot as plt

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)

    y_pred = model.predict(x_test)
    y_pred = (y_pred>0.5)
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)


    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # Creating the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    return y_test, y_pred
