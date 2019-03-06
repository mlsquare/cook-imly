import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from imly import dope
import copy
import json
import itertools
import matplotlib.pyplot as plt
from utils.correlations import concordance_correlation_coefficient as ccc

model_mappings = {
    'linear_regression': 'LinearRegression',
    'logistic_regression': 'LogisticRegression',
    'linear_discrimant_analysis': 'LinearDiscriminantAnalysis'
}

classification_models = ['logistic_regression', 'linear_discrimant_analysis']


def load_sheet():
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        '../data/client_secret.json', scope)

    gc = gspread.authorize(credentials)

    sh = gc.open('Dataset details')

    worksheet = sh.get_worksheet(0)

    return worksheet


def get_dataset_info(dataset_name):
    worksheet = load_sheet()

    dataset_list = worksheet.col_values(worksheet.find("name").col)
    url_list = worksheet.col_values(worksheet.find("link").col)
    # params_list = worksheet.row_values(worksheet.find("use_bias").row)
    # params_list = params_list[params_list.index('use_bias'):]
    # params_dict = {x:i+7 for i,x in enumerate(params_list)}
    activation_col_nb = worksheet.find("algorithm").col
    activation_list = worksheet.col_values(activation_col_nb)

    # TODO
    # Add error case. When no match is found.
    for dataset in dataset_list:
        if dataset_name == dataset:
            row_nb = dataset_list.index(dataset)
            data_url = url_list[row_nb]

    n_col_nb = worksheet.find("n").col
    p_col_nb = worksheet.find("p").col
    c_col_nb = worksheet.find("class_distribution").col
    activation_function = activation_list[row_nb]
    data_url = url_list[row_nb]

    dataset_info = {
        "url": data_url,
        "name": dataset_name,
        # 'params_dict': params_dict,
        'activation_function': activation_function,
        'n_col': n_col_nb,
        'p_col': p_col_nb,
        'c_col': c_col_nb,
        'row_nb': row_nb
    }
    return dataset_info


def write_to_mastersheet(data, X, Y, exp_results):

    worksheet = load_sheet()
    # params_dict = data['params_dict']
    scikit_params = data['scikit_params']
    keras_params = data['keras_params']
    row_nb = data['row_nb']
    n_col_nb = data['n_col']
    p_col_nb = data['p_col']
    c_col_nb = data['c_col']
    n = X.shape[0]
    p = X.shape[1]
    if data['activation_function'] in classification_models:
        unique, count = np.unique(Y, return_counts=True)
        class1 = count[0]/X.shape[0]*100
        class_distribution = round(class1, 2)
    else:
        class_distribution = 'NA'

    col_nb = worksheet.find('scikit_json').col
    worksheet.update_cell(row_nb+1, col_nb, scikit_params)

    col_nb = worksheet.find('keras_json').col
    worksheet.update_cell(row_nb+1, col_nb, keras_params)

    worksheet.update_cell(row_nb+1, n_col_nb, n)
    worksheet.update_cell(row_nb+1, p_col_nb, p)
    worksheet.update_cell(row_nb+1, c_col_nb, class_distribution)
    worksheet.update_cell(row_nb+1, worksheet.find("keras_acc").col,
                          exp_results['keras'])
    worksheet.update_cell(row_nb+1, worksheet.find("scikit_acc").col,
                          exp_results['scikit'])
    worksheet.update_cell(row_nb+1, worksheet.find("kfold").col,
                          exp_results['kfold'])
    worksheet.update_cell(row_nb+1, worksheet.find("plots").col,
                          exp_results['fig_url'])
    worksheet.update_cell(row_nb+1, worksheet.find("correlation").col,
                          exp_results['correlation'])
    worksheet.update_cell(row_nb+1, worksheet.find("type").col, data['type'])


def run_imly(dataset_info, model_name, X, Y, test_size, **kwargs):
    # TODO
    # Remove model_name from arguments. This data is available
    # in dataset_info['activation_fn']
    kwargs.setdefault('return_exp_results', False)
    kwargs.setdefault('space', False)
    correlation = 'NA'
    fig_url = 'NA'
    kwargs.setdefault('params', {})
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, stratify=Y,
        test_size=test_size, random_state=0)

    for key, value in model_mappings.items():
        if key == model_name:
            name = value

    module = __import__('sklearn.linear_model', fromlist=[name])
    # module = __import__('sklearn.discriminant_analysis', fromlist=[name]) # Find a fix!
    imported_module = getattr(module, name)
    model = imported_module
    model_instance = model()
    base_model = copy.deepcopy(model_instance)
    primal_model = copy.deepcopy(model_instance)

    # Primal
    primal_model.fit(x_train, y_train.values.ravel()) # Why use '.values.ravel()'? -- 
    y_pred = primal_model.predict(x_test)
    if (primal_model.__class__.__name__ == 'LogisticRegression') or \
       (primal_model.__class__.__name__ == 'LinearDiscriminantAnalysis'):
        primal_score = primal_model.score(x_test, y_test)
    else:
        # primal_score = primal_model.score(x_test, y_test)
        primal_score = mean_squared_error(y_test, y_pred)
    primal_params = primal_model.get_params(deep=True)

    # Keras
    # x_train = x_train.values  # Talos accepts only numpy arrays
    m = dope(base_model)
    if kwargs['params'] == {} and kwargs['space'] == False:
        # m.fit(x_train, y_train.values.ravel())
        m.fit(x_train, y_train)
    elif kwargs['space']:
        # m.fit(x_train, y_train.values.ravel(), space=kwargs['space'])
        m.fit(x_train, y_train, space=kwargs['space'])
    else:
        # m.fit(x_train, y_train.values.ravel(), params=kwargs['params'])
        m.fit(x_train, y_train, params=kwargs['params'])

    keras_score = m.score(x_test, y_test)

    # Create plot and write to s3 bucket #
    # TODO
    # Add 'class_name' mapping
    sklearn_pred = y_pred
    # keras_pred = m.predict_classes(x_test)
    # if(primal_model.__class__.__name__ == 'LinearRegression'):
    #     keras_pred = m.predict(x_test)
    #     fig = plot_correlation(sklearn_pred, keras_pred)
    #     fig_name, fig_path = get_fig_details(dataset_info)
    #     fig.savefig(fig_path, bbox_inches='tight')
    #     fig_url = write_plot_to_s3(fig_path, fig_name)
    #     correlation = ccc(sklearn_pred, keras_pred)

    # elif(primal_model.__class__.__name__ == 'LogisticRegression'):
    #     keras_pred = m.predict(x_test)
    #     cnf_matrix = confusion_matrix(sklearn_pred, keras_pred)
    #     class_names = np.unique(sklearn_pred)
    #     fig = plot_confusion_matrix(cnf_matrix, classes=class_names)
    #     fig_name, fig_path = get_fig_details(dataset_info)
    #     fig.savefig(fig_path, bbox_inches='tight')
    #     fig_url = write_plot_to_s3(fig_path, fig_name)
    # else:
    #     fig_url = 'NA'

    # Prepare Keras configuration #
    keras_params = m.__dict__['model'].get_config()
    keras_params = keras_params['layers'][0]['config']
    keras_params['kernel_initializer'] = keras_params['kernel_initializer']['class_name']
    keras_params['bias_initializer'] = keras_params['bias_initializer']['class_name']

    dataset_info['scikit_params'] = json.dumps(primal_params)
    dataset_info['keras_params'] = json.dumps(keras_params)
    dataset_info['type'] = 'Binary'
    exp_results = {
        'keras': keras_score,
        'scikit': primal_score,
        'kfold': None,
        'fig_url': fig_url,
        'correlation': correlation
    }

    if kwargs['return_exp_results']:
        return exp_results
    else:
        write_to_mastersheet(dataset_info, X, Y, exp_results)


def get_fig_details(dataset_info):
    '''
    Extracts and returns the figure name and fig_path
    when provided with the dataset info.
    fig_name = <dataset_name>_<algorithm_name>
    '''
    fig_name = ('_').join([dataset_info['name'], dataset_info['activation_function']]) + '.pdf'
    fig_path = '../data/' + fig_name
    return fig_name, fig_path


def plot_correlation(sklearn_pred, keras_pred):
    '''
    Creates and returns correlation plot
    '''
    fig = plt.figure()
    plt.scatter(sklearn_pred, keras_pred)
    plt.title('Correlation plot')
    plt.xlabel('Scikit predictions')
    plt.ylabel('Keras predictions')
    return fig


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    # np.set_printoptions(precision=2)
    return fig


def write_plot_to_s3(fig_path, fig_name):

    '''
     Writes the plot to s3 bucket from mentioned dir path and
     returns the s3 url
    '''

    import boto
    import sys
    from boto.s3.key import Key
    # from boto.s3.key import Key
    bucket_name = 'mlsquare-pdf'
    credentials_json = json.load(open('../data/aws_credentials.json'))
    AWS_ACCESS_KEY_ID = credentials_json['AWS_ACCESS_KEY_ID']
    AWS_SECRET_ACCESS_KEY = credentials_json['AWS_SECRET_ACCESS_KEY']
    REGION_HOST = 's3.ap-south-1.amazonaws.com'

    # bucket_name = AWS_ACCESS_KEY_ID.lower() + '-dump'
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID,
                           AWS_SECRET_ACCESS_KEY, host=REGION_HOST)
    bucket = conn.get_bucket('mlsquare-pdf', validate=False)

    # bucket = conn.create_bucket(bucket_name,
    #     location=boto.s3.connection.Location.DEFAULT)

    print('Uploading %s to Amazon S3 bucket %s' % (fig_path, bucket_name))

    def percent_cb(complete, total):
        sys.stdout.write('.')
        sys.stdout.flush()

    k = Key(bucket)
    k.key = fig_name
    k.set_contents_from_filename(fig_path,
                                 cb=percent_cb, num_cb=10)  # upload file
    url = k.generate_url(expires_in=0, query_auth=False)
    return url

# TODO
# Add options in the sheet for test case runs.
