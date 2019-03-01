import json


def get_model_design(name):
    """Gets function name and param name to create the base model

    # Arguments
        name: Name of the primal model

    # Returns:
        The function name and param name mapped to the primal model
    """
    mapping_data = json.load(open('../imly/utils/model_params_mapping.json'))

    try:
        fn_name = mapping_data[name]['model']
        param_name = mapping_data[name]['param']
    except KeyError:
        print('Invalid model name passed to mapping_data')

    return fn_name, param_name
