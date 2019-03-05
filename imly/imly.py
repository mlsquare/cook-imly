""" Contains dope function, using which we create the core IMLY model
"""

from utils.model_mapping import get_model_design
from architectures.sklearn.model import create_model
import copy, json, re


def dope(model, **kwargs):
    """Creates the IMLY model

    # Arguments
        model: The primal model passed by the user that needs to be transpiled.
        **kwargs: Dictionary of parameters mapped to their keras params.

    # Returns
        The transpiled model.
    """
    model_name = model.__class__.__name__
    kwargs.setdefault('using', 'dnn')

    # Map model name to it's respective wrapper
    wrapper_mapping_json = json.load(open('../imly/wrappers/keras_wrapper_mapping.json'))
    for key, value in wrapper_mapping_json.items():
        for name in value:
            if model_name == name:
                wrapper_class = key

    # Extract the wrapper_class using the wrapper_name
    path = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', wrapper_class)
    module_path = re.sub('([a-z0-9])([A-Z])', r'\1_\2', path).lower()
    package_name = module_path.split('_')[0]
    wrapper_name = '_'.join(module_path.split('_')[1:3])
    module_path = 'wrappers.' + package_name + '.' + wrapper_name
    wrapper_module = __import__(module_path, fromlist=[wrapper_class])
    wrapper_class = getattr(wrapper_module, wrapper_class)

    if kwargs['using'] == 'dnn':
        # TODO
        # Rename model, since it's causing confusion at line 63
        primal = copy.deepcopy(model)

        fn_name, param_name = get_model_design(model_name)

        build_fn = create_model(fn_name, param_name)

        params_json = json.load(open('../imly/architectures/sklearn/params.json'))
        params = params_json['params'][param_name] 
        # This params can be overwritten by the user only at the wrapper's fit method.

        # Params is loaded at 'dope' level to avoid redundant code within
        # each wrapper.
        model = wrapper_class(build_fn=build_fn, primal=primal,
                              params=params)

    return model
