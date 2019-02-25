from utils.model_mapping import get_model_design
from architectures.sklearn.model import create_model
import os
from ray.tune.suggest import HyperOptSearch

# Initialize ray
import ray
from ray import tune

ray.init(ignore_reinit_error=True)


def get_best_model(x_train, y_train, **kwargs):

    y_pred = kwargs['primal_data']['y_pred']
    model_name = kwargs['primal_data']['model_name']
    fn_name, param_name = get_model_design(model_name)

    mapping_instance = create_model(fn_name=fn_name, param_name=param_name)

    def train_model(config, reporter):
        '''
        This function is used by Tune to train the model with each iteration variations.

        Args:
            config(dict): A dictionary with the search params passed by Tune.
            Similar to the JSON we already have.
            reporter: A function used by Tune to keep a track of the metric by
            which the iterations should be optimized.
        '''

        model = mapping_instance.__call__(x_train=x_train, params=config)
        model.fit(x_train, y_pred)
        last_checkpoint = "weights_tune_{}.h5".format(config)
        model.save_weights(last_checkpoint)
        accuracy = model.evaluate(x_train, y_pred)[1]
        reporter(mean_accuracy=accuracy, checkpoint=last_checkpoint)


    # Define experiment configuration
    configuration = tune.Experiment("experiment_name",
                                    run=train_model,
                                    resources_per_trial={"cpu": 4},
                                    stop={"mean_accuracy": 95},
                                    config=kwargs['params'])

    # This validation is to check if the user has opted for hyperopt search method
    if kwargs['space']:
        print('hyperopt choosen-------')
        space = kwargs['space']
        hyperopt_search = HyperOptSearch(space, reward_attr="mean_accuracy")
        # TODO
        # Should this wrapper be avoided(instead the user passes the HyperOptSearch).
        # Add other args for hyperopt search.
        # Add the remaining search_algos if necessary.
        trials = tune.run_experiments(configuration,
                                      search_alg=hyperopt_search, verbose=2)

    else:
        trials = tune.run_experiments(configuration, verbose=2)

    metric = "mean_accuracy"

    """Restore a model from the best trial."""
    sorted_trials = get_sorted_trials(trials, metric)
    for best_trial in sorted_trials:
        try:
            print("Creating model...")
            best_model = mapping_instance.__call__(x_train=x_train, params=best_trial.config)  # TODO Pass config as argument
            # best_model = make_model(None)
            weights = os.path.join(best_trial.logdir, best_trial.last_result["checkpoint"])
            print("Loading from", weights)
            best_model.load_weights(weights)  # TODO Validate this loaded model.
            break
        except Exception as e:
            print(e)
            print("Loading failed. Trying next model")

    return best_model


# Utils from Tune tutorials(Not a part of the Tune package) #

def get_sorted_trials(trial_list, metric):
    return sorted(trial_list, key=lambda trial: trial.last_result.get(metric, 0), reverse=True)

# TODO
# Generalize metric choice.
# Add compatibility for linReg and LDA.
# Validate the loaded model.
