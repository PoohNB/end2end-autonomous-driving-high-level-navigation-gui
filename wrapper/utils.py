from wrapper import observer
from wrapper import action_wrapper 

# not the same as train
def init_component(config):
    """
    Initialize components based on the configuration.

    Args:
        config (dict): Configuration dictionary with observer and action wrapper details.

    Returns:
        tuple: Initialized observer and action wrapper objects.
    """
    try:
        observer_class = getattr(observer, config['observer_config']['name'])
        observer_ob = observer_class(**config['observer_config']['config'])
    except (AttributeError, TypeError) as e:
        raise ValueError(f"Error initializing observer: {e}")

    try:
        action_wrapper_class = getattr(action_wrapper, config['actionwrapper']['name'])
        config['actionwrapper']['config']['activate_filter_8bit']=False
        print(config['actionwrapper']['config'])
        action_wrapper_ob = action_wrapper_class(**config['actionwrapper']['config'])
    except (AttributeError, TypeError) as e:
        raise ValueError(f"Error initializing action wrapper: {e}")

    return observer_ob, action_wrapper_ob

