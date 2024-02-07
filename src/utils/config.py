import sys
from yaml import safe_load
from loguru import logger


def load_yaml(path: str) -> dict:
    """ Load a YAML file from the given path and return its content as a dictionary """

    logger.info(f'Reading YAML file {path}.')
    return safe_load(open(path, mode='r'))


def load_config() -> dict:
    """ Load and check a configuration from a YAML file and return it as a dictionary """

    cfg = load_yaml('configs/config.yaml')
    check_config(cfg)
    
    return cfg


def basic_errors(variable: object, variable_name: str, variable_type: object) -> None:
    """ Check if the variable exists and if it has a good data type """
    
    if variable is None:
        raise KeyError(f'"{variable_name}" is missing in a configuration file.')

    if not isinstance(variable, variable_type):
        raise TypeError(f'Invalid "{variable_name}" type {type(variable).__name__}, it must be a "{variable_type}".')


@logger.catch(onerror=lambda _: sys.exit(1))
def check_config(cfg: dict) -> None:
    """ Check configuration file for errors """

    # =============================================================================================
    # GENERAL SETTINGS
    # =============================================================================================

    ### CORE SYMMETRY ###
    core_symmetry = cfg.get('core_symmetry')
    basic_errors(core_symmetry, 'core_symmetry', str)

    if core_symmetry not in ['1/4', '1/8']:
        raise ValueError(f'Invalid "core_symmetry" value: "{core_symmetry}", "1/4" or "1/8" values are possible.')


    ### TRANSFORM OUTPUT ###
    transform_output = cfg.get('log_transform_output')
    basic_errors(transform_output, 'log_transform_output', bool)


    ### ONE HOT ENCODING ###
    one_hot_encoding = cfg.get('one_hot_encoding')
    basic_errors(one_hot_encoding, 'one_hot_encoding', bool)
        
    
    ### TRAIN SPLIT ###
    train_split = cfg.get('train_split')
    
    basic_errors(train_split, 'train_split', float)
        
    if (train_split < 0) | (train_split > 1):
        raise ValueError(f'Invalid "train_split" value: {train_split}, it must be a number between 0 and 1.')
    
    
    ### SAVE OUTPUT ###
    save_output = cfg.get('save_output')
    basic_errors(save_output, 'save_output', bool)
    
    
    ### OUTPUT COLUMNS ###
    output_columns = cfg.get('output_columns')
    basic_errors(output_columns, 'output_columns', list)
    
    
    # =============================================================================================
    # INPUT FILES SETTINGS
    # =============================================================================================

    ### INPUT OUTPUT FILE ###
    input_output_file_details = cfg.get('input_output_file_details')
    basic_errors(input_output_file_details, 'input_output_file_details', dict)
    
    file_path = input_output_file_details.get('file_path')
    basic_errors(file_path, 'file_path', str)
    

    ### MONOCORE FILE ###
    monocore_file_details = cfg.get('monocore_file_details')
    basic_errors(monocore_file_details, 'monocore_file_details', dict)
    
    file_path = monocore_file_details.get('file_path')
    basic_errors(file_path, 'file_path', str)
    
    
    # =============================================================================================
    # MONOCORE SETTINGS
    # =============================================================================================
    
    ### USE MONOCORES ###
    use_monocores = cfg.get('use_monocores')
    basic_errors(use_monocores, 'use_monocores', bool)
    
    
    ### CORE NUMBER COLUMN NAME ###
    core_number_column_name = cfg.get('core_number_column_name')
    basic_errors(core_number_column_name, 'core_number_column_name', str)
    
    
    ### TRANSFORM COLUMN NAMES ###
    transform_column_names = cfg.get('transform_column_names')
    basic_errors(transform_column_names, 'transform_column_names', list)
    
    
    # =============================================================================================
    # NEURAL NETWORK SETTINGS
    # =============================================================================================
    
    ### LAYERS ###
    layers = cfg.get('layers')
    basic_errors(layers, 'layers', dict)
    
    layer_output = layers.get('layer_output')
    basic_errors(layer_output, 'layer_output', dict)
    
    if layer_output.get('neurons') is not None:
        logger.warning('Number of neurons in the "layer_output" is obsolete and it will not be used.')
        
    activation = layer_output.get('activation')
    basic_errors(activation, 'activation', str)
    
    layer_names = list(layers.keys())
    layer_names.sort()
    if not all(layer_names[i] == f'layer{i+1}' for i in range(len(layer_names) - 1)):
        raise IndexError('The layer names in the neural network are not numbered sequentially.')
        
    for layer_name, layer in layers.items():
        if layer_name == 'layer_output':
            continue
        
        neurons = layer.get('neurons')
        basic_errors(neurons, 'neurons', int)
        
        if neurons < 0:
            raise ValueError(f'Invalid "neurons" value in {layer_name}: {neurons}, it must be greater than 0.')

        activation = layer.get('activation')
        basic_errors(activation, 'activation', str)
        
        
    ### NORMALIZE ###
    normalize = cfg.get('normalize')
    basic_errors(normalize, 'normalize', bool)

    
    ### LOSS FUNCTION ###
    loss_function = cfg.get('loss_function')
    basic_errors(loss_function, 'loss_function', str)
    
    
    ### LEARNING RATE ###
    learning_rate = cfg.get('learning_rate')
    basic_errors(learning_rate, 'learning_rate', float)
    
    
    ### EPOCS ###
    epochs = cfg.get('epochs')
    basic_errors(epochs, 'epochs', int)

    if epochs < 0:
        raise ValueError(f'Invalid "epochs" value: {epochs}, it must be an integer greater than 0.')


    # =============================================================================================
    # EVALUATION OF THE MODEL
    # =============================================================================================

    ### METRICS ###
    metrics = cfg.get('metrics')
    basic_errors(metrics, 'metrics', list)
    
    
    # =============================================================================================
    # SAVING THE OUTPUT
    # =============================================================================================
    
    ### OUTPUT DIRECTORY ###
    output_directory = cfg.get('output_directory')
    basic_errors(output_directory, 'output_directory', str)
    
    
    ### ADD TIMESTAMP ###
    add_timestamp = cfg.get('add_timestamp')
    basic_errors(add_timestamp, 'add_timestamp', bool)
    
    
    ### ADD PREFFIX NUMBER ###
    add_preffix_number = cfg.get('add_preffix_number')
    basic_errors(add_preffix_number, 'add_preffix_number', bool)
    
    
    ### CREATE CONFIG FILE ###
    create_config_file = cfg.get('create_config_file')
    basic_errors(create_config_file, 'create_config_file', bool)
    
    
    ### CREATE EVALUATION FILE ###
    create_evaluation_file = cfg.get('create_evaluation_file')
    basic_errors(create_evaluation_file, 'create_evaluation_file', bool)
    
    
    ### CREATE LOSS PLOT ###
    create_loss_plot = cfg.get('create_loss_plot')
    basic_errors(create_loss_plot, 'create_loss_plot', bool)
    
    
    ### CREATE ACCURACY PLOT ###
    create_accuracy_plot = cfg.get('create_accuracy_plot')
    basic_errors(create_accuracy_plot, 'create_accuracy_plot', bool)


CFG = load_config()