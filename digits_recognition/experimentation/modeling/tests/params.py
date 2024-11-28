import yaml


with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

data_params = params['data']
data_meta_params = data_params['meta']
data_meta_images_params = data_meta_params['images']
data_processed_params = data_params['processed']
image_width = data_meta_images_params['width']
image_height = data_meta_images_params['height']
image_channels = data_meta_images_params['channels']
class_count = data_meta_params['classes']['count']


def params_train():
    training_params = params['training']

    train_set_path = data_processed_params['train_set']
    val_set_path = data_processed_params['val_set']
    batch_size = training_params['batch_size']
    learning_rate = training_params['learning_rate']
    weight_decay = training_params['weight_decay']
    epochs = training_params['epochs']
    polynomial_scheduler_power = training_params['polynomial_scheduler_power']

    return (
        train_set_path,
        val_set_path,
        batch_size,
        learning_rate,
        weight_decay,
        epochs,
        polynomial_scheduler_power,
        class_count,
        image_width,
        image_height,
        image_channels
    )


def params_test():
    evaluation_params = params['evaluation']

    model_path = params['model']
    test_set_path = data_params['processed']['test_set']
    batch_size = evaluation_params['batch_size']
    random_seed = evaluation_params['random_seed']

    return (
        model_path,
        test_set_path,
        batch_size,
        random_seed,
        class_count,
        image_width,
        image_height,
        image_channels
    )
