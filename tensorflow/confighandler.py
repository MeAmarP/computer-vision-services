import yaml

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Loads the YAML configuration file and returns it as a dictionary.
    
    :param config_path: Path to the config.yaml file.
    :return: A dictionary containing configurations.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

