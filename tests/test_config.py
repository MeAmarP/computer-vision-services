import yaml
from pytorch.main import load_config

def test_load_config(tmp_path):
    data = {
        'model_name': 'm',
        'model_weights_class': 'c',
        'model_weights': 'w'
    }
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(data))
    assert load_config(str(path)) == data
