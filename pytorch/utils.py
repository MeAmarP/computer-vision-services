import random
import logging
import logging.config
import yaml

logger = logging.getLogger(__name__)

def setup_logging(config_path: str = "logger_config.yaml"):
    """Initialize logging using a YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)

def generate_color_palette(labels):
    """Generate a deterministic color palette for given labels."""
    logger.debug("Generating color palette for labels: %s", labels)
    random.seed(42)
    palette = {
        label: (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        for label in labels
    }
    palette["unknown"] = (255, 255, 255)
    logger.debug("Generated palette: %s", palette)
    return palette
