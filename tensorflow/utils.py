import random

def generate_color_palette(labels):
    random.seed(42)
    palette = {label: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for label in labels}
    palette["unknown"] = (255, 255, 255)
    return palette
