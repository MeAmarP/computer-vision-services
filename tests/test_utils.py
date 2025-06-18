import pytest
from pytorch.utils import generate_color_palette


def test_generate_color_palette():
    labels = ['cat', 'dog']
    palette = generate_color_palette(labels)
    # Should include original labels and an 'unknown' key
    assert set(palette.keys()) == set(labels + ['unknown'])
    for color in palette.values():
        assert isinstance(color, tuple) and len(color) == 3
        assert all(0 <= c <= 255 for c in color)
