tag_name_map = {'span': 0, 'div': 1, 'a': 2, 'p': 3, 'li': 4, 'h2': 5, 'h3': 6, 'h4': 7, 'h1': 7, 'bdi': 8}

import re

def extract_features(element, browser_width=1280, browser_height=800):

    cleaned_font_size = re.sub(r'[^\d.]', '', element['fontSize']).replace(',','.')
    font_size = int(float(cleaned_font_size)) if element.get('fontSize') else 0
    font_weight = int(element['fontWeight']) if element.get('fontWeight') else 0

    return [
        font_weight,
        font_size,
        element.get('textWidth', 0),
        element.get('textHeight', 0),
        int(element.get('textLength', 0)),
        int(element.get('hasDigit', False)),
        int(element.get('hasDigitCurrency', False)),
        element.get('top', 0),
        element.get('left', 0),
        element.get('width', 0),
        element.get('height', 0),
        element.get('t_r', 0),
        element.get('t_g', 0),
        element.get('t_b', 0),
        int(element.get('isClickable', 0))
    ]
