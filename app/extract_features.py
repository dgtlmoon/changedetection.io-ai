tag_name_map = {'span': 0, 'div': 1, 'a': 2, 'p': 3, 'li': 4, 'h2': 5, 'h3': 6, 'h4': 7, 'h1': 7}
from price_parser import Price
import re


def contains_mostly_integers(s: str, threshold: float = 0.4) -> bool:
    if not s:  # handle empty string case
        return False
    if '70.000' in s:
        x=1

    digit_count = sum(c.isdigit() for c in s)
    total_length = len(s)
    end_threshold = (digit_count / total_length)

    return end_threshold >= threshold

def extract_features(element, browser_width=1280, browser_height=800):
    normalized_top = element['top']
    normalized_left = element['left'] / browser_width
    normalized_width = element['width'] / browser_width
    normalized_height = element['height']
    font_size = int(float(element['fontSize'].replace("px", ""))) if element.get('fontSize') else 0
    font_weight = int(element['fontWeight']) if element.get('fontWeight') else 0

    above_fold = 1 if element['top'] <= browser_height else 0

    contains_price_text = 0



    # Some sites, like ikea CZ only have a few integers as the price
    # 'contains_mostly_integers' for skipping strings like "NICE WIDGET SIZE 500"
    if contains_mostly_integers(element.get('text')) and re.findall(r'\d{2,}', element.get('text', '')):

        price_info = Price.fromstring(element.get('text', ''))

        if price_info.amount:
            contains_price_text = 1

    if above_fold:
        normalized_top = normalized_top/browser_height

    return [
        int(element.get('hasDigitCurrency', False)),
        above_fold,
        contains_price_text,
        normalized_top,
        normalized_left,
        element['textWidth'],
        element['textHeight'],
        element['t_r'],
        element['t_g'],
        element['t_b'],
        font_size,
        font_weight
    ]
