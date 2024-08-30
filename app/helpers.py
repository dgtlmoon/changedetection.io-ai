def get_filtered_elements(data):
    # Skip elements that are of no use, price information is always of a reasonable size
    elements = []
    import random
    for idx, element in enumerate(data):
        if not all(key in element for key in ['textWidth', 'textHeight', 't_r']):
            # Probably broken record.
            continue
        if not element.get('hasDigit'):
            # Don't bother at all if no number/digit was found.
            continue
        if element['textWidth'] < 10 or element['textWidth'] > 550 or element.get('textLength', 0) > 25:
            # Way too big
            continue
        if element['top'] > 1600 or element['top'] <= 100:
            # Likely off page or in some top menu or no use anyway
            continue

        element.update({'original_idx': idx})
        elements.append(element)

    random.shuffle(elements)
    return elements
