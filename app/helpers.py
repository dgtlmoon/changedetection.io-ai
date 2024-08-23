def get_filtered_elements(data):
    # Skip elements that are of no use, price information is always of a reasonable size
    elements = []
    for idx, element in enumerate(data):
        if not all(key in element for key in ['textWidth', 'textHeight', 't_r']):
            continue
        if element['textWidth'] < 20 or element['textWidth'] > 550:
            continue
        if element['top'] > 1600:
            continue

        element.update({'original_idx': idx})
        elements.append(element)

    return elements
