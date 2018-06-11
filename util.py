def extract_matched_labels(labels: list, keywords: list)->list:
    """
    Extract sensor stream labels which contain all the keywords
    """
    results = list()
    for label in labels:
        matched = True
        for keyword in keywords:
            matched = matched and (keyword in label)
        if matched:
            results.append(label)
    return results