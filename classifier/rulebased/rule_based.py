

def rule_based(feature_list):
    result = []
    for features in feature_list:
        if max(features) == features[0]:
            result.append(1)
        elif max(features) == features[1]:
            result.append(-1)
        elif features[0] == features[1]:
            result.append(0)
    
    return result

            