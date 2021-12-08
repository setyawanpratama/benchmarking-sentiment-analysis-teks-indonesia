from sklearn.metrics import f1_score

def rule_based(feature_list, label):
    is_posneg = len(set(label)) == 2
    is_combined = len(feature_list[0]) == 4
    
    prediction = []
    if is_posneg:
        for feature in feature_list:
            pos = feature[0] if not is_combined else feature[0]+feature[2]
            neg = feature[1] if not is_combined else feature[1]+feature[3]
            
            if pos >= neg:
                prediction.append(1)
            else:
                prediction.append(0)
    else:
        for features in feature_list:
            pos = feature[0] if not is_combined else feature[0]+feature[2]
            neg = feature[1] if not is_combined else feature[1]+feature[3]
            if pos > neg:
                prediction.append(1)
            elif neg > pos:
                prediction.append(-1)
            else:
                prediction.append(0)
    
    f1_score = f1_score(label, result, average='micro')
    print("F1-score: {:.3f}".format(mean(scores), std(scores)))
    
    return f1_score