def bucket_match(bucket1, bucket2):
    if bucket2 == 'high':
        if bucket1 in ['mid','high']:
            return True
        else:
            return False
    elif bucket2 == 'low':
        if bucket1 in ['mid','low']:
            return True
        else:
            return False
    elif bucket2 == 'mid':
        if bucket1 in ['mid']:
            return True
        else:
            return False
    
def get_target_prob_from_bucket(bucket, classname):
    if classname=='formality':
        if bucket=="low":
            return 0.0
        elif bucket=="mid":
            return 0.5
        elif bucket=="high":
            return 1.0
    elif classname=='emo':
        if bucket=="low":
            return 0.0
        elif bucket=="mid":
            return 0.5
        elif bucket=="high":
            return 1.0
    elif classname=='shakespeare':
        if bucket=="low":
            return 0.0
        elif bucket=="mid":
            return 0.5
        elif bucket=="high":
            return 1.0
    

def get_buckets(prob, classname):
    if classname=='formality':
        if prob<0.2:
            return 'low'
        elif prob>0.9:
            return 'high'
        else:
            return 'mid'
    elif classname=='emo':
        if prob<0.25:
            return 'low'
        elif prob>0.9:
            return 'high'
        else:
            return 'mid'
    if classname=='shakespeare':
        if prob<0.1:
            return 'low'
        elif prob>0.9:
            return 'high'
        else:
            return 'mid'

def filter_results(suggestions, classnames, target_buckets):
    for classname, target_bucket in zip(classnames, target_buckets):
        suggestions = [x for x in suggestions if bucket_match(get_buckets(float(x['probs'][classname]),classname),target_bucket)]
    return suggestions
        

def sort_results(suggestions, classnames, target_buckets):
    if len(suggestions)>0:
        target_diff = []
        for sug in suggestions:
            temp=0
            for classname, target_bucket in zip(classnames, target_buckets):
                temp += abs(float(sug['probs'][classname]) - get_target_prob_from_bucket(target_bucket, classname))
            target_diff.append(temp)
        assert len(target_diff) == len(suggestions)
        suggestions = [x for _, x in sorted(zip(target_diff, suggestions), key=lambda pair: pair[0])]
        return suggestions
    else:
        return []
