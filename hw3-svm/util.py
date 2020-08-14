# Taken from http://web.stanford.edu/class/cs221/ Assignment #2 Support Code


def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items()) 
    """ d1.get(f,0)은 f가 있으면 f의 value를 가져오고 없으면 0을 반환 """

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.

    NOTE: This function does not return anything, but rather
    increments d1 in place. We do this because it is much faster to
    change elements of d1 in place than to build a new dictionary and
    return it.
    
    d1과d2를 합치는 함수임. w를 업데이트 할 때 쓸 수 있다.
    """
    for f, v in d2.items():
        if f == 1 or f == -1:
            continue
        else:
            d1[f] = d1.get(f, 0) + v * scale
