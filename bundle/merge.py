def merge(a, b):
    res = a.copy()
    for key, value in b.items():
        v = value
        if key in res:
            if isinstance(value, dict) and isinstance(res[key], dict):
                v = merge(res[key], value)
        res[key] = v
    return res