def slurp_file(data_file):
    with open(data_file, encoding='utf-8') as f:
        data = f.read()
    return data

