from collections import defaultdict

def load_hyperparams(param_path):
    param_file = open(param_path, "r")

    hyperparams = defaultdict()

    keywords = ["loss",'filterNumStart', "lr", "epochs", "lambda2", "batchSize", "doBatchNorm", "channels", "dropout", "depth",
                "mask", "labels", "dataPath"]
    types = ["string", "int", "listfloat", "int", "float", "int", "int", "liststring", "float", "int", "string", "string",
             "string"]

    key_type = {}
    for i in range(len(keywords)):
        key_type[keywords[i]] = types[i]
    print(key_type)

    for line in param_file:
        info = line.replace(' ', '').strip().split('=')
        # print(info)
        # print('----------------')
        list_value = []
        if (key_type[info[0]] in ["listfloat", "float"]):
            print(info)
            hyperparams[info[0]] = list(map(float, info[1].split(',')))
        elif (key_type[info[0]] in ["int"]):            
            hyperparams[info[0]] = list(map(int, info[1].split(',')))
        else:
            hyperparams[info[0]] = info[1].split(',')

    return hyperparams
