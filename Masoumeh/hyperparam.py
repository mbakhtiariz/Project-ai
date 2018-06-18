from collections import defaultdict

def load_hyperparams(param_path):
    param_file = open(param_path, "r")

    hyperparams = defaultdict()

    keywords = ["loss",'filterNumStart', "lr", "epochs",
                "lambda2", "batchSize", "doBatchNorm", "channels",
                "dropout", "depth", "valid_size", "shuffle",
                "pin_memory", "num_workers", "tolerance","cls_alpha",
                "img_w", "img_h", "mask_w", "mask_h",
                "flip_prob", "rotate_prob", "elastic_deform_prob", "blur_prob",
                "jitter_prob"]


    types = ["string", "int", "float", "int",
             "float", "int", "int", "int",
             "float", "int","float", "string",
             "string", "int", "float","float",
             "int", "int", "int", "int",
             "float", "float", "float", "float",
             "float"]


    key_type = {}
    for i in range(len(keywords)):
        key_type[keywords[i]] = types[i]
    print(key_type)

    for line in param_file:
        info = line.replace(' ', '').strip().split('=')
        # print(info)
        # print('----------------')
        list_value = []
        if (key_type[info[0]] in ["float"]):
            print(info)
            #hyperparams[info[0]] = list(map(float, info[1].split(',')))
            hyperparams[info[0]] = float(info[1])
        elif (key_type[info[0]] in ["int"]):            
            #hyperparams[info[0]] = list(map(int, info[1].split(',')))
            hyperparams[info[0]] = int(info[1])
        else:
            #hyperparams[info[0]] = info[1].split(',')
            hyperparams[info[0]] = info[1]

    print(hyperparams)
    return hyperparams


if __name__ == '__main__':

    # Load PARAMS:
    hyper_params = load_hyperparams("hyper_params")
    print(hyper_params["batchSize"])