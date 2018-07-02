import shutil
import numpy as np
import os

adrs = '../data/Glas/'
new_adrs = '../data/datasets/'

f = open(adrs+'Grade.csv')

if not os.path.exists(new_adrs):
    os.makedirs(new_adrs)

grd_file = open(new_adrs + "/grd.csv", "w")

num_train = 85
indices = list(range(num_train))
np.random.shuffle(indices)
valid_size = 0.1
split = int(np.ceil(valid_size * num_train))
print(split)

counter = 0
for l in f:
    if "train" in l:
        if counter in indices[0:split]:#for writing val in csv
            print("counter",counter)
            grd_file.write("val" + l[5:])
            name = l.split(',')[0]
            new_name = "val"+name[5:]
        else:#for writing train in new csv
            grd_file.write(l)
            name = l.split(',')[0]
            new_name = name

        print(name, new_name)
        shutil.copy2(adrs+name+'.bmp', new_adrs+new_name+'.bmp')# for changing name of file from train to val if needed.
        shutil.copy2(adrs + name + '_anno.bmp', new_adrs + new_name + '_anno.bmp')
        counter += 1
    else:
        grd_file.write(l)# for writing tests and first line in csv
        name = l.split(',')[0]
        print(name)
        if not 'name' in l:# for copying test images to new adrs
            shutil.copy2(adrs + name+'.bmp', new_adrs+name+'.bmp')
            shutil.copy2(adrs + name + '_anno.bmp', new_adrs + name + '_anno.bmp')

    grd_file.close()
    grd_file = open(new_adrs + "/grd.csv", "a")