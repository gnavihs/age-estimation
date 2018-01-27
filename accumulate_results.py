import os
import json
from pprint import pprint
import operator

final_list = []
for root, dirs, files in os.walk("/data/chercheurs/agarwals/DEX/gauss/logs"):
    for file in files:
        if file.endswith(".log"):
            result = {}
            with open(os.path.join(root, file)) as f:
                counter = 0
                val_mean_absolute_error = 10000
                mean_absolute_error = 0
                val_loss = 0
                loss = 0
                epoch = 0
                for line in f:
                    if counter == 0:
                        data=json.loads(line)
                        counter+=1
                        continue
                    else:
                        data=json.loads(line)
                        # pprint(data)
                        if data['val_mean_absolute_error'] < val_mean_absolute_error:
                            val_mean_absolute_error = data['val_mean_absolute_error']
                            mean_absolute_error = data['mean_absolute_error']
                            val_loss = data['val_loss']
                            loss = data['loss']
                            epoch = data['epoch']
            
            result = {'config':file , 'val_mean_absolute_error': val_mean_absolute_error, 'epoch': epoch,
                      'mean_absolute_error': mean_absolute_error, 'val_loss': val_loss, 'loss': loss }
            final_list.append(result)
    final_list.sort(key=operator.itemgetter('val_mean_absolute_error'))

for item in final_list:
    print("Configuration:", item["config"], "\t Epoch:", item["epoch"]+1, "\t VMAE:", round(item["val_mean_absolute_error"],3),
           "\t MAE:", round(item["mean_absolute_error"],3), "\t Val_loss:", round(item["val_loss"],8), "\t Loss:", round(item["loss"],8))
print("Processed ", len(final_list), "configs")