import numpy as np
import pandas as pd
import statistics
import csv

input = ["preds_resnet34_20epochs.csv", "preds_resnet50_8epochs.csv", "preds_resnext101_12epochs.csv"]
df1 = pd.read_csv('data/' + input[0])
df2 = pd.read_csv('data/' + input[1])
df3 = pd.read_csv('data/' + input[2])

dfs = [df1, df2, df3]

print(dfs[0])
print(dfs[1])
print(dfs[2])

fields = ["path", "class"]
filename = "consolidated_preds.csv"
with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
    output = csv.writer(csvfile)
    output.writerow(fields)

    for idx in range(dfs[0].shape[0]):
        img = dfs[0].iloc[idx]["path"]
        results = []

        for df in dfs:
            results.append(df.iloc[idx]["class"])
        
        res = statistics.mode(results)
        output.writerow([img, res])
        


