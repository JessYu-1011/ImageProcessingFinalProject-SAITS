import os
import pandas as pd
import matplotlib.pyplot as plt

folder_path = './evaluation_results'
files = list()

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        full_path = os.path.join(folder_path, filename)
        files.append(full_path)

df = None
for file in files:
    temp_df = pd.read_csv(file)
    if "saits" in file.lower():
        temp_df['model'] = 'SAITS'
    elif "brits" in file.lower():
        temp_df['model'] = 'BRITS'
    elif "gpvae" in file.lower():
        temp_df['model'] = 'GPVAE'
    elif "csdi" in file.lower():
        temp_df['model'] = 'CSDI'
    df = pd.concat([df, temp_df], ignore_index=True)

df.drop('Feature', axis=1, inplace=True)
df.sort_values(by=['Missing_Rate', 'model'], ascending=[True, False], inplace=True)
df['Missing_Rate'] = df['Missing_Rate'].astype(float)
df.to_csv('summary_results.csv', index=False)