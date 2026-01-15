import pandas as pd
import matplotlib.pyplot as plt
import os


#files = [
    #'data/processed/windowed/dataset_windowed_train.csv',
    #'data/processed/windowed/dataset_windowed_val.csv',
    #'data/processed/windowed/dataset_windowed_test.csv'
#]
files = [
    'data/processed/windowed/dataset_windowed_train.csv',
    'data/processed/windowed/dataset_windowed_val.csv',
    'data/processed/windowed/dataset_windowed_test.csv'
]
names = ['Train', 'Val', 'Test']
counts_0 = []
counts_1 = []
for f in files:
    df = pd.read_csv(f)
    counts_0.append((df['Seizure'] == 0).sum())
    counts_1.append((df['Seizure'] == 1).sum())

plt.figure(figsize=(6,4))
plt.bar(names, counts_0, label='No Seizure (0)', color='lightgray')
plt.bar(names, counts_1, bottom=counts_0, label='Seizure (1)', color='black')
plt.ylabel('Samples')
plt.title('Class Distribution in Windowed Sets')
plt.legend()
plt.tight_layout()
os.makedirs('images/results', exist_ok=True)
plt.savefig('images/results/desbalanced2.png', dpi=200)
print('Saved to images/results/desbalanced.png')
