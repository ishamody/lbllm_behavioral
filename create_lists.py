import pandas as pd
import random

# =========================
# LOAD MASTER SPREADSHEET
# =========================
input_file = 'stimuli/LB-LLM-master-spreadsheet.xlsx'
df = pd.read_excel(input_file)

# Keep only first 800 sentence IDs, so no word lists
df['sentence_id'] = pd.to_numeric(df['sentence_id'], errors='coerce') #convert string to numbers
df = df[df['sentence_id'] <= 800]
df['sentence_id'] = df['sentence_id'].astype('Int64').astype(str)

# Keep only rows with actual sentences
ndf = df[['sentence_id', 'sentence', 'bucket']].dropna(subset=['sentence'])

# Shuffle all sentences
ndf = ndf.sample(frac=1, random_state=None).reset_index(drop=True)

# =========================
# CREATE 8 LISTS OF 100
# =========================
n_lists = 8
sentences_per_list = 100

# Use first 800 rows
ndf = ndf.iloc[:n_lists * sentences_per_list].copy()
list_numbers = []

for i in range(n_lists):
    list_numbers.extend([i + 1] * sentences_per_list)

random.shuffle(list_numbers)
ndf['list_number'] = list_numbers

# Sort by list number for readability
ndf = ndf.sort_values('list_number').reset_index(drop=True)

# =========================
# SAVE MASTER ASSIGNMENT
# =========================
output_file = 'sentence_lists.csv'
ndf.to_csv(output_file, index=False)

print('Created:', output_file)

# =========================
# SAVE INDIVIDUAL LIST FILES
# =========================
for i in range(1, 9):
    list_df = ndf[ndf['list_number'] == i]
    list_df.to_csv(f'list_{i}.csv', index=False)
    print(f"items in list_{i}", len(list_df))

print('Created list_1.csv through list_8.csv')