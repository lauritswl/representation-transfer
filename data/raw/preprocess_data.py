# This script loads the fiction4 sentiment dataset from Huggingface and saves it as a CSV file.
# %%
# Import necessary libraries
import sys, os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Load from hugginface and save as csv
from src.loader import CorpusLoader
loader = CorpusLoader(text_col='text', label_col='label')
loader.load_from_huggingface("chcaa/fiction4sentiment")
# Save to df
loader.df = loader.df[['sentence', 'label', 'annotator_1', 'annotator_2', 'annotator_3']]
# Only pick first two columns:
loader.df.columns = ['text', 'valence', 'annotator_1', 'annotator_2', 'annotator_3']
loader.df = loader.df[['text', 'valence']]
# Save to csv
loader.df.to_csv('data/clean_text/fiction4.csv', index=False)

# %%
# Load the EmoBank Corpus
Emobank = pd.read_csv("data/raw/emobank.csv")
# Select V, A and text columns
Emobank = Emobank[["V", "A", "D", "text"]]
# Rename and reorder columns to have a common format
Emobank.columns = ["valence", "arousal", "dominance", "text"]
Emobank = Emobank[["text", "valence", "arousal", "dominance"]]
Emobank.head()

# Save to CSV
Emobank.to_csv('data/clean_text/emobank.csv', index=False)

# %%
# Load the Facebook Corpus
Facebook = pd.read_csv("data/raw/facebook.csv")
# Facebook has Valence1 and Valence2 columns, we will use the mean of both as the valence score
Facebook["Valence"] = Facebook[["Valence1", "Valence2"]].mean(axis=1)
# The same for Arousal
Facebook["Arousal"] = Facebook[["Arousal1", "Arousal2"]].mean(axis=1)
# Select Anonymized Message and Valence columns and Arousal columns
Facebook = Facebook[["Anonymized Message", "Valence", "Arousal"]]
# Rename columns to have a common format
Facebook.columns = ["text", "valence", "arousal"]

# Save to CSV
Facebook.to_csv('data/clean_text/facebook.csv', index=False)

# %%
