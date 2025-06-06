import pandas as pd
from collections import defaultdict, Counter
import glob
import re
from itertools import islice

# Config 
SOURCE_COLUMNS = ["blip2_meaning", "blip2_gesture"]  # Columns to analyze
NGRAM_RANGE = (3, 8)            
TOP_K = 10                       # Top N phrases per culture per column

# Preprocessing 
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    return text

def generate_ngrams(tokens, n):
    return zip(*(islice(tokens, i, None) for i in range(n)))

#  Collect files 
file_paths = glob.glob("*.txt")
if not file_paths:
    print(" No TXT files found.")
    exit()
else:
    print(f" Found {len(file_paths)} .txt file(s).")

#  Phrase counter: {culture, source} → Counter
phrases_by_culture_source = defaultdict(Counter)

#  Process each file
for file_path in file_paths:
    print(f"\n Processing {file_path}")
    try:
        df = pd.read_csv(file_path)

        if 'culture' not in df.columns or not any(col in df.columns for col in SOURCE_COLUMNS):
            print(f" Skipping {file_path} (missing 'culture' or target columns)")
            continue

        for source_col in SOURCE_COLUMNS:
            if source_col not in df.columns:
                continue

            for _, row in df.iterrows():
                culture = str(row['culture']).strip().capitalize()
                text = str(row[source_col])

                if text.lower().strip() in ["", "nan", "none"]:
                    continue

                tokens = clean_text(text).split()
                if not tokens:
                    continue

                for n in range(NGRAM_RANGE[0], NGRAM_RANGE[1] + 1):
                    for gram in generate_ngrams(tokens, n):
                        phrase = " ".join(gram)
                        phrases_by_culture_source[(culture, source_col)][phrase] += 1

    except Exception as e:
        print(f" Error reading {file_path}: {e}")

#  Save and display output
print("\n Most Common Phrases per Culture and Source Column ")
rows = []
for (culture, source_col), counter in sorted(phrases_by_culture_source.items()):
    print(f"\n {culture} — {source_col}:")
    for phrase, count in counter.most_common(TOP_K):
        print(f"  {phrase} ({count})")
        rows.append({
            "culture": culture,
            "source": source_col,
            "phrase": phrase,
            "count": count
        })

output_file = "common_phrases_blip2_by_culture.csv"
pd.DataFrame(rows).to_csv(output_file, index=False)
print(f"\n Saved results to '{output_file}'")

