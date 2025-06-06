import pandas as pd
from collections import defaultdict
import glob
import evaluate

# Load BERTScore
bertscore = evaluate.load("bertscore")

# Get all .txt files in current directory
file_paths = glob.glob("*.txt")

if not file_paths:
    print("No TXT files found in the current directory.")
else:
    print(f"Found {len(file_paths)} TXT files:")
    for f in file_paths:
        print(f"- {f}")

# Data containers
score_counts = defaultdict(lambda: {'meaning_score': 0, 'gesture_score': 0})
bert_scores_by_culture = defaultdict(lambda: {
    'result_vs_label': []
})

# Process each file
for file_path in file_paths:
    print(f"\nProcessing: {file_path}")
    try:
        df = pd.read_csv(file_path)

        expected_cols = {'culture', 'meaning_score', 'gesture_score', 'label', 'result'}
        if not expected_cols.issubset(df.columns):
            print(f"  Skipping {file_path} — missing required columns: {expected_cols - set(df.columns)}")
            continue

        # Normalize culture
        df['culture'] = df['culture'].astype(str).str.strip()
        invalid_mask = df['culture'].str.lower().isin(['', 'nan', 'none']) | df['culture'].isna()
        invalid_rows = df[invalid_mask]

        if not invalid_rows.empty:
            print(f"   Found {len(invalid_rows)} row(s) with invalid 'culture' in {file_path}:")
            if 'filename' in df.columns:
                print("  → Affected image filenames:")
                for fn in invalid_rows['filename']:
                    print(f"    - {fn}")

        df = df[~invalid_mask]

        # Safe conversion
        df['meaning_score'] = pd.to_numeric(df['meaning_score'], errors='coerce').fillna(0).astype(int)
        df['gesture_score'] = pd.to_numeric(df['gesture_score'], errors='coerce').fillna(0).astype(int)

        # Aggregate scores
        grouped = df.groupby(df['culture'].str.strip().str.capitalize())[
            ['meaning_score', 'gesture_score']
        ].apply(lambda x: (x == 1).sum())

        for culture, row in grouped.iterrows():
            score_counts[culture]['meaning_score'] += row['meaning_score']
            score_counts[culture]['gesture_score'] += row['gesture_score']
        print(f"  Added scores for cultures: {', '.join(grouped.index)}")

        # BERTScore per culture (result vs label)
        for culture in df['culture'].unique():
            sub_df = df[df['culture'] == culture]
            if len(sub_df) == 0:
                continue

            refs = sub_df['label'].astype(str).tolist()
            preds = sub_df['result'].astype(str).tolist()

            try:
                result_rl = bertscore.compute(predictions=preds, references=refs, lang="en")
                f1_rl = sum(result_rl['f1']) / len(result_rl['f1'])
                bert_scores_by_culture[culture]['result_vs_label'].append(f1_rl)
            except Exception as e:
                print(f"   BERTScore failed for culture '{culture}': {e}")

    except Exception as e:
        print(f"   Error reading {file_path}: {e}")

#  Save culture scores 
print("\n Total Scores by Culture ")
culture_rows = []
for culture in sorted(score_counts):
    scores = score_counts[culture]
    print(f"{culture}: Meaning Score = {scores['meaning_score']}, Gesture Score = {scores['gesture_score']}")
    culture_rows.append({
        'culture': culture,
        'meaning_score': scores['meaning_score'],
        'gesture_score': scores['gesture_score']
    })
pd.DataFrame(culture_rows).to_csv("culture_score_summary.csv", index=False)
print(" Saved culture scores to 'culture_score_summary.csv'.")

# Save BERTScores 
print("\n BERTScores by Culture (Result vs Label) ")
bert_rows = []
for culture in sorted(bert_scores_by_culture):
    rl = bert_scores_by_culture[culture]['result_vs_label']
    avg_rl = sum(rl) / len(rl) if rl else 0.0

    print(f"{culture}:  Result vs Label BERTScore F1 = {avg_rl:.4f}")

    bert_rows.append({
        'culture': culture,
        'bert_f1_result_vs_label': avg_rl
    })
pd.DataFrame(bert_rows).to_csv("bert_score_summary.csv", index=False)
print(" Saved BERTScores to 'bert_score_summary.csv'.")

