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
    'label_vs_meaning': [],
    'label_vs_gesture': [],
    'meaning_vs_gesture': []
})

# Process each file
for file_path in file_paths:
    print(f"\nProcessing: {file_path}")
    try:
        df = pd.read_csv(file_path)

        expected_cols = {'culture', 'meaning_score', 'gesture_score'}
        if not expected_cols.issubset(df.columns):
            print(f"  Skipping {file_path} — missing required columns.")
            continue

        # Normalize culture
        df['culture'] = df['culture'].astype(str).str.strip()
        invalid_mask = df['culture'].str.lower().isin(['', 'nan', 'none']) | df['culture'].isna()
        invalid_rows = df[invalid_mask]

        if not invalid_rows.empty:
            print(f"   Found {len(invalid_rows)} row(s) with invalid 'culture' in {file_path}:")
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
        print(f"   Added scores for cultures: {', '.join(grouped.index)}")

        # BERTScore per culture
        for culture in df['culture'].unique():
            sub_df = df[df['culture'] == culture]
            if len(sub_df) == 0:
                continue

            refs = sub_df['label'].astype(str).tolist()
            preds_meaning = sub_df['blip2_meaning'].astype(str).tolist()
            preds_gesture = sub_df['blip2_gesture'].astype(str).tolist()

            result_lm = bertscore.compute(predictions=preds_meaning, references=refs, lang="en")
            result_lg = bertscore.compute(predictions=preds_gesture, references=refs, lang="en")
            result_mg = bertscore.compute(predictions=preds_gesture, references=preds_meaning, lang="en")

            f1_lm = sum(result_lm['f1']) / len(result_lm['f1'])
            f1_lg = sum(result_lg['f1']) / len(result_lg['f1'])
            f1_mg = sum(result_mg['f1']) / len(result_mg['f1'])

            bert_scores_by_culture[culture]['label_vs_meaning'].append(f1_lm)
            bert_scores_by_culture[culture]['label_vs_gesture'].append(f1_lg)
            bert_scores_by_culture[culture]['meaning_vs_gesture'].append(f1_mg)

    except Exception as e:
        print(f"   Error reading {file_path}: {e}")

# Print and save culture scores
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

# Print and save BERTScores
print("\nBERTScores by Culture ")
bert_rows = []
for culture in sorted(bert_scores_by_culture):
    lm = bert_scores_by_culture[culture]['label_vs_meaning']
    lg = bert_scores_by_culture[culture]['label_vs_gesture']
    mg = bert_scores_by_culture[culture]['meaning_vs_gesture']

    avg_lm = sum(lm)/len(lm) if lm else 0.0
    avg_lg = sum(lg)/len(lg) if lg else 0.0
    avg_mg = sum(mg)/len(mg) if mg else 0.0

    print(f"{culture}:")
    print(f"   Label vs Meaning:  F1 = {avg_lm:.4f}")
    print(f"   Label vs Gesture:  F1 = {avg_lg:.4f}")
    print(f"   Meaning vs Gesture: F1 = {avg_mg:.4f}")

    bert_rows.append({
        'culture': culture,
        'bert_f1_label_vs_meaning': avg_lm,
        'bert_f1_label_vs_gesture': avg_lg,
        'bert_f1_meaning_vs_gesture': avg_mg
    })
pd.DataFrame(bert_rows).to_csv("bert_score_summary.csv", index=False)
print(" Saved BERTScores to 'bert_score_summary.csv'.")

