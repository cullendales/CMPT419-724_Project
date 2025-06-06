import os
import pandas as pd
import matplotlib.pyplot as plt

# Set working directory
base_path = "."
plot_output_path = os.path.join(base_path, "plots")
os.makedirs(plot_output_path, exist_ok=True)

model_folders = {
    "BLIP2": "BLIP2_Evaluation",
    "LLaVA": "Llava_Evaluation",
    "Perplexity": "Perplexity_Evaluation"
}

dfs_bert = []
dfs_culture_score = []

for model_name, folder in model_folders.items():
    bert_path = os.path.join(base_path, folder, "bert_score_summary.csv")
    score_path = os.path.join(base_path, folder, "culture_score_summary.csv")

    # Read BERTScore summary
    df_bert = pd.read_csv(bert_path)
    df_bert.columns = df_bert.columns.str.strip()
    df_bert['model'] = model_name

    # Normalize for comparison
    if model_name in ['LLaVA', 'Perplexity'] and 'bert_f1_result_vs_label' in df_bert.columns:
        df_bert.rename(columns={'bert_f1_result_vs_label': 'bert_f1_label_vs_gesture'}, inplace=True)

    if 'bert_f1_label_vs_gesture' not in df_bert.columns:
        raise ValueError(f"'bert_f1_label_vs_gesture' column missing in {bert_path}")

    df_bert = df_bert[['culture', 'bert_f1_label_vs_gesture', 'model']]
    df_bert = df_bert.rename(columns={'bert_f1_label_vs_gesture': 'bert_f1_score'})
    df_bert['comparison_type'] = 'Label vs Gesture'
    dfs_bert.append(df_bert)

    # Read culture score summary
    df_score = pd.read_csv(score_path)
    df_score.columns = df_score.columns.str.strip()
    df_score['model'] = model_name
    dfs_culture_score.append(df_score)

# Combine and plot BERTScore
bert_combined = pd.concat(dfs_bert)
plt.figure(figsize=(12, 6))
pivot_bert = bert_combined.pivot(index='culture', columns='model', values='bert_f1_score')
pivot_bert.plot(kind='bar', figsize=(12, 6))
plt.title('BERTScore Comparison: Label vs Gesture')
plt.ylabel('BERT F1 Score')
plt.ylim(0.6, 0.9)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(title='Model')
plt.savefig(os.path.join(plot_output_path, "label_vs_gesture.png"))
plt.close()

# Combine and plot culture scores
culture_combined = pd.concat(dfs_culture_score)

for score_type in ['meaning_score', 'gesture_score']:
    plt.figure(figsize=(12, 6))
    pivot_score = culture_combined.pivot(index='culture', columns='model', values=score_type)
    pivot_score.plot(kind='bar', figsize=(12, 6))
    plt.title(f'{score_type.replace("_", " ").title()} Comparison Across Models')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title='Model')
    filename = f"{score_type}.png"
    plt.savefig(os.path.join(plot_output_path, filename))
    plt.close()

print(f"All plots saved in: {plot_output_path}")

