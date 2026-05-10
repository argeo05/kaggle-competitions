from pathlib import Path
import pandas as pd

def blend_submissions(weight_dict, output_path, target_col):
    dataframes = []

    # Load each submission with its weight
    for path, weight in weight_dict.items():
        df = pd.read_csv(path)

        # Add a weighted prediction column
        df["weighted_pred"] = df[target_col] * weight

        # Append to list
        dataframes.append(df[["id", "weighted_pred"]])

    # Merge all submissions on 'id'
    merged = dataframes[0]
    for df in dataframes[1:]:
        merged = merged.merge(df, on="id", how="inner", suffixes=("", "_dup"))

        if "weighted_pred_dup" in merged.columns:
            merged["weighted_pred"] += merged["weighted_pred_dup"]
            merged.drop(columns=["weighted_pred_dup"], inplace=True)

    total_weight = sum(weight_dict.values())

    merged[target_col] = merged["weighted_pred"] / total_weight

    blended = merged[["id", target_col]]

    blended.to_csv(output_path, index=False)
    print(f"Blended submission saved to {output_path}")