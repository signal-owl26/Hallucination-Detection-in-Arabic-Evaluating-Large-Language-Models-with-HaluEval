import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Display settings
# ============================================================

MODEL_DISPLAY_NAMES = {
    "gpt4o": "GPT-4o",
    "deepseek-chat": "DeepSeek-Chat",
    "deepseek-reasoner": "Deepseek-Reasoner",
    "ace": "AceGPT",
    "mistral": "Mistral",
    "llama": "Llama",
}

MODEL_ORDER = [
    "gpt4o",
    "deepseek-chat",
    "deepseek-reasoner",
    "ace",
    "mistral",
    "llama",
]

CONFIG_ORDER = ["Ar+Ar", "En+Ar", "En+En"]

METRIC_ROWS = [
    ("Accuracy%", "accuracy"),
    ("TN", "TN"),
    ("FP", "FP"),
    ("FN", "FN"),
    ("TP", "TP"),
    ("failed", "failed"),
    ("MCC", "mcc"),
    ("precision", "precision"),
    ("Recall", "recall"),
    ("F1", "f1"),
    ("Specificity", "specificity"),
]


# ============================================================
# Label normalization
# ============================================================

def normalize_label(value):
    """
    Convert model output / ground truth to binary labels.

    1 = hallucinated
    0 = supported
    None = failed / invalid
    """
    if value is None:
        return None

    text = str(value).strip().lower().strip(" .,!؟;:")

    if text in {"yes", "نعم"}:
        return 1

    if text in {"no", "لا"}:
        return 0

    if text == "failed!":
        return None

    return None


# ============================================================
# File loading
# ============================================================

def load_jsonl(file_path):
    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSONL in file {file_path} at line {line_number}: {e}"
                )

    return pd.DataFrame(rows)


def find_result_files(results_dir, configs=None, models=None):
    results_dir = Path(results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    result_files = []

    for config_dir in sorted(results_dir.iterdir()):
        if not config_dir.is_dir():
            continue

        config_name = config_dir.name

        if configs is not None and config_name not in configs:
            continue

        for file_path in sorted(config_dir.glob("*.json")):
            model_name = file_path.stem

            if models is not None and model_name not in models:
                continue

            result_files.append(
                {
                    "config": config_name,
                    "model": model_name,
                    "path": file_path,
                }
            )

    return result_files


# ============================================================
# Metric calculation
# ============================================================

def compute_metrics(file_path):
    df = load_jsonl(file_path)

    required_columns = {"id", "ground_truth", "judgement"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"File {file_path} is missing columns: {sorted(missing_columns)}"
        )

    df["y_true"] = df["ground_truth"].apply(normalize_label)
    df["y_pred"] = df["judgement"].apply(normalize_label)

    failed = int(df["y_pred"].isna().sum())

    df_valid = df.dropna(subset=["y_true", "y_pred"]).copy()

    if df_valid.empty:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "specificity": None,
            "mcc": None,
            "failed": failed,
            "TN": 0,
            "FP": 0,
            "FN": 0,
            "TP": 0,
        }

    df_valid["y_true"] = df_valid["y_true"].astype(int)
    df_valid["y_pred"] = df_valid["y_pred"].astype(int)

    y_true = df_valid["y_true"]
    y_pred = df_valid["y_pred"]

    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    TP = int(((y_true == 1) & (y_pred == 1)).sum())

    valid_samples = len(df_valid)

    accuracy = (TP + TN) / valid_samples
    precision = TP / (TP + FP) if (TP + FP) else None
    recall = TP / (TP + FN) if (TP + FN) else None

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    specificity = TN / (TN + FP) if (TN + FP) else None

    denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    mcc = ((TP * TN) - (FP * FN)) / math.sqrt(denominator) if denominator else None

    return {
        "accuracy": accuracy * 100,
        "precision": precision * 100 if precision is not None else None,
        "recall": recall * 100 if recall is not None else None,
        "f1": f1 * 100 if f1 is not None else None,
        "specificity": specificity * 100 if specificity is not None else None,
        "mcc": mcc,
        "failed": failed,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TP": TP,
    }


def build_metrics_df(result_files):
    rows = []

    for item in result_files:
        metrics = compute_metrics(item["path"])

        rows.append(
            {
                "model": item["model"],
                "configuration": item["config"],
                **metrics,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Agreement calculation
# ============================================================

def load_predictions(file_path):
    df = load_jsonl(file_path)

    required_columns = {"id", "judgement"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"File {file_path} is missing columns: {sorted(missing_columns)}"
        )

    return pd.DataFrame(
        {
            "id": df["id"].astype(str),
            "prediction": df["judgement"].apply(normalize_label),
        }
    )


def cohen_kappa_binary(y_a, y_b):
    if len(y_a) == 0:
        return None

    observed_agreement = float((y_a == y_b).mean())

    p_a_yes = float((y_a == 1).mean())
    p_b_yes = float((y_b == 1).mean())

    expected_agreement = (
        ((1 - p_a_yes) * (1 - p_b_yes))
        + (p_a_yes * p_b_yes)
    )

    if np.isclose(1 - expected_agreement, 0.0):
        return None

    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def compute_agreement(file_a, file_b):
    df_a = load_predictions(file_a).rename(columns={"prediction": "pred_a"})
    df_b = load_predictions(file_b).rename(columns={"prediction": "pred_b"})

    merged = df_a.merge(df_b, on="id", how="inner")
    valid = merged.dropna(subset=["pred_a", "pred_b"]).copy()

    if valid.empty:
        return {
            "agreement": None,
            "cohen_kappa": None,
            "no_to_yes": None,
            "yes_to_no": None,
        }

    y_a = valid["pred_a"].astype(int).to_numpy()
    y_b = valid["pred_b"].astype(int).to_numpy()

    return {
        "agreement": float((y_a == y_b).mean()) * 100,
        "cohen_kappa": cohen_kappa_binary(y_a, y_b),
        "no_to_yes": int(((y_a == 0) & (y_b == 1)).sum()),
        "yes_to_no": int(((y_a == 1) & (y_b == 0)).sum()),
    }


def build_agreement_df(result_files):
    path_map = {
        (item["model"], item["config"]): item["path"]
        for item in result_files
    }

    agreement_pairs = [
        ("En+Ar", "Ar+Ar"),
        ("En+En", "Ar+Ar"),
        ("En+En", "En+Ar"),
    ]

    rows = []

    for model in sorted({item["model"] for item in result_files}):
        for config_a, config_b in agreement_pairs:
            key_a = (model, config_a)
            key_b = (model, config_b)

            if key_a not in path_map or key_b not in path_map:
                continue

            result = compute_agreement(path_map[key_a], path_map[key_b])

            rows.append(
                {
                    "model": model,
                    "comparison": f"{config_a} vs {config_b}",
                    **result,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Table formatting
# ============================================================

def format_value(row_name, value):
    if value is None or pd.isna(value):
        return ""

    # Integer rows
    if row_name in {"TN", "FP", "FN", "TP", "failed", "No→Yes", "Yes→No"}:
        return str(int(value))

    # All decimal rows: two digits after the comma/dot
    return f"{float(value):.2f}"


def make_metrics_table(metrics_df):
    columns = []

    available_models = list(metrics_df["model"].unique())

    ordered_models = [m for m in MODEL_ORDER if m in available_models]
    ordered_models += [m for m in sorted(available_models) if m not in ordered_models]

    for model in ordered_models:
        available_configs = set(
            metrics_df.loc[metrics_df["model"] == model, "configuration"]
        )

        ordered_configs = [c for c in CONFIG_ORDER if c in available_configs]
        ordered_configs += [c for c in sorted(available_configs) if c not in ordered_configs]

        for config in ordered_configs:
            columns.append((model, config))

    table = []

    for row_name, metric_name in METRIC_ROWS:
        row = []

        for model, config in columns:
            match = metrics_df[
                (metrics_df["model"] == model)
                & (metrics_df["configuration"] == config)
            ]

            if match.empty:
                row.append("")
            else:
                row.append(format_value(row_name, match.iloc[0][metric_name]))

        table.append(row)

    row_labels = [row_name for row_name, _ in METRIC_ROWS]
    col_labels = [config for _, config in columns]
    group_labels = [MODEL_DISPLAY_NAMES.get(model, model) for model, _ in columns]

    return table, row_labels, col_labels, group_labels


def make_agreement_table(agreement_df):
    wanted = [
        ("ace", "En+Ar vs Ar+Ar"),
        ("gpt4o", "En+Ar vs Ar+Ar"),
        ("gpt4o", "En+En vs Ar+Ar"),
        ("gpt4o", "En+En vs En+Ar"),
    ]

    columns = []

    for model, comparison in wanted:
        match = agreement_df[
            (agreement_df["model"] == model)
            & (agreement_df["comparison"] == comparison)
        ]

        if not match.empty:
            columns.append((model, comparison))

    row_map = [
        ("Agreement (%)", "agreement"),
        ("Cohen's κ", "cohen_kappa"),
        ("No→Yes", "no_to_yes"),
        ("Yes→No", "yes_to_no"),
    ]

    table = []

    for row_name, metric_name in row_map:
        row = []

        for model, comparison in columns:
            match = agreement_df[
                (agreement_df["model"] == model)
                & (agreement_df["comparison"] == comparison)
            ]

            if match.empty:
                row.append("")
            else:
                row.append(format_value(row_name, match.iloc[0][metric_name]))

        table.append(row)

    row_labels = [row_name for row_name, _ in row_map]
    col_labels = [comparison for _, comparison in columns]
    group_labels = [MODEL_DISPLAY_NAMES.get(model, model) for model, _ in columns]

    return table, row_labels, col_labels, group_labels


# ============================================================
# Save table as PNG
# ============================================================

def save_table_png(table, row_labels, col_labels, group_labels, output_path, title=None):
    n_rows = len(table)
    n_cols = len(col_labels)

    if n_cols == 0:
        print(f"Skipping empty table: {output_path}")
        return

    fig_width = max(16, n_cols * 2.05)
    fig_height = max(5.5, n_rows * 0.6 + 2.4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    full_table = ax.table(
        cellText=table,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
        bbox=[0.03, 0.03, 0.94, 0.76],
    )

    full_table.auto_set_font_size(False)
    full_table.set_fontsize(13)
    full_table.scale(1.3, 1.8)

    for (row, col), cell in full_table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.7)

        if row == 0:
            cell.set_text_props(weight="bold", fontsize=13)
            cell.set_height(cell.get_height() * 1.25)

        if col == -1:
            cell.set_text_props(weight="bold", fontsize=13)
            cell.set_width(cell.get_width() * 1.45)

    # Add model names above configuration/comparison headers
    previous_group = None
    start_col = 0

    for i, group in enumerate(group_labels + [None]):
        if i == 0:
            previous_group = group
            start_col = 0
            continue

        if group != previous_group:
            end_col = i - 1

            # Center of the model group
            center = 0.03 + 0.94 * ((start_col + end_col + 1) / (2 * n_cols))

            ax.text(
                center,
                0.88,
                previous_group,
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                transform=ax.transAxes,
            )

            # Line under model group name
            left = 0.03 + 0.94 * (start_col / n_cols)
            right = 0.03 + 0.94 * ((end_col + 1) / n_cols)

            ax.plot(
                [left + 0.01, right - 0.01],
                [0.84, 0.84],
                color="black",
                linewidth=0.9,
                transform=ax.transAxes,
            )

            start_col = i
            previous_group = group

    if title:
        ax.set_title(title, fontsize=17, fontweight="bold", pad=25)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=400, bbox_inches="tight", pad_inches=0.25)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    project_root = Path("../..").resolve()

    results_dir = project_root / "data" / "evaluation"
    output_dir = project_root / "result"

    output_dir.mkdir(parents=True, exist_ok=True)

    result_files = find_result_files(results_dir=results_dir)

    if not result_files:
        raise FileNotFoundError(
            f"No result files found in {results_dir}. "
            f"Expected files like data/evaluation/En+Ar/gpt4o.json"
        )

    metrics_df = build_metrics_df(result_files)
    agreement_df = build_agreement_df(result_files)

    metrics_table, metrics_rows, metrics_cols, metrics_groups = make_metrics_table(metrics_df)
    agreement_table, agreement_rows, agreement_cols, agreement_groups = make_agreement_table(agreement_df)

    metrics_png = output_dir / "metrics_table.png"
    agreement_png = output_dir / "agreement_table.png"

    save_table_png(
        table=metrics_table,
        row_labels=metrics_rows,
        col_labels=metrics_cols,
        group_labels=metrics_groups,
        output_path=metrics_png,
        title=None,
    )

    save_table_png(
        table=agreement_table,
        row_labels=agreement_rows,
        col_labels=agreement_cols,
        group_labels=agreement_groups,
        output_path=agreement_png,
        title=None,
    )

    print("Saved:")
    print(metrics_png)
    print(agreement_png)

if __name__ == "__main__":
    main()