"""Render LoCoMo10 evaluation metrics as a Rich table."""
import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from examples.locomo_eval import run_locomo_baseline


def fmt(num: float) -> str:
    return f"{num:.4f}"


def render_topk_table(dataset: Path, top_ks: list[int]) -> None:
    table = Table(title=f"LoCoMo10 keyword baseline (dataset={dataset.name})")
    table.add_column("top_k", justify="right")
    table.add_column("questions", justify="right")
    table.add_column("macro_P", justify="right")
    table.add_column("macro_R", justify="right")
    table.add_column("macro_F1", justify="right")
    table.add_column("hit", justify="right")
    table.add_column("mrr", justify="right")
    table.add_column("micro_P", justify="right")
    table.add_column("micro_R", justify="right")
    table.add_column("micro_F1", justify="right")

    for top_k in top_ks:
        metrics = run_locomo_baseline(dataset_path=dataset, top_k=top_k)
        table.add_row(
            str(top_k),
            str(metrics["questions"]),
            fmt(metrics["macro_precision"]),
            fmt(metrics["macro_recall"]),
            fmt(metrics["macro_f1"]),
            fmt(metrics["hit_rate"]),
            fmt(metrics["mrr"]),
            fmt(metrics["micro_precision"]),
            fmt(metrics["micro_recall"]),
            fmt(metrics["micro_f1"]),
        )
    Console().print(table)


def render_category_table(dataset: Path, top_k: int, categories: list[int]) -> None:
    table = Table(title=f"Category breakdown (top_k={top_k}, dataset={dataset.name})")
    table.add_column("category", justify="right")
    table.add_column("questions", justify="right")
    table.add_column("macro_P", justify="right")
    table.add_column("macro_R", justify="right")
    table.add_column("macro_F1", justify="right")
    table.add_column("hit", justify="right")
    table.add_column("mrr", justify="right")
    table.add_column("micro_P", justify="right")
    table.add_column("micro_R", justify="right")
    table.add_column("micro_F1", justify="right")

    for cat in categories:
        metrics = run_locomo_baseline(dataset_path=dataset, top_k=top_k, categories={cat})
        table.add_row(
            str(cat),
            str(metrics["questions"]),
            fmt(metrics["macro_precision"]),
            fmt(metrics["macro_recall"]),
            fmt(metrics["macro_f1"]),
            fmt(metrics["hit_rate"]),
            fmt(metrics["mrr"]),
            fmt(metrics["micro_precision"]),
            fmt(metrics["micro_recall"]),
            fmt(metrics["micro_f1"]),
        )
    Console().print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoCoMo10 keyword baseline and render Rich tables."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data_eval" / "locomo10.json",
        help="Path to the LoCoMo dataset JSON.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[3, 5, 8],
        help="List of top-k values to sweep for the overall table.",
    )
    parser.add_argument(
        "--category-top-k",
        type=int,
        default=5,
        help="Top-k to use for the category breakdown table.",
    )
    parser.add_argument(
        "--categories",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Categories to include in the breakdown table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    render_topk_table(args.dataset, args.top_k)
    render_category_table(args.dataset, args.category_top_k, args.categories)


if __name__ == "__main__":
    main()
