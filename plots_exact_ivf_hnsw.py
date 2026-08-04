import csv
from pathlib import Path

import matplotlib.pyplot as plt


# =========================================================
# ΑΡΧΕΙΑ ΕΙΣΟΔΟΥ
# =========================================================

BASIC_RESULTS_PATH = Path(
    "benchmark_exact_ivf_hnsw_results.csv"
)

SCALING_RESULTS_PATH = Path(
    "benchmark_scaling_exact_ivf_hnsw_results.csv"
)


# =========================================================
# ΦΑΚΕΛΟΣ ΕΞΟΔΟΥ
# =========================================================

OUTPUT_DIRECTORY = Path(
    "plots_exact_ivf_hnsw"
)


# Σταθερή σειρά εμφάνισης των μεθόδων
METHOD_ORDER = [
    "Exact",
    "FAISS IVF",
    "FAISS HNSW",
]


def convert_to_float(value):
    """
    Μετατρέπει μία τιμή CSV σε float.

    Αν η τιμή είναι κενή ή δεν μπορεί να μετατραπεί,
    επιστρέφει None.
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    try:
        return float(value)

    except ValueError:
        return None


def read_csv_file(file_path: Path):
    """
    Διαβάζει ένα CSV αρχείο και επιστρέφει
    τις γραμμές του ως λίστα από dictionaries.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Δεν βρέθηκε το αρχείο: {file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        return list(reader)


def sort_rows_by_method(rows):
    """
    Ταξινομεί τις γραμμές στη σειρά:
    Exact, FAISS IVF, FAISS HNSW.
    """

    method_positions = {
        method: position
        for position, method in enumerate(
            METHOD_ORDER
        )
    }

    return sorted(
        rows,
        key=lambda row: method_positions.get(
            row["method"],
            len(METHOD_ORDER),
        ),
    )


def save_plot(
    output_name: str,
):
    """
    Αποθηκεύει το τρέχον γράφημα στον φάκελο
    εξόδου και το κλείνει.
    """

    output_path = (
        OUTPUT_DIRECTORY / output_name
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Created: {output_path}")


# =========================================================
# ΒΑΣΙΚΟ BENCHMARK
# =========================================================

def plot_basic_latency(rows):
    """
    Σύγκριση average latency των τριών μεθόδων.
    """

    rows = sort_rows_by_method(rows)

    methods = [
        row["method"]
        for row in rows
    ]

    average_latencies = [
        convert_to_float(
            row["average_latency_ms"]
        )
        for row in rows
    ]

    standard_deviations = [
        convert_to_float(
            row["std_latency_ms"]
        ) or 0.0
        for row in rows
    ]

    plt.figure(figsize=(9, 6))

    plt.bar(
        methods,
        average_latencies,
        yerr=standard_deviations,
        capsize=5,
    )

    plt.xlabel("Search method")
    plt.ylabel("Average latency (ms)")
    plt.title(
        "Average Search Latency: "
        "Exact vs IVF vs HNSW"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    save_plot(
        "basic_latency_comparison.png"
    )


def plot_basic_recall(rows):
    """
    Σύγκριση Recall@5 των τριών μεθόδων.
    """

    rows = sort_rows_by_method(rows)

    methods = [
        row["method"]
        for row in rows
    ]

    recall_values = [
        convert_to_float(
            row["recall_at_k"]
        )
        for row in rows
    ]

    plt.figure(figsize=(9, 6))

    bars = plt.bar(
        methods,
        recall_values,
    )

    plt.xlabel("Search method")
    plt.ylabel("Recall@5")
    plt.title(
        "Recall@5: Exact vs IVF vs HNSW"
    )

    plt.ylim(0, 1.08)

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        recall_values,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    save_plot(
        "basic_recall_comparison.png"
    )


def plot_basic_speedup(rows):
    """
    Σύγκριση speedup σε σχέση με το Exact Search.
    """

    rows = sort_rows_by_method(rows)

    methods = [
        row["method"]
        for row in rows
    ]

    speedups = [
        convert_to_float(
            row["speedup"]
        )
        for row in rows
    ]

    plt.figure(figsize=(9, 6))

    bars = plt.bar(
        methods,
        speedups,
    )

    plt.xlabel("Search method")
    plt.ylabel("Speedup vs Exact")
    plt.title(
        "Search Speedup: "
        "Exact vs IVF vs HNSW"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        speedups,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}x",
            ha="center",
            va="bottom",
        )

    save_plot(
        "basic_speedup_comparison.png"
    )


# =========================================================
# SCALING BENCHMARK
# =========================================================

def group_scaling_rows(rows):
    """
    Ομαδοποιεί τα scaling αποτελέσματα
    ανά μέθοδο και τα ταξινομεί ανά dataset size.
    """

    grouped_rows = {
        method: []
        for method in METHOD_ORDER
    }

    for row in rows:
        method = row["method"]

        if method not in grouped_rows:
            grouped_rows[method] = []

        grouped_rows[method].append(row)

    for method in grouped_rows:
        grouped_rows[method] = sorted(
            grouped_rows[method],
            key=lambda row: int(
                float(row["dataset_size"])
            ),
        )

    return grouped_rows


def plot_scaling_metric(
    grouped_rows,
    metric_name: str,
    y_label: str,
    title: str,
    output_name: str,
    include_exact: bool = True,
):
    """
    Δημιουργεί γενικό line plot για ένα scaling metric.
    """

    plt.figure(figsize=(10, 6))

    for method in METHOD_ORDER:
        if (
            method == "Exact"
            and not include_exact
        ):
            continue

        method_rows = grouped_rows.get(
            method,
            [],
        )

        if not method_rows:
            continue

        dataset_sizes = [
            int(float(row["dataset_size"]))
            for row in method_rows
        ]

        metric_values = [
            convert_to_float(
                row[metric_name]
            )
            for row in method_rows
        ]

        plt.plot(
            dataset_sizes,
            metric_values,
            marker="o",
            linewidth=2,
            label=method,
        )

    plt.xlabel("Dataset size")
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend()

    plt.grid(
        alpha=0.3,
    )

    save_plot(output_name)


def plot_scaling_latency(grouped_rows):
    plot_scaling_metric(
        grouped_rows=grouped_rows,
        metric_name="average_latency_ms",
        y_label="Average latency (ms)",
        title=(
            "Average Search Latency "
            "vs Dataset Size"
        ),
        output_name=(
            "scaling_latency_vs_dataset_size.png"
        ),
    )


def plot_scaling_recall(grouped_rows):
    plt.figure(figsize=(10, 6))

    for method in [
        "FAISS IVF",
        "FAISS HNSW",
    ]:
        method_rows = grouped_rows.get(
            method,
            [],
        )

        if not method_rows:
            continue

        dataset_sizes = [
            int(float(row["dataset_size"]))
            for row in method_rows
        ]

        recall_values = [
            convert_to_float(
                row["recall_at_k"]
            )
            for row in method_rows
        ]

        plt.plot(
            dataset_sizes,
            recall_values,
            marker="o",
            linewidth=2,
            label=method,
        )

    plt.xlabel("Dataset size")
    plt.ylabel("Recall@5")

    plt.title(
        "ANN Recall@5 vs Dataset Size"
    )

    plt.ylim(0, 1.05)

    plt.legend()

    plt.grid(
        alpha=0.3,
    )

    save_plot(
        "scaling_recall_vs_dataset_size.png"
    )


def plot_scaling_speedup(grouped_rows):
    plot_scaling_metric(
        grouped_rows=grouped_rows,
        metric_name="speedup_vs_exact",
        y_label="Speedup vs Exact",
        title=(
            "ANN Speedup vs Dataset Size"
        ),
        output_name=(
            "scaling_speedup_vs_dataset_size.png"
        ),
        include_exact=False,
    )


def plot_scaling_build_time(grouped_rows):
    plot_scaling_metric(
        grouped_rows=grouped_rows,
        metric_name="build_time_sec",
        y_label="Build time (seconds)",
        title=(
            "Index Build Time "
            "vs Dataset Size"
        ),
        output_name=(
            "scaling_build_time_vs_dataset_size.png"
        ),
    )


def plot_scaling_index_size(grouped_rows):
    plot_scaling_metric(
        grouped_rows=grouped_rows,
        metric_name="index_size_mb",
        y_label="Serialized index size (MB)",
        title=(
            "Index Size vs Dataset Size"
        ),
        output_name=(
            "scaling_index_size_vs_dataset_size.png"
        ),
    )


def plot_scaling_p95_latency(grouped_rows):
    plot_scaling_metric(
        grouped_rows=grouped_rows,
        metric_name="p95_latency_ms",
        y_label="P95 latency (ms)",
        title=(
            "P95 Search Latency "
            "vs Dataset Size"
        ),
        output_name=(
            "scaling_p95_latency_vs_dataset_size.png"
        ),
    )


# =========================================================
# MAIN
# =========================================================

def main():
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading benchmark results...")

    basic_rows = read_csv_file(
        BASIC_RESULTS_PATH
    )

    scaling_rows = read_csv_file(
        SCALING_RESULTS_PATH
    )

    print(
        f"Basic benchmark rows: "
        f"{len(basic_rows)}"
    )

    print(
        f"Scaling benchmark rows: "
        f"{len(scaling_rows)}"
    )

    grouped_scaling_rows = (
        group_scaling_rows(
            scaling_rows
        )
    )

    print("\nCreating basic benchmark plots...")

    plot_basic_latency(basic_rows)
    plot_basic_recall(basic_rows)
    plot_basic_speedup(basic_rows)

    print("\nCreating scaling plots...")

    plot_scaling_latency(
        grouped_scaling_rows
    )

    plot_scaling_p95_latency(
        grouped_scaling_rows
    )

    plot_scaling_recall(
        grouped_scaling_rows
    )

    plot_scaling_speedup(
        grouped_scaling_rows
    )

    plot_scaling_build_time(
        grouped_scaling_rows
    )

    plot_scaling_index_size(
        grouped_scaling_rows
    )

    print(
        "\nAll plots were created successfully."
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIRECTORY.resolve()}"
    )


if __name__ == "__main__":
    main()