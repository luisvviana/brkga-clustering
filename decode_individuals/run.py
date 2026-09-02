# main.py

import sys
import os
import warnings
import pandas
import random

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.simplefilter("ignore")

from decoder import decode_2

METRICS = ["sil", "db", "ch"]
METRIC_COLUMN = {
    "sil": "silhouette",
    "db":  "davies-bouldin",
    "ch":  "calinski-harabasz"
}

def compute_metric(metric, X_sel, labels):
    if metric == "sil":
        return silhouette_score(X_sel, labels)
    elif metric == "db":
        return davies_bouldin_score(X_sel, labels)
    elif metric == "ch":
        return calinski_harabasz_score(X_sel, labels)

def parse_chromosome(parts):
    """Extrai o cromossomo (genes entre o nome do dataset e o tempo final)."""
    chromosome = []
    for gene in parts[1:-1]:
        gene = gene.strip().replace(",", "")
        if gene == "":
            continue
        try:
            chromosome.append(float(gene))
        except ValueError:
            return []
    return chromosome

def process_results_file(results_path, files_by_name, metric, output):
    with open(results_path, "r", encoding="utf-8") as f:
        results_lines = [line.strip() for line in f if line.strip()]

    for line in results_lines:
        parts = line.strip().split()
        if len(parts) < 3:
            print(f"Linha inválida em {results_path}: {line}")
            continue

        dataset_name = os.path.basename(parts[0]).rstrip(",")

        if dataset_name not in files_by_name:
            print(f"Dataset {dataset_name} não encontrado nos data-lakes.")
            continue

        file_path = files_by_name[dataset_name]

        chromosome = parse_chromosome(parts)
        if not chromosome:
            print(f"Cromossomo inválido para {dataset_name}.")
            continue

        time_taken = parts[-1].rstrip("s")

        try:
            dataset = pandas.read_csv(file_path)
            dataset = dataset.iloc[:, :-1].values
            dataset = StandardScaler().fit_transform(dataset)

            k, A = decode_2(dataset, chromosome)
            X_sel = dataset[:, [i for i in range(len(A)) if A[i] == 1]]

            if X_sel.shape[1] == 0 or k <= 1:
                print(f"Seleção inválida para {dataset_name}: {X_sel.shape[1]} features, {k} clusters.")
                continue

            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X_sel)
            score = compute_metric(metric, X_sel, labels)

            output.write(f"{dataset_name},{time_taken},{k},{score}\n")
            output.flush()

        except Exception as e:
            print(f"Erro ao processar {dataset_name}: {e}")

def main():
    metric_arg = sys.argv[1] if len(sys.argv) > 1 else "sil"
    random.seed(42)

    metrics_to_run = METRICS if metric_arg == "all" else [metric_arg]

    if metric_arg != "all" and metric_arg not in METRICS:
        print(f"Métrica inválida: '{metric_arg}'. Use: sil, db, ch ou all.")
        sys.exit(1)

    # Indexa todos os datasets pelo nome do arquivo
    files_by_name = {
        os.path.basename(str(f)): str(f)
        for f in Path("data-lakes/").rglob("*.csv")
    }

    for metric in metrics_to_run:
        # individuals_dir = Path(f"decode_individuals/individuals/{metric}") # 12 threads
        individuals_dir = Path(f"decode_individuals/individuals/1_thread/{metric}") # 1 thread
        if not individuals_dir.exists():
            print(f"Pasta não encontrada: {individuals_dir}. Pulando métrica '{metric}'.")
            continue

        results_files = sorted(individuals_dir.glob("*.txt"))
        if not results_files:
            print(f"Nenhum arquivo .txt encontrado em {individuals_dir}.")
            continue

        # output_dir = Path(f"results/brkga/{metric}") # 12 threads
        output_dir = Path(f"results/brkga/1_thread/{metric}") # 1 thread
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Processando métrica: {metric} ({len(results_files)} arquivo(s)) ===")

        for results_path in results_files:
            run_number = results_path.stem  # "1", "2", etc.
            output_path = output_dir / f"{run_number}.csv"

            print(f"  Lendo {results_path} → {output_path}")

            with open(output_path, "wt", encoding="utf-8") as output:
                output.write(f"dataset,time,number-clusters,{METRIC_COLUMN[metric]}\n")
                process_results_file(results_path, files_by_name, metric, output)

    print("\nProcessamento concluído. Resultados salvos em results/brkga/")

if __name__ == "__main__":
    main()