# main.py

import sys
import warnings
warnings.simplefilter("ignore")

import time
import pandas
import random

from pathlib import Path
from sklearn.preprocessing import StandardScaler

import romulo

METRICS = ["sil", "db", "ch"]
METRIC_COLUMN = {
    "sil": "silhouette",
    "db":  "davies-bouldin",
    "ch":  "calinski-harabasz",
}

def get_output_dir(metric):
    if metric == "all":
        return Path("results/romulo/original")
    return Path(f"results/romulo/{metric}")

def get_header(metric):
    if metric == "all":
        return "dataset,time,number-clusters,silhouette,davies-bouldin,calinski-harabasz\n"
    return f"dataset,time,number-clusters,{METRIC_COLUMN[metric]}\n"

def write_result(output, file, elapsed, clustering, metric):
    name = Path(file).name
    k    = clustering.number_clusters

    if metric == "sil":
        output.write(f"{name},{elapsed},{k},{clustering.silhouette}\n")
    elif metric == "db":
        output.write(f"{name},{elapsed},{k},{clustering.davies_bouldin}\n")
    elif metric == "ch":
        output.write(f"{name},{elapsed},{k},{clustering.calinski_harabasz}\n")
    else:  # all
        output.write(f"{name},{elapsed},{k},{clustering.silhouette},{clustering.davies_bouldin},{clustering.calinski_harabasz}\n")

def run(metric, num_runs, configuration, files):
    output_dir = get_output_dir(metric)
    output_dir.mkdir(parents=True, exist_ok=True)

    for run_idx in range(1, num_runs + 1):
        output_path = output_dir / f"{run_idx}.csv"
        print(f"\n=== [{metric}] Execução {run_idx} de {num_runs} ===")

        clusterer = romulo.Clusterer(configuration, metric=metric)

        with open(output_path, "wt", encoding="utf-8") as output:
            output.write(get_header(metric))

            for file in files:
                print(f"  Processando: {Path(file).name}")

                try:
                    dataset = pandas.read_csv(file)
                    dataset = dataset.iloc[:, :-1].values
                    dataset = StandardScaler().fit_transform(dataset)
                    dataset = pandas.DataFrame(data=dataset)

                    t0 = time.time()
                    clustering = clusterer.work(dataset)
                    t1 = time.time()

                    write_result(output, file, t1 - t0, clustering, metric)
                    output.flush()

                except Exception as e:
                    print(f"  Erro ao processar {file}: {e}")

        print(f"  Resultados salvos em: {output_path}")

def main():
    metric_arg = sys.argv[1] if len(sys.argv) > 1 else "sil"
    num_runs   = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    valid = METRICS + ["all"]
    if metric_arg not in valid:
        print(f"Métrica inválida: '{metric_arg}'. Use: sil, db, ch ou all.")
        sys.exit(1)

    if num_runs < 1:
        print("O número de execuções deve ser maior que zero.")
        sys.exit(1)

    random.seed(42)
    configuration = {"CXPB": 0.30, "MUTPB": 0.15, "NGEN": 35, "PSIZE": 35, "MU": 0.70, "LAMBDA": 0.70}

    files = sorted([str(f) for f in Path("data-lakes/").rglob("*.csv")])

    metrics_to_run = METRICS if metric_arg == "all" else [metric_arg]

    # Quando "all", roda uma única execução com o fitness combinado (pasta original)
    # Quando métrica específica, roda num_runs vezes na pasta correspondente
    if metric_arg == "all":
        run("all", num_runs, configuration, files)
    else:
        for metric in metrics_to_run:
            run(metric, num_runs, configuration, files)

    print("\nProcessamento concluído. Resultados salvos em results/romulo/")

if __name__ == "__main__":
    main()