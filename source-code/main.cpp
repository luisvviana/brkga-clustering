#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <vector>
#include <filesystem>
#include "../brkga_mp_ipr/brkga_mp_ipr.hpp"
#include "../headers/instance.hpp"
#include "../headers/decoder.hpp"

using namespace std;
using namespace std::filesystem;
using namespace BRKGA;

vector<vector<double>> read_csv_data(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Não foi possível abrir o arquivo: " + filename);
    }

    vector<vector<double>> data;
    string line;
    bool first_line = true;

    while (getline(file, line)) {
        if (line.empty()) continue;
        if (first_line) {
            first_line = false;
            continue;
        }

        stringstream ss(line);
        string cell;
        vector<double> row;
        while (getline(ss, cell, ',')) {
            try {
                row.push_back(stod(cell));
            } catch (...) {
                throw runtime_error("Valor invalido no CSV: \"" + cell + "\"");
            }
        }

        if (!row.empty()) {
            row.pop_back();
            data.push_back(row);
        }
    }

    return data;
}

void run_for_metric(
    const string& metric,
    const unsigned num_runs,
    const unsigned seed,
    const string& config_file,
    const unsigned max_time_seconds,
    const unsigned num_threads
) {
    const string results_dir = "decode_individuals/individuals/1_thread/" + metric;
    create_directories(results_dir);

    for (unsigned run = 1; run <= num_runs; run++) {
        const string output_path = results_dir + "/" + to_string(run) + ".txt";
        ofstream output_file(output_path);

        if (!output_file.is_open()) {
            cerr << "Não foi possível criar o arquivo: " << output_path << endl;
            continue;
        }

        cout << "\n=== [" << metric << "] Execução " << run << " de " << num_runs << " ===" << endl;

        for (const auto& entry : recursive_directory_iterator("data-lakes")) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".csv") continue;

            string dataset_name = entry.path().filename().string();
            cout << "  Processando: " << dataset_name << endl;

            try {
                vector<vector<double>> X = read_csv_data(entry.path().string());
                if (X.empty() || X[0].empty()) throw runtime_error("Arquivo vazio ou mal formatado.");

                HybridInstance instance(X);
                HybridDecoderV2 decoder(instance, metric);

                auto [brkga_params, control_params] = readConfiguration(config_file);
                control_params.maximum_running_time = chrono::seconds{max_time_seconds};

                size_t chromosome_size = X[0].size() + 2;

                BRKGA_MP_IPR<HybridDecoderV2> algorithm(
                    decoder, Sense::MINIMIZE, seed,
                    chromosome_size, brkga_params, num_threads
                );

                auto start = chrono::steady_clock::now();
                const auto result = algorithm.run(control_params);
                auto end = chrono::steady_clock::now();
                chrono::duration<double> duration = end - start;

                output_file << dataset_name << ", ";
                for (auto gene : algorithm.getBestChromosome())
                    output_file << gene << " ";
                output_file << ", " << duration.count() << "s" << endl;

            } catch (const exception& e) {
                cerr << "  Erro com " << dataset_name << ": " << e.what() << endl;
            }
        }

        output_file.close();
        cout << "  Resultados salvos em: " << output_path << endl;
    }

    cout << "\nExecuções de \"" << metric << "\" concluídas. Resultados em: " << results_dir << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        cerr << "Usage: " << argv[0]
             << " <seed> <config-file> <maximum-running-time-seconds> <metric: sil|db|ch|all> <num-runs>" << endl;
        return 1;
    }

    const unsigned seed = stoi(argv[1]);
    const string config_file = argv[2];
    const unsigned max_time_seconds = stoi(argv[3]);
    const string metric = argv[4];
    const unsigned num_runs = stoi(argv[5]);

    const vector<string> valid_metrics = {"sil", "db", "ch"};

    if (metric != "all" && find(valid_metrics.begin(), valid_metrics.end(), metric) == valid_metrics.end()) {
        cerr << "Métrica inválida: \"" << metric << "\". Use: sil, db, ch ou all." << endl;
        return 1;
    }

    if (num_runs == 0) {
        cerr << "O número de execuções deve ser maior que zero." << endl;
        return 1;
    }

    const unsigned num_threads = 1;

    if (metric == "all") {
        for (const auto& m : valid_metrics)
            run_for_metric(m, num_runs, seed, config_file, max_time_seconds, num_threads);

        cout << "\nTodas as métricas concluídas com " << num_runs << " execuções cada." << endl;
    } else {
        run_for_metric(metric, num_runs, seed, config_file, max_time_seconds, num_threads);

        cout << "\nTodas as " << num_runs << " execuções concluídas. Resultados em: decode_individuals/individuals/" << metric << endl;
    }

    return 0;
}