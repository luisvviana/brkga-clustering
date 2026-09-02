#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include <streambuf>
#include "../brkga_mp_ipr/brkga_mp_ipr.hpp"
#include "../headers/instance.hpp"
#include "../headers/decoder.hpp"

using namespace std;
using namespace BRKGA;

// ---- Função para ler CSV e remover última coluna (label)
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
        if (first_line) { first_line = false; continue; } // pula cabeçalho

        stringstream ss(line);
        string cell;
        vector<double> row;
        while (getline(ss, cell, ',')) {
            try {
                row.push_back(stod(cell));
            } catch (...) {
                throw runtime_error("Valor inválido no CSV: \"" + cell + "\"");
            }
        }
        if (!row.empty()) {
            row.pop_back();  // Remove a última coluna (label)
            data.push_back(row);
        }
    }
    return data;
}

int main(int argc, char* argv[]) {
    // ---- Parâmetros default
    string instance_file = "";
    unsigned seed;
    double elite_frac;
    double mutant_frac;
    int num_elite_parents;
    int total_parents;
    int num_independent_populations;
    unsigned max_time_seconds = 30;
    unsigned num_threads = 8;

    // ---- Parse manual dos argumentos
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--instance") instance_file = argv[++i];
        else if (arg == "--seed") seed = stoi(argv[++i]);
        else if (arg == "--elitefrac") elite_frac = stod(argv[++i]);
        else if (arg == "--mutantfrac") mutant_frac = stod(argv[++i]);
        else if (arg == "--numeliteparents") num_elite_parents = stod(argv[++i]);
        else if (arg == "--totalparents") total_parents = stod(argv[++i]);
        else if (arg == "--numindependentpopulations") num_independent_populations = stod(argv[++i]);
        else if (arg == "--time") max_time_seconds = stoi(argv[++i]);
        else if (arg == "--threads") num_threads = stoi(argv[++i]);
    }

    if (instance_file.empty()) {
        cerr << "Uso: " << argv[0]
             << " --instance <arquivo.csv> [--seed N --pop N --elite f --mutant f --time N --threads N]" << endl;
        return 1;
    }

    try {
        // 1. Ler dados
        vector<vector<double>> X = read_csv_data(instance_file);
        if (X.empty() || X[0].empty()) throw runtime_error("Arquivo vazio ou mal formatado.");

        // 2. Instância e Decoder
        HybridInstance instance(X);
        HybridDecoderV2 decoder(instance, "sil");

        // 3. Parâmetros BRKGA
        BrkgaParams brkga_params;
        ControlParams control_params;
        
        brkga_params.elite_percentage            = elite_frac;
        brkga_params.mutants_percentage          = mutant_frac;
        brkga_params.num_elite_parents           = num_elite_parents;
        brkga_params.total_parents               = total_parents;
        brkga_params.num_independent_populations = num_independent_populations;

        brkga_params.population_size             = 100;
        brkga_params.bias_type                   = BiasFunctionType::LOGINVERSE;
        brkga_params.pr_number_pairs             = 0;
        brkga_params.pr_minimum_distance         = 0.15;
        brkga_params.pr_type                     = PathRelinking::Type::PERMUTATION;
        brkga_params.pr_selection                = PathRelinking::Selection::BESTSOLUTION;
        brkga_params.alpha_block_size            = 1.0;
        brkga_params.pr_percentage               = 1.0;

        control_params.maximum_running_time = chrono::seconds{max_time_seconds};

        size_t chromosome_size = X[0].size() + 2;

        BRKGA_MP_IPR<HybridDecoderV2> algorithm(
            decoder, Sense::MINIMIZE, seed,
            chromosome_size, brkga_params, num_threads
        );

        std::streambuf* oldCout = std::cout.rdbuf();
        std::ostringstream oss;
        std::cout.rdbuf(oss.rdbuf());  

        // 4. Executar
        const auto result = algorithm.run(control_params);

        std::cout.rdbuf(oldCout);

        // 5. Imprimir APENAS o valor objetivo
        cout << -result.best_fitness << endl;

    } catch (const exception& e) {
        cerr << "Erro: " << e.what() << endl;
        return 1;
    }

    return 0;
}