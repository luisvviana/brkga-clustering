# BRKGA-Based Clustering

[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](./paper/BRKGA_Based_Clustering_SBPO.pdf)

Repositório dedicado ao armazenamento do código desenvolvido no trabalho intitulado **Agrupamento de dados em Data Lakes utilizando a meta-heurística BRKGA**, publicado no **LVIII Simpósio Brasileiro de Pesquisa Operacional**.

## Como rodar

### Código principal (algoritmo)

```console
g++ -std=c++20 -fopenmp -O3 -Ibrkga_mp_ipr source-code/main.cpp source-code/decoder.cpp source-code/instance.cpp source-code/silhouette.cpp source-code/davies_bouldin.cpp source-code/calinski_harabasz.cpp -o main

./main seed config_file max_time metric num_runs

exemplo:
./main 42 config.conf 300 sil 10
```
> Nota: Use `all` no parâmetro `metric` para rodar o código `num_runs` vezes para todas as métricas

---

### irace (otimização de hiperparâmetros)

O pipeline de irace deste repositório foi feito para rodar no **Windows** (usa `target-runner.bat` e um script `.R` que chama `setwd()` com caminho do Windows). Pré-requisitos: [R](https://cran.r-project.org/) com os pacotes `irace` e `iraceplot`, e um `g++` no PATH (ex.: via [MSYS2/MinGW-w64](https://www.msys2.org/)).

1. Compile o executável **dentro da pasta `irace/`**, com o nome `main-irace.exe` (é o nome que `target-runner.bat` espera encontrar):

    ```console
    g++ -std=c++20 -fopenmp -O3 -Ibrkga_mp_ipr irace/main.cpp source-code/decoder.cpp source-code/instance.cpp source-code/silhouette.cpp source-code/davies_bouldin.cpp source-code/calinski_harabasz.cpp -o irace/main-irace.exe
    ```

2. Abra `irace/script.R`, ajuste a linha `setwd(...)` para o caminho da pasta `irace/` na sua máquina, e rode o script no R/RStudio. Ele lê `irace/scenario.txt` (que por sua vez aponta para `target-runner.bat` e `parameters/parameters.txt`), executa o irace e no final chama `report()` sobre `irace/logs/irace.log`.

3. Os hiperparâmetros tunados estão definidos em `irace/parameters/parameters.txt` (`elitefrac`, `mutantfrac`, `numeliteparents`, `totalparents`, `numindependentpopulations`). Os demais parâmetros do BRKGA (ex.: `population_size`) estão hard-coded em `irace/main.cpp` e **não** são lidos de `config.conf`.

> Nota: o irace otimiza **apenas** os parâmetros informados em `parameters.txt`; os demais estão hard-coded no arquivo `irace/main.cpp`.