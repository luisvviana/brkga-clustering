# Setup
# Ajuste o caminho abaixo para a pasta "irace" deste repositório na sua máquina,
# ex.: setwd("C:/Users/<usuario>/brkga-clustering/irace")
setwd("C:/caminho/para/brkga-clustering/irace")
library(irace)
library(iraceplot)

# Irace
scenario <- readScenario(filename = "scenario.txt",
                         scenario = defaultScenario())
irace_main(scenario = scenario)

# Read log file
report("logs/irace.log")