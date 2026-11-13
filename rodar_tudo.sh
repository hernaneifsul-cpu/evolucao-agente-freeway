#!/bin/bash
# Roda os 3 experimentos com 3 sementes cada e gera o gráfico comparativo.
# A semente 0 gera os arquivos principais (fitness_A.csv, etc.).
python main.py --taxa 0.01 --intensidade 0.05 --nome A --seed 0
python main.py --taxa 0.10 --intensidade 0.20 --nome B --seed 0
python main.py --taxa 0.40 --intensidade 0.50 --nome C --seed 0
python main.py --comparar
# Repetições com outras sementes (opcional, para dar robustez à análise)
for s in 1 2; do
  python main.py --taxa 0.01 --intensidade 0.05 --nome A_seed$s --seed $s
  python main.py --taxa 0.10 --intensidade 0.20 --nome B_seed$s --seed $s
  python main.py --taxa 0.40 --intensidade 0.50 --nome C_seed$s --seed $s
done
