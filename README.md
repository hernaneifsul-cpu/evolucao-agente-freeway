# Neuroevolução no Freeway (ALE/Freeway-v5)

UniSenac Pelotas - Fundamentos de Inteligência Artificial - 2026/2
Hernane Lemes

## Como executar

```bash
pip install "gymnasium[atari]" numpy matplotlib
python main.py --taxa 0.01 --intensidade 0.05 --nome A
python main.py --taxa 0.10 --intensidade 0.20 --nome B
python main.py --taxa 0.40 --intensidade 0.50 --nome C
python main.py --comparar          # gráfico juntando A, B e C
python main.py --render --nome B   # mostra o melhor agente do experimento B jogando
```
Ou tudo de uma vez: `./rodar_tudo.sh`. Argumentos extras: `--populacao`, `--geracoes`, `--elite`, `--seed`.

## Configuração
- **Ambiente:** ALE/Freeway-v5 (Gymnasium), observação RAM (vetor de 128 valores), ações 0=NOOP, 1=UP, 2=DOWN.
- **Rede (indivíduo):** 128 -> 16 (tanh) -> 3; ação = argmax da saída. Pesos e bias são o DNA.
- **População:** 20 | **Gerações:** 30
- **Seleção:** truncamento (50% de maior fitness viram pais). _Justificativa: simples e mantém pressão seletiva clara._
- **Elitismo:** 2 indivíduos copiados sem mutação.
- **Reprodução:** cópia (deepcopy) de um pai sorteado + mutação. Sem crossover.
- **Mutação:** cada parâmetro muta com probabilidade `taxa`, somando ruído gaussiano de desvio `intensidade`.
- **Fitness:** soma das recompensas do episódio. Todos os indivíduos de uma geração usam a mesma seed.

## Resultados dos experimentos
| Exp | Taxa | Intensidade | Melhor final | Média final | Melhor global |
|-----|------|-------------|--------------|-------------|---------------|
| A | 1% | 0,05 | | | |
| B | 10% | 0,20 | | | |
| C | 40% | 0,50 | | | |

Gráficos: `resultados/fitness_A.png`, `fitness_B.png`, `fitness_C.png`, `comparacao_experimentos.png`.

## Baseline aleatório x melhor agente
_(copie de `resultados/baseline_*.txt`)_

## Análise
**Exploração:** _(população inicial aleatória + mutações)_
**Convergência:** _(o que os gráficos mostram; em qual geração estabilizou)_
**Efeito da mutação (A x B x C):** _(interpretação)_
**O que mudaríamos para melhorar:** _(ex.: crossover, mais gerações, avaliar em vários episódios)_
