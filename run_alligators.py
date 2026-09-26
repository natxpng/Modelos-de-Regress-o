"""
run_alligators.py
==================
Executa a análise completa (Etapas 1-4) para o dataset Alligators.

Y = weight (peso)
X = length (comprimento)
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from regressao_analise import rodar_analise_completa, reajustar_com_transformacao

# ----------------------------------------------------------------------
# CONFIGURAÇÃO
# ----------------------------------------------------------------------
CAMINHO_ARQUIVO = "data/alligators.csv"
SEP = ","
X_VAR = "length"
Y_VAR = "weight"
NOME_DATASET = "alligators"
OUT_DIR = "outputs/alligators"

# ----------------------------------------------------------------------
# ETAPAS 1 a 4
# ----------------------------------------------------------------------
df, modelo, ranking = rodar_analise_completa(
    caminho_arquivo=CAMINHO_ARQUIVO,
    x_var=X_VAR,
    y_var=Y_VAR,
    nome_dataset=NOME_DATASET,
    out_dir=OUT_DIR,
    sep=SEP,
)

# ----------------------------------------------------------------------
# ETAPA 5 — Reajuste com a melhor transformação
# ----------------------------------------------------------------------
# Peso vs comprimento costuma ter relação de potência (weight ~ length^3),
# então log-log é o candidato natural — confirme com o ranking gerado.
TRANS_Y = "log10"   # <-- ajuste conforme o ranking
TRANS_X = "log10"   # <-- ajuste conforme o ranking

if TRANS_Y != "raw" or TRANS_X != "raw":
    df_t, modelo_t, col_x, col_y = reajustar_com_transformacao(
        df, X_VAR, Y_VAR, NOME_DATASET, OUT_DIR, TRANS_Y, TRANS_X
    )
    print(f"\nModelo reajustado com {col_y} ~ {col_x}")
    print(f"R² ajustado (transformado): {modelo_t.rsquared_adj:.4f}")
    print(f"R² ajustado (original):     {modelo.rsquared_adj:.4f}")
