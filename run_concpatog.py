"""
run_concpatog.py
=================
Executa a análise completa (Etapas 1-4) para o dataset ConcPatog.txt.

Y = CPatog2 (concentração do patógeno 2)
X = CPatog1 (concentração do patógeno 1)

Atenção: este arquivo é .txt, normalmente separado por TAB — confirme
o separador correto abaixo (SEP) e ajuste se o arquivo usar vírgula
ou espaço.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from regressao_analise import rodar_analise_completa, reajustar_com_transformacao

# ----------------------------------------------------------------------
# CONFIGURAÇÃO
# ----------------------------------------------------------------------
CAMINHO_ARQUIVO = "data/ConcPatog.txt"
SEP = "\t"          # troque para "," ou r"\s+" se necessário
X_VAR = "CPatog1"
Y_VAR = "CPatog2"
NOME_DATASET = "concpatog"
OUT_DIR = "outputs/concpatog"

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
# Este dataset pode conter zeros (concentração = 0). A função de
# transformação já trata isso automaticamente usando log10(x + 1)
# quando encontra valores <= 0.
TRANS_Y = "log10"   # <-- ajuste conforme o ranking
TRANS_X = "log10"   # <-- ajuste conforme o ranking

if TRANS_Y != "raw" or TRANS_X != "raw":
    df_t, modelo_t, col_x, col_y = reajustar_com_transformacao(
        df, X_VAR, Y_VAR, NOME_DATASET, OUT_DIR, TRANS_Y, TRANS_X
    )
    print(f"\nModelo reajustado com {col_y} ~ {col_x}")
    print(f"R² ajustado (transformado): {modelo_t.rsquared_adj:.4f}")
    print(f"R² ajustado (original):     {modelo.rsquared_adj:.4f}")
