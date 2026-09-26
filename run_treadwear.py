"""
run_treadwear.py
=================
Executa a análise completa (Etapas 1-4) para o dataset Treadwear.

Y = groove (profundidade do sulco)
X = mileage (quilometragem)

Depois de rodar, veja o ranking impresso no terminal (e o arquivo
outputs/treadwear/5_ranking_transformacoes_treadwear.csv) para decidir
qual transformação usar na Etapa 5, e então ajuste as variáveis
TRANS_Y / TRANS_X abaixo e rode de novo (ou rode só o bloco da Etapa 5).
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from regressao_analise import rodar_analise_completa, reajustar_com_transformacao

# ----------------------------------------------------------------------
# CONFIGURAÇÃO — ajuste os caminhos/nomes se seus arquivos forem diferentes
# ----------------------------------------------------------------------
CAMINHO_ARQUIVO = "data/treadwear.csv"   # troque para .txt se necessário
SEP = ","                                 # troque para "\t" se o arquivo for tab-separado
X_VAR = "mileage"
Y_VAR = "groove"
NOME_DATASET = "treadwear"
OUT_DIR = "outputs/treadwear"

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
# Depois de olhar o grid (outputs/treadwear/5_grid_transformacoes_treadwear.png)
# e o ranking impresso acima, escolha a combinação vencedora e preencha aqui.
# Opções válidas: "raw", "sqrt", "sq", "log10"
TRANS_Y = "log10"   # <-- ajuste conforme o ranking
TRANS_X = "log10"   # <-- ajuste conforme o ranking

if TRANS_Y != "raw" or TRANS_X != "raw":
    df_t, modelo_t, col_x, col_y = reajustar_com_transformacao(
        df, X_VAR, Y_VAR, NOME_DATASET, OUT_DIR, TRANS_Y, TRANS_X
    )
    print(f"\nModelo reajustado com {col_y} ~ {col_x}")
    print(f"R² ajustado (transformado): {modelo_t.rsquared_adj:.4f}")
    print(f"R² ajustado (original):     {modelo.rsquared_adj:.4f}")
