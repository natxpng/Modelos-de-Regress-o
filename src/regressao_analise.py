"""
regressao_analise.py
=====================
Módulo reutilizável para a Atividade 1: Regressão Linear Simples,
diagnóstico de resíduos (premissas LINE) e busca da melhor
transformação de escala entre 9 combinações.

Este módulo é importado pelos scripts run_treadwear.py, run_alligators.py
e run_concpatog.py — o código aqui NÃO precisa ser editado para cada
dataset, apenas configurado nos scripts run_*.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# ETAPA 1: Gráfico de dispersão inicial
# ---------------------------------------------------------------------------
def plot_dispersao(df, x_var, y_var, titulo, out_path):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=x_var, y=y_var, color="steelblue", alpha=0.7)
    plt.title(titulo)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Gráfico de dispersão salvo em: {out_path}")


# ---------------------------------------------------------------------------
# ETAPA 2: Ajuste OLS + IC/IP + diagnóstico de resíduos (LINE)
# ---------------------------------------------------------------------------
def ajustar_modelo_ols(df, x_var, y_var):
    """Ajusta y ~ x via OLS e retorna o objeto de resultado do statsmodels."""
    formula = f"{y_var} ~ {x_var}"
    modelo = smf.ols(formula, data=df).fit()
    return modelo


def salvar_sumario_ols(modelo, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(modelo.summary().as_text())
    print(f"[OK] Sumário OLS salvo em: {out_path}")


def plot_reta_ic_ip(df, x_var, y_var, modelo, titulo, out_path):
    """Plota a reta de regressão com banda de IC (média) e IP (nova obs.), 95%."""
    x_seq = np.linspace(df[x_var].min(), df[x_var].max(), 200)
    df_pred = pd.DataFrame({x_var: x_seq})
    pred = modelo.get_prediction(df_pred).summary_frame(alpha=0.05)

    plt.figure(figsize=(9, 6))
    plt.scatter(df[x_var], df[y_var], label="Dados observados", alpha=0.6)
    plt.plot(x_seq, pred["mean"], color="red", linewidth=2, label="Reta de regressão")
    plt.fill_between(
        x_seq, pred["mean_ci_lower"], pred["mean_ci_upper"],
        color="red", alpha=0.2, label="IC 95% (resposta média)"
    )
    plt.fill_between(
        x_seq, pred["obs_ci_lower"], pred["obs_ci_upper"],
        color="blue", alpha=0.1, label="IP 95% (nova observação)"
    )
    plt.title(titulo)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Gráfico reta + IC/IP salvo em: {out_path}")


def plot_diagnostico_residuos(modelo, titulo_prefixo, out_path):
    """
    Gera painel com 3 gráficos para checar as premissas LINE:
      (a) Histograma dos resíduos + curva normal teórica  -> Normalidade
      (b) Resíduos vs. valores ajustados                  -> Linearidade / Homocedasticidade
      (c) QQ-plot dos resíduos                             -> Normalidade
    """
    residuos = modelo.resid
    ajustados = modelo.fittedvalues

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Histograma dos resíduos com curva normal teórica
    sns.histplot(residuos, kde=True, stat="density", ax=axs[0], color="skyblue")
    xmin, xmax = axs[0].get_xlim()
    x_pdf = np.linspace(xmin, xmax, 100)
    p_pdf = stats.norm.pdf(x_pdf, np.mean(residuos), np.std(residuos))
    axs[0].plot(x_pdf, p_pdf, "r--", linewidth=2, label="Normal teórica")
    axs[0].set_title("Histograma dos Resíduos")
    axs[0].set_xlabel("Resíduos")
    axs[0].legend()

    # (b) Resíduos vs. valores ajustados
    axs[1].scatter(ajustados, residuos, alpha=0.7, color="purple")
    axs[1].axhline(0, color="red", linestyle="--")
    axs[1].set_xlabel(r"Valores ajustados ($\hat{y}$)")
    axs[1].set_ylabel(r"Resíduos ($e_i$)")
    axs[1].set_title("Resíduos vs. Ajustados")
    axs[1].grid(True, linestyle="--", alpha=0.5)

    # (c) QQ-plot dos resíduos
    sm.qqplot(residuos, line="45", fit=True, ax=axs[2])
    axs[2].set_title("QQ-Plot dos Resíduos")

    fig.suptitle(titulo_prefixo, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Diagnóstico de resíduos salvo em: {out_path}")


def etapa_1_2(df, x_var, y_var, nome_dataset, out_dir, sufixo=""):
    """
    Executa em sequência: dispersão, ajuste OLS, sumário, reta+IC/IP e
    diagnóstico de resíduos. `sufixo` (ex.: '_log') diferencia arquivos
    quando esta função é chamada de novo após uma transformação (Etapa 5).
    """
    tag = f"{nome_dataset}{sufixo}"

    plot_dispersao(
        df, x_var, y_var,
        titulo=f"Dispersão: {y_var} vs {x_var} ({nome_dataset})",
        out_path=os.path.join(out_dir, f"1_dispersao_{tag}.png"),
    )

    modelo = ajustar_modelo_ols(df, x_var, y_var)
    print(modelo.summary())

    salvar_sumario_ols(modelo, os.path.join(out_dir, f"2_sumario_ols_{tag}.txt"))

    plot_reta_ic_ip(
        df, x_var, y_var, modelo,
        titulo=f"Regressão com IC/IP 95%: {y_var} ~ {x_var} ({nome_dataset})",
        out_path=os.path.join(out_dir, f"3_reta_ic_ip_{tag}.png"),
    )

    plot_diagnostico_residuos(
        modelo,
        titulo_prefixo=f"Diagnóstico dos Resíduos — {nome_dataset}{sufixo}",
        out_path=os.path.join(out_dir, f"4_diagnostico_residuos_{tag}.png"),
    )

    return modelo


# ---------------------------------------------------------------------------
# ETAPAS 3 e 4: Grid de 9 transformações de escala
# ---------------------------------------------------------------------------
def _aplicar_transformacao(serie, tipo):
    """Aplica sqrt, quadrado ou log10 a uma série, tratando zeros/negativos."""
    if tipo == "raw":
        return serie
    elif tipo == "sqrt":
        if (serie < 0).any():
            raise ValueError("sqrt: série contém valores negativos.")
        return np.sqrt(serie)
    elif tipo == "sq":
        return serie ** 2
    elif tipo == "log10":
        min_val = serie.min()
        shift = 1.0 if min_val <= 0 else 0.0
        if shift:
            print(f"  [aviso] valores <= 0 encontrados; usando log10(x + 1).")
        return np.log10(serie + shift)
    else:
        raise ValueError(f"Transformação desconhecida: {tipo}")


def grid_transformacoes(df, x_var, y_var, nome_dataset, out_dir):
    """
    Gera o painel 3x3 com as 9 combinações de transformação (sqrt, sq, log10
    para Y cruzado com sqrt, sq, log10 para X), calcula a correlação de
    Pearson de cada combinação e devolve um DataFrame ordenado (ranking).
    """
    transformacoes = ["sqrt", "sq", "log10"]

    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(
        f"Grid de 9 Transformações de Escala — {nome_dataset} ({y_var} vs {x_var})",
        fontsize=16,
    )

    combos = []
    for i, t_y in enumerate(transformacoes):
        for j, t_x in enumerate(transformacoes):
            try:
                y_trans = _aplicar_transformacao(df[y_var], t_y)
                x_trans = _aplicar_transformacao(df[x_var], t_x)
                corr = np.corrcoef(x_trans, y_trans)[0, 1]
                erro = None
            except ValueError as e:
                y_trans, x_trans, corr, erro = None, None, np.nan, str(e)

            ax = axs[i, j]
            if erro is None:
                ax.scatter(x_trans, y_trans, alpha=0.6, color="teal")
                ax.set_title(f"Y:{t_y} vs X:{t_x}  (r={corr:.3f})", fontsize=10)
            else:
                ax.text(0.5, 0.5, "inválido\n(valores negativos)",
                        ha="center", va="center", fontsize=9, color="red")
                ax.set_title(f"Y:{t_y} vs X:{t_x}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.4)

            combos.append({"Trans_Y": t_y, "Trans_X": t_x, "Pearson_r": corr})

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"5_grid_transformacoes_{nome_dataset}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Grid de transformações salvo em: {out_path}")

    df_combos = pd.DataFrame(combos)
    df_combos["Abs_r"] = df_combos["Pearson_r"].abs()
    df_combos = df_combos.sort_values(by="Abs_r", ascending=False).reset_index(drop=True)

    ranking_path = os.path.join(out_dir, f"5_ranking_transformacoes_{nome_dataset}.csv")
    df_combos.to_csv(ranking_path, index=False)
    print(f"[OK] Ranking de transformações salvo em: {ranking_path}")
    print(df_combos)

    return df_combos


# ---------------------------------------------------------------------------
# ETAPA 5: Reajuste do modelo com a melhor transformação
# ---------------------------------------------------------------------------
def reajustar_com_transformacao(df, x_var, y_var, nome_dataset, out_dir,
                                 trans_y, trans_x):
    """
    Cria as colunas transformadas (ex.: log10(Y), log10(X)) e roda
    novamente a etapa_1_2 completa (dispersão, OLS, IC/IP, diagnóstico)
    sobre os dados transformados. Retorna (df_transformado, modelo, nomes das colunas).
    """
    df_t = df.copy()

    col_y = f"{y_var}_{trans_y}"
    col_x = f"{x_var}_{trans_x}"

    df_t[col_y] = _aplicar_transformacao(df_t[y_var], trans_y)
    df_t[col_x] = _aplicar_transformacao(df_t[x_var], trans_x)

    modelo = etapa_1_2(
        df_t, col_x, col_y, nome_dataset, out_dir, sufixo=f"_trans_{trans_y}_{trans_x}"
    )

    return df_t, modelo, col_x, col_y


# ---------------------------------------------------------------------------
# Função "tudo em um": roda etapas 1, 2, 3 e 4 automaticamente.
# A etapa 5 é chamada à parte pois exige decisão humana sobre qual
# transformação usar (com base no grid e no ranking).
# ---------------------------------------------------------------------------
def rodar_analise_completa(caminho_arquivo, x_var, y_var, nome_dataset,
                            out_dir, sep=","):
    """
    Executa etapas 1 a 4 de ponta a ponta para um dataset:
      1) leitura dos dados
      2) dispersão inicial
      3) OLS + sumário + IC/IP + diagnóstico de resíduos
      4) grid de 9 transformações + ranking por correlação
    Retorna (df, modelo_original, df_ranking_transformacoes).
    """
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(caminho_arquivo, sep=sep)
    df = df[[x_var, y_var]].dropna()
    print(f"\n===== {nome_dataset}: {len(df)} observações carregadas =====")

    modelo = etapa_1_2(df, x_var, y_var, nome_dataset, out_dir)

    df_ranking = grid_transformacoes(df, x_var, y_var, nome_dataset, out_dir)

    return df, modelo, df_ranking
