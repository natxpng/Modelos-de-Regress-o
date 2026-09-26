# Atividade 1 — Regressão Linear Simples

Projeto organizado para rodar a análise completa (Etapas 1 a 5) nos três
datasets da atividade: **Treadwear**, **Alligators** e **ConcPatog**.

## 1. Estrutura de pastas

Crie/organize as pastas exatamente assim (o script já cria `outputs/`
sozinho, mas `data/` você precisa preencher):

```
atividade1_regressao/
├── data/                          <- COLOQUE SEUS ARQUIVOS DE DADOS AQUI
│   ├── treadwear.csv
│   ├── alligators.csv
│   └── ConcPatog.txt
├── outputs/                       <- gráficos e tabelas são salvos aqui automaticamente
│   ├── treadwear/
│   ├── alligators/
│   └── concpatog/
├── src/
│   └── regressao_analise.py       <- toda a lógica reutilizável (não precisa editar)
├── run_treadwear.py               <- rode este para o dataset Treadwear
├── run_alligators.py              <- rode este para o dataset Alligators
├── run_concpatog.py               <- rode este para o dataset ConcPatog
├── requirements.txt
└── README.md
```

**Importante:** os nomes dos arquivos em `data/` precisam bater com o que
está configurado em cada `run_*.py` (variável `CAMINHO_ARQUIVO`). Se seus
arquivos tiverem nomes diferentes (ex.: `Treadwear.csv` com T maiúsculo,
ou `treadwear.txt`), ajuste essa linha no script correspondente.

Também confira o **separador** de cada arquivo (variável `SEP`):
- CSV normal → `SEP = ","`
- Arquivo `.txt` separado por tabulação → `SEP = "\t"`
- Se não tiver certeza, abra o arquivo em um editor de texto simples e veja
  se as colunas estão separadas por vírgula, tab ou espaço.

## 2. Instalação (uma vez só)

Abra um terminal dentro da pasta `atividade1_regressao/` e rode:

```bash
# (opcional, mas recomendado) criar um ambiente virtual
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# instalar as dependências
pip install -r requirements.txt
```

## 3. Como rodar cada dataset

Depois que os arquivos estiverem em `data/`, rode cada script separadamente
a partir da pasta raiz do projeto:

```bash
python run_treadwear.py
python run_alligators.py
python run_concpatog.py
```

Cada execução vai:
1. Ler o dataset e mostrar quantas observações foram carregadas.
2. Gerar o **gráfico de dispersão** inicial (Etapa 1).
3. Ajustar o **modelo OLS**, imprimir a tabela de resultados no terminal e
   salvar em `.txt` (Etapa 2).
4. Gerar o gráfico da **reta de regressão com IC e IP de 95%** (Etapa 2).
5. Gerar o **painel de diagnóstico dos resíduos** (histograma, resíduos vs.
   ajustados, QQ-plot) para checar as premissas LINE (Etapa 2).
6. Gerar o **grid 3×3 das 9 transformações** com a correlação de cada uma e
   salvar um **ranking em CSV**, ordenado da transformação mais forte para
   a mais fraca (Etapas 3 e 4).
7. Reajustar o modelo com a transformação escolhida (Etapa 5) — **veja o
   passo 4 abaixo**, isso exige uma decisão sua.

Tudo que for gerado (imagens `.png` e tabelas `.txt`/`.csv`) fica salvo
dentro de `outputs/<nome_do_dataset>/`, numerado na ordem das etapas
(`1_dispersao_...`, `2_sumario_ols_...`, etc.), para facilitar montar o
relatório depois.

## 4. Decidindo a transformação (Etapa 5)

Depois de rodar um script pela primeira vez:

1. Abra `outputs/<dataset>/5_grid_transformacoes_<dataset>.png` e observe
   visualmente qual combinação parece mais linear.
2. Confira `outputs/<dataset>/5_ranking_transformacoes_<dataset>.csv` (ou o
   que foi impresso no terminal) — ele está ordenado pela correlação de
   Pearson em módulo (`Abs_r`), do maior para o menor.
3. Abra o script `run_<dataset>.py` e ajuste as duas linhas no final:

   ```python
   TRANS_Y = "log10"   # troque para "sqrt", "sq", "log10" ou "raw"
   TRANS_X = "log10"
   ```

   conforme a combinação vencedora que você identificou.
4. Rode o script de novo (`python run_<dataset>.py`). Ele vai gerar um novo
   conjunto de gráficos e sumário OLS já com a transformação aplicada,
   prefixados com `_trans_<Y>_<X>` dentro da mesma pasta de outputs, para
   você comparar com o modelo original.

As transformações disponíveis são: `"raw"` (sem transformar), `"sqrt"`
(raiz quadrada), `"sq"` (elevar ao quadrado) e `"log10"` (logaritmo na
base 10 — trata automaticamente valores zero/negativos usando
`log10(x + 1)`).

## 5. Para o relatório em PDF

- **Não inclua trechos de código no PDF final** (conforme pedido pela
  atividade) — use apenas os gráficos e tabelas gerados em `outputs/`.
- Ao comentar cada tabela OLS, destaque: R², R² ajustado, p-valor dos
  coeficientes e estatística F.
- Ao comentar os gráficos de resíduos, relacione explicitamente com as
  premissas L-I-N-E (Linearidade, Independência, Normalidade,
  Homocedasticidade) e explique como a transformação escolhida melhorou
  (ou não) cada uma delas.

## 6. Problemas comuns

| Sintoma | Causa provável | Solução |
|---|---|---|
| `FileNotFoundError` | Caminho/nome do arquivo errado em `CAMINHO_ARQUIVO` | Confira o nome exato do arquivo em `data/` (maiúsculas/minúsculas importam) |
| Colunas todas juntas em uma só | Separador errado | Ajuste `SEP` no script (`","`, `"\t"` ou `r"\s+"`) |
| `KeyError` no nome da coluna | Nome da coluna no arquivo é diferente do configurado | Abra o arquivo e confira o cabeçalho exato; ajuste `X_VAR`/`Y_VAR` |
| Erro no `sqrt` de valores negativos | A transformação sqrt não é válida para negativos | Use `log10` ou `sq` para essa variável, ou pule essa célula do grid |
