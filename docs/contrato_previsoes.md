# Contrato entre o treino (Pedro) e a avaliação/explicabilidade (Letícia)

Para que estatística, explicabilidade e análise de erro rodem sem retreinar nada,
o treino de cada modelo salva três coisas no repositório (ou link, para os pesos).

## 1. Partição por paciente — `dados/particao.csv`

| coluna | descrição |
|---|---|
| `StudyInstanceUID` | id do exame (= paciente neste desafio) |
| `patient_overall` | rótulo 0/1 |
| `conjunto` | `treino`, `val` ou `teste` |

Estratificada por `patient_overall`, semente 42, gerada uma única vez e nunca
alterada. Época, hiperparâmetros e limiar são escolhidos em `val`; `teste` só é
usado na avaliação final. Exames excluídos (arquivo corrompido etc.) vão em
`dados/excluidos.csv` com o motivo.

## 2. Previsões — `resultados/previsoes.csv`

Uma linha por paciente, por modelo, por semente, para `val` E `teste`:

| coluna | exemplo |
|---|---|
| `StudyInstanceUID` | 1.2.826.0.1.3680043.10001 |
| `y_true` | 1 |
| `prob` | 0.83 (probabilidade de fratura, após sigmoide) |
| `modelo` | `resnet50_congelado`, `resnet50_ft`, `dinov2_vits14_ft`, `rf_ap1`, `trivial_ap1` ... |
| `semente` | 0, 1, 2 |
| `conjunto` | `val` ou `teste` |

Os baselines da AP1 (trivial e Random Forest retreinados na partição nova)
entram no mesmo arquivo, com `modelo = trivial_ap1` e `rf_ap1`.

## 3. Pesos — `pesos/<modelo>_s<semente>.pth` (por link, se grande)

`torch.save(modelo.state_dict(), ...)` e, no código, uma função
`carregar_modelo(nome) -> torch.nn.Module` que devolve o logit de fratura para
uma entrada (B, 3, H, W) normalizada como em `src/aumento.py` (`sem_aumento`).

## 4. Entrada dos modelos (sugestão comum)

Três cortes axiais adjacentes (n-1, n, n+1) em HU como os 3 canais (2.5D),
transformados por `src/aumento.Politica(nome, tamanho)`. O cache dos dados deve
guardar HU em `int16` (e não a imagem já janelada), para permitir a política
alternativa de variação da janela.
