# tp2-g5-cervical-spine

AP2 — Detecção de Fratura Cervical com Transferência de Aprendizado, Aumento de Dados e Explicabilidade (RSNA 2022) — Grupo G5

**Disciplina:** Tópicos Especiais em Sistemas de Informação — Prof. Décio Gonçalves de Aguiar Neto
**Integrantes:** Maria Letícia Pimentel Carlos, Pedro Victor da Silva Lima
**Desafio:** [RSNA 2022 Cervical Spine Fracture Detection](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection)
**AP1 (baseline clássico):** https://github.com/pedrovictor-glitch/tp1-g5-cervical-spine

> Repositório em construção. Este README é atualizado a cada etapa.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `src/aumento.py` | Políticas de aumento de dados (sem aumento, principal, alternativa) com justificativa clínica |
| `src/estatistica.py` | AUC, IC 95 % por bootstrap de pacientes, DeLong pareado, bootstrap pareado, Holm, análise estratificada |
| `src/explicabilidade.py` | Seleção prévia de exemplos, Grad-CAM, Integrated Gradients, teste de sanidade em cascata, pointing game, análise de erro |
| `scripts/gerar_grade_aumento.py` | Gera a grade visual de aumento de dados e a verificação das caixas de fratura (roda no Kaggle) |
| `docs/contrato_previsoes.md` | Formato da partição, das previsões e dos pesos trocados entre treino e avaliação |
| `tests/` | Testes automáticos dos módulos (`python -m pytest tests/`) |
| `figuras/` | Figuras do artigo |

## Dados

Os dados não são redistribuídos. É preciso aceitar as regras da competição no
Kaggle e rodar os notebooks no Kaggle Notebooks, com a competição adicionada
como Input (`/kaggle/input/competitions/rsna-2022-cervical-spine-fracture-detection`).

**Atenção — DICOM comprimido:** parte dos exames vem em *JPEG Lossless*
(10 de 60 numa amostra testada). Sem o descompressor, o `pydicom` falha. Antes
de ler qualquer exame:

```
!pip install -q pylibjpeg pylibjpeg-libjpeg python-gdcm
```

e **reinicie o kernel** (Run → Restart). Na AP1, esse erro era engolido por um
`try/except`, o que provavelmente explica os 37 pacientes perdidos.

## Como gerar a grade de aumento de dados (Kaggle)

```
!git clone -q https://github.com/leticia-pimentelc/tp2-g5-cervical-spine ap2
!cd ap2 && python scripts/gerar_grade_aumento.py
```

Saídas em `/kaggle/working`: `grade_aumento.png/.pdf`, `verificacao_caixas.csv`
e `exemplos_grade.csv` (critério de escolha fixo, semente 42).

## Testes

```
pip install -r requirements.txt
python -m pytest tests/
```

## Uso de IA generativa

Ferramentas de IA generativa (Claude, da Anthropic; Gemini, do Google) foram
usadas como apoio na escrita de código e na revisão de texto. Todo código foi
executado e testado, e os resultados foram verificados pelos autores.
