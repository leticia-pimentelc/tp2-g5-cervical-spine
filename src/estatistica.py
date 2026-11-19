"""Protocolo estatístico da AP2 (G5). Responsável: Maria Letícia Pimentel Carlos.

Entrada: CSV de previsões no formato do contrato (docs/contrato_previsoes.md):
    StudyInstanceUID, y_true, prob, modelo, semente, conjunto
Uma linha por paciente, por modelo, por semente. `conjunto` é 'val' ou 'teste'.

O que este módulo calcula
- AUC-ROC (métrica principal), média e desvio entre as 3 sementes;
- IC 95 % por bootstrap em nível de paciente (reamostra pacientes, recalcula a
  AUC média das sementes em cada réplica);
- teste de DeLong pareado (DeLong et al., 1988; algoritmo rápido de Sun e Xu,
  2014) sobre a probabilidade média das sementes;
- bootstrap pareado da diferença de AUC (média das sementes);
- métricas no limiar (sensibilidade, especificidade, F1, acurácia balanceada),
  com o limiar escolhido NA VALIDAÇÃO (Youden) e aplicado ao teste;
- análise estratificada por subgrupo (positivos do subgrupo contra todos os
  negativos), por exemplo nº de vértebras fraturadas e nível C1-C2 x C3-C7.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             roc_curve)

SEED = 42


# ----------------------------------------------------------------------------
# DeLong (Sun & Xu, IEEE Signal Processing Letters 21(11), 2014)
# ----------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        a = i
        while a < n and z[a] == z[i]:
            a += 1
        t[i:a] = 0.5 * (i + a - 1) + 1
        i = a
    out = np.empty(n)
    out[j] = t
    return out


def _delong_cov(y: np.ndarray, preds: np.ndarray):
    """preds: (k, n). Devolve (aucs[k], cov[k,k])."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")  # positivos primeiro
    preds = np.atleast_2d(preds)[:, order]
    m = int(y.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        raise ValueError("DeLong precisa de positivos e negativos")
    k = preds.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(preds[r, :m])
        ty[r] = _midrank(preds[r, m:])
        tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_ic(y, p, alpha=0.05):
    """AUC e IC (1-alpha) de DeLong para um modelo."""
    auc, cov = _delong_cov(y, np.asarray(p)[None, :])
    se = float(np.sqrt(cov[0, 0]))
    z = stats.norm.ppf(1 - alpha / 2)
    return float(auc[0]), max(0.0, auc[0] - z * se), min(1.0, auc[0] + z * se)


def delong_pareado(y, p1, p2):
    """Teste de DeLong para duas AUCs nos MESMOS pacientes. Devolve dict."""
    aucs, cov = _delong_cov(y, np.vstack([p1, p2]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        return {"auc_1": aucs[0], "auc_2": aucs[1], "diferenca": diff, "z": 0.0, "p": 1.0}
    z = diff / np.sqrt(var)
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "diferenca": diff,
            "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


# ----------------------------------------------------------------------------
# Bootstrap em nível de paciente
# ----------------------------------------------------------------------------
def _matriz(df: pd.DataFrame, modelo: str, conjunto: str = "teste"):
    """Pivot pacientes x sementes para um modelo. Devolve (uids, y, P[n, s])."""
    d = df[(df.modelo == modelo) & (df.conjunto == conjunto)]
    if d.empty:
        raise ValueError(f"sem previsões para modelo={modelo!r}, conjunto={conjunto!r}")
    piv = d.pivot_table(index="StudyInstanceUID", columns="semente", values="prob")
    y = d.groupby("StudyInstanceUID").y_true.first().loc[piv.index].values
    return piv.index.values, y.astype(int), piv.values


def _auc_media_sementes(y, P):
    return float(np.mean([roc_auc_score(y, P[:, s]) for s in range(P.shape[1])]))


def bootstrap_ic(y, P, n_boot=2000, seed=SEED, alpha=0.05):
    """IC por bootstrap de pacientes da AUC média entre sementes. P: (n, s)."""
    P = np.asarray(P).reshape(len(y), -1)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    while len(vals) < n_boot:
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue  # réplica sem uma das classes
        vals.append(_auc_media_sementes(y[idx], P[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_pareado(y, P1, P2, n_boot=2000, seed=SEED):
    """Diferença de AUC média (modelo 1 - modelo 2) com IC 95 % e p bilateral."""
    P1 = np.asarray(P1).reshape(len(y), -1)
    P2 = np.asarray(P2).reshape(len(y), -1)
    obs = _auc_media_sementes(y, P1) - _auc_media_sementes(y, P2)
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    while len(d) < n_boot:
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        d.append(_auc_media_sementes(y[idx], P1[idx]) - _auc_media_sementes(y[idx], P2[idx]))
    d = np.array(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diferenca": float(obs), "ic_inf": float(lo), "ic_sup": float(hi),
            "p": float(min(1.0, max(p, 1.0 / n_boot)))}


# ----------------------------------------------------------------------------
# Limiar e métricas no limiar
# ----------------------------------------------------------------------------
def limiar_youden(y_val, p_val) -> float:
    """Limiar que maximiza sensibilidade + especificidade - 1, escolhido na VALIDAÇÃO."""
    fpr, tpr, thr = roc_curve(y_val, p_val)
    return float(thr[np.argmax(tpr - fpr)])


def metricas_no_limiar(y, p, limiar) -> dict:
    pred = (np.asarray(p) >= limiar).astype(int)
    y = np.asarray(y).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return {"sensibilidade": tp / max(tp + fn, 1), "especificidade": tn / max(tn + fp, 1),
            "f1": f1_score(y, pred, zero_division=0),
            "acuracia_balanceada": balanced_accuracy_score(y, pred),
            "VP": tp, "VN": tn, "FP": fp, "FN": fn}


# ----------------------------------------------------------------------------
# Tabelas para o artigo
# ----------------------------------------------------------------------------
def tabela_principal(df: pd.DataFrame, n_boot=2000) -> pd.DataFrame:
    """Uma linha por modelo: AUC média ± dp entre sementes, IC 95 % bootstrap,
    e métricas no limiar (escolhido na validação, média das sementes)."""
    linhas = []
    for modelo in df.modelo.unique():
        _, y, P = _matriz(df, modelo, "teste")
        aucs = [roc_auc_score(y, P[:, s]) for s in range(P.shape[1])]
        lo, hi = bootstrap_ic(y, P, n_boot=n_boot)
        linha = {"modelo": modelo, "n_sementes": P.shape[1], "n_pacientes": len(y),
                 "auc_media": float(np.mean(aucs)),
                 "auc_dp": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                 "ic95_inf": lo, "ic95_sup": hi}
        try:
            _, yv, Pv = _matriz(df, modelo, "val")
            lim = limiar_youden(yv, Pv.mean(axis=1))
            linha["limiar_val"] = lim
            linha.update(metricas_no_limiar(y, P.mean(axis=1), lim))
        except ValueError:
            pass  # sem validação: só AUC
        linhas.append(linha)
    return pd.DataFrame(linhas).sort_values("auc_media", ascending=False).reset_index(drop=True)


def comparar(df: pd.DataFrame, modelo_1: str, modelo_2: str, n_boot=2000) -> dict:
    """Comparação pareada entre dois modelos no conjunto de teste."""
    u1, y1, P1 = _matriz(df, modelo_1)
    u2, y2, P2 = _matriz(df, modelo_2)
    comuns = np.intersect1d(u1, u2)
    i1 = pd.Index(u1).get_indexer(comuns); i2 = pd.Index(u2).get_indexer(comuns)
    y = y1[i1]
    dl = delong_pareado(y, P1[i1].mean(axis=1), P2[i2].mean(axis=1))
    bs = bootstrap_pareado(y, P1[i1], P2[i2], n_boot=n_boot)
    return {"modelo_1": modelo_1, "modelo_2": modelo_2, "n_pacientes": len(comuns),
            "dif_auc": bs["diferenca"], "ic95_inf": bs["ic_inf"], "ic95_sup": bs["ic_sup"],
            "p_bootstrap": bs["p"], "p_delong": dl["p"]}


def corrigir_holm(pvals):
    """Correção de Holm-Bonferroni para várias comparações."""
    p = np.asarray(pvals, dtype=float)
    ordem = np.argsort(p)
    m = len(p)
    ajust = np.empty(m)
    acum = 0.0
    for r, i in enumerate(ordem):
        acum = max(acum, min(1.0, (m - r) * p[i]))
        ajust[i] = acum
    return ajust


def subgrupos_rsna(train_csv: pd.DataFrame) -> pd.DataFrame:
    """Subgrupos clinicamente motivados a partir dos rótulos C1-C7 do desafio."""
    t = train_csv.copy()
    niveis = [f"C{i}" for i in range(1, 8)]
    t["n_vertebras"] = t[niveis].sum(axis=1)
    t["faixa_n_vertebras"] = pd.cut(t.n_vertebras, [-1, 0, 1, 2, 7],
                                    labels=["0", "1", "2", ">=3"]).astype(str)
    alta = t[["C1", "C2"]].max(axis=1).astype(bool)
    baixa = t[niveis[2:]].max(axis=1).astype(bool)
    t["nivel"] = np.select([alta & baixa, alta, baixa], ["C1-C2 e C3-C7", "C1-C2", "C3-C7"],
                           default="sem fratura")
    return t[["StudyInstanceUID", "n_vertebras", "faixa_n_vertebras", "nivel"]]


def analise_estratificada(df: pd.DataFrame, modelo: str, grupos: pd.DataFrame,
                          coluna: str, n_boot=1000) -> pd.DataFrame:
    """AUC por subgrupo: positivos do subgrupo contra TODOS os negativos do teste,
    e sensibilidade no limiar da validação. Usa a média das sementes."""
    uids, y, P = _matriz(df, modelo)
    p = P.mean(axis=1)
    g = grupos.set_index("StudyInstanceUID").loc[uids, coluna].values
    try:
        _, yv, Pv = _matriz(df, modelo, "val")
        lim = limiar_youden(yv, Pv.mean(axis=1))
    except ValueError:
        lim = 0.5
    neg = y == 0
    linhas = []
    for valor in sorted(set(g[y == 1])):
        pos = (y == 1) & (g == valor)
        sel = pos | neg
        yy, pp = y[sel], p[sel]
        lo, hi = bootstrap_ic(yy, pp[:, None], n_boot=n_boot)
        linhas.append({"subgrupo": f"{coluna}={valor}", "n_positivos": int(pos.sum()),
                       "auc": roc_auc_score(yy, pp), "ic95_inf": lo, "ic95_sup": hi,
                       "sensibilidade": float((p[pos] >= lim).mean())})
    return pd.DataFrame(linhas)
