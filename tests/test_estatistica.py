import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from estatistica import *

def dados(n=400, sep=1.0, seed=0):
    r = np.random.default_rng(seed); y = r.integers(0, 2, n)
    return y, y * sep + r.normal(size=n)

def test_auc_delong_igual_sklearn():
    y, p = dados()
    p = np.round(p, 1)  # com empates
    assert abs(delong_ic(y, p)[0] - roc_auc_score(y, p)) < 1e-10

def test_variancia_delong_proxima_do_bootstrap():
    y, p = dados(600)
    auc, lo, hi = delong_ic(y, p)
    blo, bhi = bootstrap_ic(y, p[:, None], n_boot=1500)
    assert abs((hi - lo) - (bhi - blo)) < 0.02

def test_delong_mesmo_modelo_p1():
    y, p = dados()
    assert delong_pareado(y, p, p)["p"] == 1.0

def test_delong_detecta_diferenca():
    r = np.random.default_rng(1); y = r.integers(0, 2, 500)
    bom = y * 1.5 + r.normal(size=500); ruim = y * 0.3 + r.normal(size=500)
    assert delong_pareado(y, bom, ruim)["p"] < 0.001

def test_erro_tipo_I_delong():
    # dois modelos igualmente bons e correlacionados: p<0.05 em ~5% das vezes
    rej = 0
    for s in range(300):
        r = np.random.default_rng(100 + s); y = r.integers(0, 2, 200)
        base = y * 1.0 + r.normal(size=200)
        rej += delong_pareado(y, base + r.normal(0, .7, 200), base + r.normal(0, .7, 200))["p"] < 0.05
    assert 0.02 < rej / 300 < 0.09, rej / 300

def test_tabela_e_comparacao_com_sementes():
    r = np.random.default_rng(3); n = 300
    y = r.integers(0, 2, n); uids = [f"u{i}" for i in range(n)]
    linhas = []
    for modelo, sep in [("cnn", 1.2), ("rf_ap1", 0.5)]:
        for s in (0, 1, 2):
            for conj in ("val", "teste"):
                p = 1 / (1 + np.exp(-(y * sep + r.normal(size=n))))
                linhas += [dict(StudyInstanceUID=u, y_true=t, prob=q, modelo=modelo, semente=s, conjunto=conj)
                           for u, t, q in zip(uids, y, p)]
    df = pd.DataFrame(linhas)
    tab = tabela_principal(df, n_boot=300)
    assert list(tab.modelo) == ["cnn", "rf_ap1"]
    assert (tab.ic95_inf < tab.auc_media).all() and (tab.auc_media < tab.ic95_sup).all()
    c = comparar(df, "cnn", "rf_ap1", n_boot=300)
    assert c["dif_auc"] > 0 and c["p_delong"] < 0.01 and c["p_bootstrap"] < 0.01

def test_estratificada_e_holm():
    r = np.random.default_rng(4); n = 300
    t = pd.DataFrame({"StudyInstanceUID": [f"u{i}" for i in range(n)]})
    for i in range(1, 8): t[f"C{i}"] = (r.random(n) < 0.15).astype(int)
    y = t[[f"C{i}" for i in range(1, 8)]].max(axis=1).values
    g = subgrupos_rsna(t)
    assert set(g.nivel) <= {"sem fratura", "C1-C2", "C3-C7", "C1-C2 e C3-C7"}
    df = pd.DataFrame(dict(StudyInstanceUID=t.StudyInstanceUID, y_true=y, prob=y + r.normal(size=n),
                           modelo="m", semente=0, conjunto="teste"))
    est = analise_estratificada(df, "m", g, "nivel", n_boot=200)
    assert len(est) >= 2 and est.n_positivos.sum() == y.sum()
    assert np.allclose(corrigir_holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
