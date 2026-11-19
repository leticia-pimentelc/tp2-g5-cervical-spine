import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from explicabilidade import *

def lote(n, g):
    x = torch.randn(n, 3, 64, 64, generator=g) * 0.3; y = torch.randint(0, 2, (n,), generator=g); caixas = []
    for i in range(n):
        cx, cy = torch.randint(8, 50, (2,), generator=g).tolist()
        if y[i]: x[i, :, cy:cy+6, cx:cx+6] += 2.5
        caixas.append((cx, cy, 6, 6))
    return x, y.float(), caixas

class Rede(nn.Module):
    def __init__(s):
        super().__init__()
        s.c1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        s.c2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        s.c3 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        s.fc = nn.Linear(32, 1)
    def forward(s, x):
        return s.fc(s.c3(s.c2(s.c1(x))).amax(dim=(2, 3))).squeeze(1)

_cache = {}
def modelo_treinado():
    if "m" in _cache: return _cache["m"]
    torch.manual_seed(0); g = torch.Generator().manual_seed(0); m = Rede()
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(150):
        x, y, _ = lote(32, g); loss = nn.functional.binary_cross_entropy_with_logits(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    _cache["m"] = m.eval(); return _cache["m"]

def test_modelo_aprende():
    m = modelo_treinado(); x, y, _ = lote(200, torch.Generator().manual_seed(9))
    with torch.no_grad(): assert roc_auc_score(y, m(x)) > 0.95

def test_mapas_apontam_para_a_caixa():
    m = modelo_treinado(); g = torch.Generator().manual_seed(5); acertos_gc = acertos_ig = n = 0; razoes = []
    while n < 15:
        x, y, cx = lote(1, g)
        if y[0] == 0: continue
        n += 1
        gc = mapa_gradcam(m, m.c3, x); ig = mapa_integrated_gradients(m, x, base=torch.zeros_like(x))
        acertos_gc += metricas_localizacao(gc, cx[0])["pointing_game"]
        r = metricas_localizacao(ig, cx[0]); acertos_ig += r["pointing_game"]; razoes.append(r["razao_energia_area"])
    assert acertos_gc / n >= 0.8 and acertos_ig / n >= 0.8 and np.mean(razoes) > 3

def test_sanidade_cascata_derruba_similaridade():
    m = modelo_treinado(); g = torch.Generator().manual_seed(7)
    while True:
        x, y, _ = lote(1, g)
        if y[0] == 1: break
    gerar = lambda mod, xx: mapa_gradcam(mod, mod.c3, xx)
    t = sanidade_cascata(m, ["fc", "c3", "c2", "c1"], gerar, x)
    assert t.spearman.iloc[0] == 1.0 and t.spearman.iloc[-1] < 0.5 and t.ssim.iloc[-1] < 0.6

def test_selecao_criterio_fixo(tmp_path):
    r = np.random.default_rng(0); n = 100
    prev = pd.DataFrame({"StudyInstanceUID": [f"u{i}" for i in range(n)], "y_true": r.integers(0, 2, n), "prob": r.random(n)})
    sel = selecionar_exemplos(prev, 0.5, k=2, caminho_csv=tmp_path / "sel.csv")
    assert set(sel.categoria) == set(CATEGORIAS) and (tmp_path / "sel.csv").exists()
    vp = prev[(prev.y_true == 1) & (prev.prob >= .5)].prob.nlargest(2)
    assert set(sel[sel.criterio == "VP de maior confiança"].prob) == set(vp)

def test_reshape_vit():
    t = torch.randn(2, 1 + 4 + 16 * 16, 384)
    assert reshape_vit(5)(t).shape == (2, 384, 16, 16)
