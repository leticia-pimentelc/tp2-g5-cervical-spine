"""Explicabilidade e análise de erro da AP2 (G5). Responsável: Maria Letícia Pimentel Carlos.

Protocolo (exigências da Seção 6 do enunciado)
1. Seleção de exemplos ANTES de olhar os mapas (`selecionar_exemplos`): no
   conjunto de teste, com o limiar escolhido na validação, os k VP, FP, FN e VN
   de maior confiança e os k casos mais próximos do limiar. A lista é salva em
   CSV antes de qualquer mapa ser gerado.
2. Dois métodos de naturezas distintas:
   - Grad-CAM (Selvaraju et al., ICCV 2017), baseado em gradiente x ativação;
   - Integrated Gradients (Sundararajan et al., ICML 2017), atribuição por
     integração de gradientes ao longo de um caminho a partir de uma imagem base.
   Para ViT/DINOv2 o Grad-CAM usa `reshape_vit`, que reorganiza os tokens em grade.
3. Teste de sanidade por randomização em cascata (Adebayo et al., NeurIPS 2018):
   reinicializa as camadas do topo para a base e mede a similaridade com o mapa
   original (correlação de Spearman do valor absoluto e SSIM). Mapas confiáveis
   devem mudar muito quando os pesos são randomizados.
4. Métricas quantitativas com as caixas de fratura (`train_bounding_boxes.csv`):
   pointing game (o pico do mapa cai dentro da caixa?) e fração de energia
   dentro da caixa, comparada com a fração de área da caixa (o valor esperado
   para um mapa aleatório uniforme).

Contrato com o modelo: `modelo(x)` recebe (B, 3, H, W) já normalizado (ver
src/aumento.py, política `sem_aumento`) e devolve o logit de fratura (B,) ou
(B, 1). Para um modelo que agrega cortes por exame, os mapas são gerados no
corte: o corte com caixa de fratura (positivos anotados) ou o de maior escore
no exame (demais casos).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

# ----------------------------------------------------------------------------
# 1. Seleção de exemplos (antes de ver mapas)
# ----------------------------------------------------------------------------
CATEGORIAS = ("VP", "FP", "FN", "VN")


def categorizar(y, p, limiar):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= limiar).astype(int)
    return np.select([(y == 1) & (pred == 1), (y == 0) & (pred == 1),
                      (y == 1) & (pred == 0)], ["VP", "FP", "FN"], default="VN")


def selecionar_exemplos(prev: pd.DataFrame, limiar: float, k: int = 2,
                        caminho_csv: str | None = None) -> pd.DataFrame:
    """`prev`: colunas StudyInstanceUID, y_true, prob (média das sementes, teste).

    Critério fixo: para cada categoria, os k de maior confiança (VP/FP: maior
    prob; FN/VN: menor prob), mais os k casos mais próximos do limiar."""
    d = prev.copy()
    d["categoria"] = categorizar(d.y_true, d.prob, limiar)
    d["distancia_limiar"] = (d.prob - limiar).abs()
    partes = []
    for cat in CATEGORIAS:
        sub = d[d.categoria == cat]
        asc = cat in ("FN", "VN")
        partes.append(sub.sort_values("prob", ascending=asc).head(k)
                      .assign(criterio=f"{cat} de maior confiança"))
    partes.append(d.sort_values("distancia_limiar").head(k).assign(criterio="mais próximo do limiar"))
    sel = (pd.concat(partes).drop_duplicates("StudyInstanceUID")
           .assign(limiar=limiar).reset_index(drop=True))
    if caminho_csv:
        sel.to_csv(caminho_csv, index=False)
    return sel


# ----------------------------------------------------------------------------
# 2. Métodos de explicabilidade
# ----------------------------------------------------------------------------
class _LogitPositivo(torch.nn.Module):
    """Garante saída (B, 1) com o logit de fratura (exigência do pytorch-grad-cam)."""

    def __init__(self, modelo):
        super().__init__()
        self.modelo = modelo

    def forward(self, x):
        return self.modelo(x).reshape(x.shape[0], -1)[:, :1]


def reshape_vit(n_prefixo: int = 1):
    """Tokens (B, 1+R+N, C) -> mapa (B, C, h, w). `n_prefixo` = CLS + registradores
    (DINOv2 com registradores: 1 + 4)."""
    def f(t):
        t = t[:, n_prefixo:, :]
        lado = int(round(t.shape[1] ** 0.5))
        return t.reshape(t.shape[0], lado, lado, t.shape[2]).permute(0, 3, 1, 2)
    return f


def _normalizar_mapa(m: np.ndarray) -> np.ndarray:
    m = np.nan_to_num(m.astype(np.float64))
    m = m - m.min()
    return m / m.max() if m.max() > 0 else m


def mapa_gradcam(modelo, camada, x, reshape_transform=None) -> np.ndarray:
    """Grad-CAM para a classe 'fratura'. x: (1, 3, H, W). Devolve (H, W) em [0, 1]."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    with GradCAM(model=_LogitPositivo(modelo).eval(), target_layers=[camada],
                 reshape_transform=reshape_transform) as cam:
        m = cam(input_tensor=x, targets=[ClassifierOutputTarget(0)])[0]
    return _normalizar_mapa(m)


def mapa_integrated_gradients(modelo, x, base=None, passos: int = 32) -> np.ndarray:
    """Integrated Gradients (Captum). `base`: imagem de referência; por padrão a
    imagem toda preta (0 na janela óssea) já normalizada, ou seja, 'só ar'.
    Devolve |atribuição| somada nos canais, em [0, 1]."""
    from captum.attr import IntegratedGradients
    if base is None:
        from aumento import DESVIO_IMAGENET, MEDIA_IMAGENET
        m = torch.tensor(MEDIA_IMAGENET).view(1, 3, 1, 1)
        s = torch.tensor(DESVIO_IMAGENET).view(1, 3, 1, 1)
        base = ((torch.zeros_like(x) - m) / s).to(x.device)
    ig = IntegratedGradients(_LogitPositivo(modelo).eval())
    attr = ig.attribute(x, baselines=base, target=0, n_steps=passos,
                        internal_batch_size=passos)
    return _normalizar_mapa(attr.abs().sum(dim=1)[0].detach().cpu().numpy())


# ----------------------------------------------------------------------------
# 3. Teste de sanidade (randomização em cascata)
# ----------------------------------------------------------------------------
def _reinicializar(mod: torch.nn.Module, seed: int):
    torch.manual_seed(seed)
    for sub in mod.modules():
        if hasattr(sub, "reset_parameters"):
            sub.reset_parameters()


def _ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, data_range=1.0))


def sanidade_cascata(modelo, nomes_camadas, gerar_mapa, x, seed: int = 42) -> pd.DataFrame:
    """`nomes_camadas`: nomes (model.named_modules) do TOPO para a BASE, ex.
    ['fc', 'layer4', 'layer3', 'layer2', 'layer1', 'conv1'].
    `gerar_mapa(modelo, x) -> (H, W)`: função do método (ex. Grad-CAM na camada
    do modelo recebido). Devolve Spearman(|mapa|) e SSIM contra o mapa original."""
    original = gerar_mapa(modelo, x)
    linhas = [{"etapa": "original", "spearman": 1.0, "ssim": 1.0}]
    m = copy.deepcopy(modelo)
    modulos = dict(m.named_modules())
    for i, nome in enumerate(nomes_camadas):
        _reinicializar(modulos[nome], seed + i)
        mapa = gerar_mapa(m, x)
        rho = spearmanr(np.abs(original).ravel(), np.abs(mapa).ravel()).correlation
        linhas.append({"etapa": f"até {nome}", "spearman": float(np.nan_to_num(rho)),
                       "ssim": _ssim(original, mapa)})
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------
# 4. Métricas de localização com as caixas de fratura
# ----------------------------------------------------------------------------
def caixa_para_mapa(caixa_xywh, tamanho_original, tamanho_mapa):
    """Converte a caixa (pixels da imagem original) para a resolução do mapa."""
    (h0, w0), (h, w) = tamanho_original, tamanho_mapa
    x, y, bw, bh = caixa_xywh
    return x * w / w0, y * h / h0, bw * w / w0, bh * h / h0


def metricas_localizacao(mapa: np.ndarray, caixa_xywh) -> dict:
    """Pointing game e fração de energia na caixa (caixa já na resolução do mapa)."""
    h, w = mapa.shape
    x, y, bw, bh = caixa_xywh
    x0, y0 = max(int(np.floor(x)), 0), max(int(np.floor(y)), 0)
    x1, y1 = min(int(np.ceil(x + bw)), w), min(int(np.ceil(y + bh)), h)
    mascara = np.zeros_like(mapa, dtype=bool)
    mascara[y0:y1, x0:x1] = True
    pico = np.unravel_index(np.argmax(mapa), mapa.shape)
    energia = float(mapa[mascara].sum() / max(mapa.sum(), 1e-12))
    area = float(mascara.mean())
    return {"pointing_game": bool(mascara[pico]), "energia_na_caixa": energia,
            "fracao_area_caixa": area, "razao_energia_area": energia / max(area, 1e-12)}


# ----------------------------------------------------------------------------
# 5. Análise de erro
# ----------------------------------------------------------------------------
def tabela_erros(prev: pd.DataFrame, limiar: float, grupos: pd.DataFrame) -> pd.DataFrame:
    """Taxa de erro por subgrupo (ex. nível da fratura, nº de vértebras).
    `grupos`: saída de estatistica.subgrupos_rsna."""
    d = prev.merge(grupos, on="StudyInstanceUID", how="left")
    d["categoria"] = categorizar(d.y_true, d.prob, limiar)
    pos = d[d.y_true == 1]
    out = []
    for col in ("nivel", "faixa_n_vertebras"):
        for val, g in pos.groupby(col):
            out.append({"variavel": col, "valor": val, "n_positivos": len(g),
                        "falsos_negativos": int((g.categoria == "FN").sum()),
                        "taxa_fn": float((g.categoria == "FN").mean())})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# 6. Figura
# ----------------------------------------------------------------------------
def painel(linhas, caminho=None, titulo_colunas=("Corte (janela óssea)", "Grad-CAM", "Integrated Gradients")):
    """`linhas`: lista de dicts com 'imagem' (H, W em [0,1]), 'gradcam', 'ig',
    'rotulo' (ex. 'VP p=0,93') e opcional 'caixa' (XYWH na resolução do mapa)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(len(linhas), 3, figsize=(5.4, 1.85 * len(linhas)), squeeze=False)
    for i, ln in enumerate(linhas):
        for j, chave in enumerate((None, "gradcam", "ig")):
            ax = axes[i, j]
            ax.imshow(ln["imagem"], cmap="gray", vmin=0, vmax=1)
            if chave:
                ax.imshow(ln[chave], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            if ln.get("caixa") is not None:
                x, y, bw, bh = ln["caixa"]
                ax.add_patch(Rectangle((x, y), bw, bh, fill=False, ec="white", lw=0.8, ls="--"))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(titulo_colunas[j], fontsize=7)
            if j == 0:
                ax.set_ylabel(ln["rotulo"], fontsize=7)
    plt.tight_layout(pad=0.2)
    if caminho:
        plt.savefig(caminho, dpi=300, bbox_inches="tight")
    return fig
