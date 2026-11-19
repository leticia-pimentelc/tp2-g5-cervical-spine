"""Aumento de dados para a AP2 (G5 — RSNA 2022 Cervical Spine Fracture Detection).

Responsável: Maria Letícia Pimentel Carlos.

Formato de entrada (contrato com o pipeline de dados do grupo)
---------------------------------------------------------------
Cada amostra é um tensor float32 de forma (3, H, W) em Unidades Hounsfield (HU):
três cortes axiais adjacentes (n-1, n, n+1), estratégia 2.5D. Trabalhar em HU
(e não na imagem já janelada) permite que a política alternativa varie a janela
de forma fisicamente plausível.

Ordem das operações em todas as políticas
  1. transformações geométricas (em HU, preenchendo com ar = -1024 HU)
  2. redimensionamento para `tamanho` x `tamanho`
  3. janelamento ósseo (centro 400 HU, largura 1800 HU) -> [0, 1]
  4. transformações fotométricas (só na política alternativa)
  5. normalização do pré-treino (média e desvio do ImageNet)

O aumento é aplicado SOMENTE no treino. Validação e teste usam `sem_aumento`.

Justificativa clínica de cada transformação (resumo para o artigo)
------------------------------------------------------------------
- Espelhamento horizontal (esquerda-direita): o rótulo `patient_overall`
  (fratura em qualquer vértebra C1-C7) não é lateralizado, e a coluna cervical é
  aproximadamente simétrica no plano axial. NÃO usamos espelhamento vertical,
  que inverteria anterior/posterior (via aérea atrás do corpo vertebral), uma
  anatomia que não ocorre.
- Rotação de até 10 graus: simula a inclinação da cabeça/pescoço no tomógrafo.
  Não cria nem apaga descontinuidade cortical; a fratura continua no mesmo osso.
- Escala de 0,9 a 1,1 e translação de até 5 %: simulam diferenças de campo de
  visão (FOV) e de centralização entre as 12 instituições do conjunto. Os
  limites mantêm o canal vertebral dentro da imagem (verificado com as caixas
  de fratura em `caixa_inteira_visivel`).
- Variação da janela (só na alternativa: centro +-50 HU, largura +-10 %):
  simula diferenças de calibração e de kernel de reconstrução entre
  equipamentos. Os limites mantêm o osso cortical (> ~700 HU) claramente acima
  do tecido mole (< ~100 HU), preservando a linha de fratura.
- Ruído gaussiano leve (só na alternativa, sigma = 0,02 na escala [0, 1],
  cerca de 36 HU): simula TC de baixa dose. É a transformação de maior risco
  (pode mascarar fraturas sem desvio), e por isso fica na política alternativa,
  para que o efeito seja medido, e não na principal.
- NÃO usamos ajuste de brilho/contraste genérico (ColorJitter): o enunciado
  alerta que alterar contraste pode apagar achados sutis.

O conjunto de treino é quase balanceado (961 com fratura e 1.058 sem, 47,6 %),
portanto o aumento serve à REGULARIZAÇÃO, e não ao rebalanceamento de classes.
"""
from __future__ import annotations

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
JANELA_OSSO = (400.0, 1800.0)  # (centro, largura) em HU, a mesma da AP1
AR_HU = -1024.0  # valor de preenchimento nas transformações geométricas
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIO_IMAGENET = (0.229, 0.224, 0.225)
POLITICAS = ("sem_aumento", "principal", "alternativa")


def janelar(hu: torch.Tensor, centro: float, largura: float) -> torch.Tensor:
    """Converte HU para [0, 1] com a janela (centro, largura)."""
    minimo = centro - largura / 2.0
    return ((hu - minimo) / largura).clamp(0.0, 1.0)


class JanelaHU(torch.nn.Module):
    """Janelamento em HU; com `variacao_*` > 0 sorteia a janela a cada chamada."""

    def __init__(self, centro=JANELA_OSSO[0], largura=JANELA_OSSO[1],
                 variacao_centro=0.0, variacao_largura_rel=0.0):
        super().__init__()
        self.centro, self.largura = float(centro), float(largura)
        self.vc, self.vl = float(variacao_centro), float(variacao_largura_rel)

    def forward(self, hu: torch.Tensor) -> torch.Tensor:
        c, w = self.centro, self.largura
        if self.vc > 0:
            c += float(torch.empty(1).uniform_(-self.vc, self.vc))
        if self.vl > 0:
            w *= 1.0 + float(torch.empty(1).uniform_(-self.vl, self.vl))
        return janelar(hu, c, w)


class RuidoGaussiano(torch.nn.Module):
    """Ruído aditivo N(0, sigma) na escala [0, 1], aplicado com probabilidade p."""

    def __init__(self, sigma=0.02, p=0.5):
        super().__init__()
        self.sigma, self.p = float(sigma), float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigma <= 0 or float(torch.rand(1)) >= self.p:
            return x
        return (x + torch.randn_like(x) * self.sigma).clamp(0.0, 1.0)


def _geometrica(politica: str, tamanho: int) -> T.Compose:
    redimensiona = T.Resize((tamanho, tamanho), antialias=True)
    if politica == "sem_aumento":
        return T.Compose([redimensiona])
    # principal e alternativa compartilham a mesma parte geométrica: assim a
    # comparação principal x alternativa isola o efeito fotométrico.
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1),
                       fill=AR_HU),
        redimensiona,
    ])


def _fotometrica(politica: str) -> torch.nn.Module:
    if politica == "alternativa":
        return T.Compose([
            JanelaHU(variacao_centro=50.0, variacao_largura_rel=0.10),
            RuidoGaussiano(sigma=0.02, p=0.5),
        ])
    return JanelaHU()


class Politica:
    """Política de aumento completa. Uso: `x = Politica('principal', 224)(hu)`.

    Com `caixas` (formato XYWH, em pixels da imagem original) devolve também as
    caixas transformadas, o que permite verificar que a fratura continua visível.
    """

    def __init__(self, nome: str, tamanho: int = 224, normalizar: bool = True):
        if nome not in POLITICAS:
            raise ValueError(f"política desconhecida: {nome}; use {POLITICAS}")
        self.nome, self.tamanho = nome, tamanho
        self.geo = _geometrica(nome, tamanho)
        self.foto = _fotometrica(nome)
        self.norm = (T.Normalize(MEDIA_IMAGENET, DESVIO_IMAGENET)
                     if normalizar else torch.nn.Identity())

    def __call__(self, hu: torch.Tensor, caixas=None):
        if hu.ndim == 2:
            hu = hu.unsqueeze(0)
        if hu.shape[0] == 1:  # um único corte: replica nos 3 canais
            hu = hu.repeat(3, 1, 1)
        img = tv_tensors.Image(hu.float())
        if caixas is None:
            x = self.geo(img)
            return self.norm(self.foto(x.as_subclass(torch.Tensor)))
        bb = tv_tensors.BoundingBoxes(torch.as_tensor(caixas, dtype=torch.float32).reshape(-1, 4),
                                      format="XYWH", canvas_size=tuple(hu.shape[-2:]))
        x, bb = self.geo(img, bb)
        x = self.norm(self.foto(x.as_subclass(torch.Tensor)))
        return x, bb.as_subclass(torch.Tensor)

    def __repr__(self):
        return f"Politica({self.nome!r}, tamanho={self.tamanho})"


def desnormalizar(x: torch.Tensor) -> torch.Tensor:
    """Desfaz a normalização ImageNet (para visualizar)."""
    m = torch.tensor(MEDIA_IMAGENET).view(3, 1, 1)
    s = torch.tensor(DESVIO_IMAGENET).view(3, 1, 1)
    return (x * s + m).clamp(0, 1)


def caixa_inteira_visivel(caixa_xywh, tamanho_original, politica: Politica,
                          n: int = 500, seed: int = 42) -> float:
    """Sorteia `n` aumentos e devolve a fração deles em que a caixa de fratura
    continua INTEIRA dentro da imagem (a caixa transformada não encosta na
    borda, ou seja, não foi cortada). Evidência quantitativa de que a política
    não remove o achado que define o rótulo."""
    torch.manual_seed(seed)
    h, w = tamanho_original
    hu = torch.zeros(3, h, w)
    s = politica.tamanho
    inteiras = 0
    for _ in range(n):
        _, bb = politica(hu, [caixa_xywh])
        x, y, bw, bh = bb[0].tolist()
        tol = 0.5  # meio pixel
        inteiras += int(x > tol and y > tol and x + bw < s - tol and y + bh < s - tol
                        and bw > 0 and bh > 0)
    return inteiras / n
