"""Gera a grade visual de aumento de dados e a verificação quantitativa das caixas.

Roda no Kaggle (competição RSNA 2022 adicionada como Input):
    !git clone -q https://github.com/leticia-pimentelc/tp2-g5-cervical-spine ap2
    !cd ap2 && python scripts/gerar_grade_aumento.py

Saídas em /kaggle/working (ou --saida):
    grade_aumento.png / .pdf      figura para o artigo e o repositório
    verificacao_caixas.csv        % de aumentos em que a caixa de fratura fica inteira
    exemplos_grade.csv            pacientes/cortes usados (critério fixo, semente 42)

Critério de escolha dos exemplos (definido antes de gerar a figura):
    3 cortes com fratura sorteados de train_bounding_boxes.csv (semente 42,
    pacientes distintos) e 1 paciente sem fratura (semente 42, corte central).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from aumento import POLITICAS, Politica, caixa_inteira_visivel, desnormalizar  # noqa: E402

SEED = 42


def checar_descompressor():
    try:
        import pylibjpeg  # noqa: F401
        import libjpeg  # noqa: F401
    except ImportError:
        sys.exit("Instale antes: pip install pylibjpeg pylibjpeg-libjpeg e REINICIE o kernel.")


def ler_hu(caminho):
    import pydicom
    ds = pydicom.dcmread(caminho)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    return ds.pixel_array.astype(np.float32) * slope + intercept


def ler_25d(pasta_paciente, n):
    """Cortes n-1, n, n+1 (nome do arquivo = número do corte). Nas bordas, repete."""
    numeros = sorted(int(f.split(".")[0]) for f in os.listdir(pasta_paciente) if f.endswith(".dcm"))
    i = numeros.index(n)
    viz = [numeros[max(i - 1, 0)], numeros[i], numeros[min(i + 1, len(numeros) - 1)]]
    return torch.from_numpy(np.stack([ler_hu(os.path.join(pasta_paciente, f"{k}.dcm")) for k in viz]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/kaggle/input/competitions/rsna-2022-cervical-spine-fracture-detection")
    ap.add_argument("--saida", default="/kaggle/working")
    ap.add_argument("--tamanho", type=int, default=224)
    ap.add_argument("--n_amostras", type=int, default=3, help="aumentos por política na grade")
    ap.add_argument("--n_caixas", type=int, default=200, help="caixas na verificação quantitativa")
    args = ap.parse_args()
    if not os.path.exists(args.base):
        args.base = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection"
    checar_descompressor()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    os.makedirs(args.saida, exist_ok=True)
    caixas = pd.read_csv(os.path.join(args.base, "train_bounding_boxes.csv"))
    train = pd.read_csv(os.path.join(args.base, "train.csv"))
    img_dir = os.path.join(args.base, "train_images")

    # ---- exemplos com critério fixo -------------------------------------------
    pos = (caixas.sample(frac=1, random_state=SEED)
           .drop_duplicates("StudyInstanceUID").head(3).copy())
    pos["tipo"] = "fratura"
    neg_uid = train[train.patient_overall == 0].sample(1, random_state=SEED).StudyInstanceUID.iloc[0]
    nums = sorted(int(f.split(".")[0]) for f in os.listdir(os.path.join(img_dir, neg_uid)))
    neg = pd.DataFrame([{"StudyInstanceUID": neg_uid, "slice_number": nums[len(nums) // 2],
                         "x": np.nan, "y": np.nan, "width": np.nan, "height": np.nan,
                         "tipo": "sem fratura"}])
    exemplos = pd.concat([pos, neg], ignore_index=True)
    exemplos.to_csv(os.path.join(args.saida, "exemplos_grade.csv"), index=False)

    # ---- grade -------------------------------------------------------------------
    pols = {p: Politica(p, args.tamanho) for p in POLITICAS}
    colunas = [("sem_aumento", 0)] + [("principal", k) for k in range(args.n_amostras)] \
        + [("alternativa", k) for k in range(args.n_amostras)]
    fig, axes = plt.subplots(len(exemplos), len(colunas),
                             figsize=(1.6 * len(colunas), 1.7 * len(exemplos)))
    torch.manual_seed(SEED)
    for i, ex in exemplos.iterrows():
        hu = ler_25d(os.path.join(img_dir, ex.StudyInstanceUID), int(ex.slice_number))
        tem_caixa = ex.tipo == "fratura"
        cx = [[ex.x, ex.y, ex.width, ex.height]] if tem_caixa else [[0, 0, 1, 1]]
        for j, (nome, k) in enumerate(colunas):
            x, bb = pols[nome](hu, cx)
            ax = axes[i, j]
            ax.imshow(desnormalizar(x)[1].numpy(), cmap="gray", vmin=0, vmax=1)
            if tem_caixa:
                bx, by, bw, bh = bb[0].tolist()
                ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec="red", lw=0.8))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                titulo = {"sem_aumento": "Original", "principal": "Principal",
                          "alternativa": "Alternativa"}[nome]
                ax.set_title(titulo if nome == "sem_aumento" else f"{titulo} #{k + 1}", fontsize=7)
            if j == 0:
                ax.set_ylabel(ex.tipo, fontsize=7)
    plt.tight_layout(pad=0.2)
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(args.saida, f"grade_aumento.{ext}"), dpi=300, bbox_inches="tight")
    print("Grade salva em", os.path.join(args.saida, "grade_aumento.png"))

    # ---- verificação quantitativa das caixas -------------------------------------
    amostra = caixas.sample(min(args.n_caixas, len(caixas)), random_state=SEED)
    linhas = []
    for nome in POLITICAS:
        fr = [caixa_inteira_visivel([r.x, r.y, r.width, r.height], (512, 512), pols[nome],
                                    n=50, seed=SEED) for r in amostra.itertuples()]
        linhas.append({"politica": nome, "n_caixas": len(fr),
                       "media_caixa_inteira": float(np.mean(fr)),
                       "pior_caixa": float(np.min(fr))})
    tab = pd.DataFrame(linhas)
    tab.to_csv(os.path.join(args.saida, "verificacao_caixas.csv"), index=False)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
