from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analise import CORES_ESPECIES


def aplicar_pca(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    medidas = df.drop(columns="especie")
    padronizadas = StandardScaler().fit_transform(medidas)
    pca = PCA()
    valores = pca.fit_transform(padronizadas)

    nomes = [f"CP{numero}" for numero in range(1, len(medidas.columns) + 1)]
    componentes = pd.DataFrame(valores, columns=nomes, index=df.index)
    componentes["especie"] = df["especie"]
    variancia = pd.Series(pca.explained_variance_ratio_, index=nomes)
    pesos = pd.DataFrame(pca.components_, index=nomes, columns=medidas.columns)

    mostrar_resultado_pca(variancia, pesos)
    return componentes, variancia


def mostrar_resultado_pca(variancia: pd.Series, pesos: pd.DataFrame) -> None:
    print("\n**PCA: variância explicada (%): ")
    print(
        pd.DataFrame(
            {
                "variancia": (variancia * 100).round(1),
                "acumulada": (variancia.cumsum() * 100).round(1),
            }
        )
    )
    print("\n**PCA: peso de cada medida nas componentes: ")
    print(pesos.round(2))


def grafico_pca(componentes: pd.DataFrame, variancia: pd.Series, pasta_saida: Path) -> None:
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho_arquivo = pasta_saida / "pca.png"

    figura, eixo = plt.subplots(figsize=(7, 5))
    for especie, grupo in componentes.groupby("especie"):
        eixo.scatter(
            grupo["CP1"],
            grupo["CP2"],
            color=CORES_ESPECIES[especie],
            label=especie,
            s=45,
            edgecolors="white",
            linewidths=0.8,
        )
    eixo.set_xlabel(f"CP1 ({variancia['CP1'] * 100:.1f}% da variância)")
    eixo.set_ylabel(f"CP2 ({variancia['CP2'] * 100:.1f}% da variância)")
    eixo.set_title("PCA: duas primeiras componentes principais")
    eixo.grid(alpha=0.3)
    eixo.legend(title="Espécie")
    figura.tight_layout()
    figura.savefig(caminho_arquivo, dpi=150)
    plt.close(figura)
    print(f"Gráfico salvo em {caminho_arquivo}")
