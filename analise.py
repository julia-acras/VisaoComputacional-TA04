from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CORES_ESPECIES = {
    "Iris-setosa": "hotpink",
    "Iris-versicolor": "yellow",
    "Iris-virginica": "green",
}

def modas(serie: pd.Series) -> list[float]:
    return serie.mode().tolist()


def tabela_estatisticas(medidas: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "media": medidas.mean().round(2),
            "desvio_padrao": medidas.std().round(2),
            "moda": [modas(medidas[coluna]) for coluna in medidas.columns],
        }
    )


def estatisticas_descritivas(df: pd.DataFrame) -> None:
    print("\n**Estatísticas gerais: ")
    print(tabela_estatisticas(df.drop(columns="especie")))

    for especie, grupo in df.groupby("especie"):
        print(f"\n**Estatísticas de {especie}: ")
        print(tabela_estatisticas(grupo.drop(columns="especie")))


def frequencia_categorias(df: pd.DataFrame) -> None:
    frequencia = pd.DataFrame(
        {
            "frequencia": df["especie"].value_counts(),
            "percentual": (df["especie"].value_counts(normalize=True) * 100).round(1),
        }
    )
    print("\n**Frequência das espécies: ")
    print(frequencia)


def grafico_dispersao(df: pd.DataFrame, coluna_x: str, coluna_y: str, caminho_arquivo: Path) -> None:
    figura, eixo = plt.subplots(figsize=(7, 5))
    for especie, grupo in df.groupby("especie"):
        eixo.scatter(
            grupo[coluna_x],
            grupo[coluna_y],
            color=CORES_ESPECIES[especie],
            label=especie,
            s=45,
            edgecolors="white",
            linewidths=0.8,
        )
    eixo.set_xlabel(f"{coluna_x} (cm)")
    eixo.set_ylabel(f"{coluna_y} (cm)")
    eixo.set_title(f"{coluna_x} × {coluna_y}")
    eixo.grid(alpha=0.3)
    eixo.legend(title="Espécie")
    figura.tight_layout()
    figura.savefig(caminho_arquivo, dpi=150)
    plt.close(figura)


def correlacao_por_especie(df: pd.DataFrame, coluna_x: str, coluna_y: str) -> None:
    print(f"Correlação geral: {df[coluna_x].corr(df[coluna_y]):.2f}")
    for especie, grupo in df.groupby("especie"):
        print(f"Correlação em {especie}: {grupo[coluna_x].corr(grupo[coluna_y]):.2f}")


def analisar_relacao(df: pd.DataFrame, coluna_x: str, coluna_y: str, nome: str, pasta_saida: Path) -> None:
    print(f"\n**Relação {coluna_x} × {coluna_y} ({nome}): ")
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho_arquivo = pasta_saida / f"dispersao_{nome}.png"
    grafico_dispersao(df, coluna_x, coluna_y, caminho_arquivo)
    correlacao_por_especie(df, coluna_x, coluna_y)
    print(f"Gráfico salvo em {caminho_arquivo}")


def grafico_matriz_dispersao(df: pd.DataFrame, caminho_arquivo: Path) -> None:
    medidas = df.drop(columns="especie").columns
    figura, eixos = plt.subplots(4, 4, figsize=(12, 12))
    for linha, coluna_y in enumerate(medidas):
        for coluna, coluna_x in enumerate(medidas):
            eixo = eixos[linha, coluna]
            for especie, grupo in df.groupby("especie"):
                cor = CORES_ESPECIES[especie]
                if linha == coluna:
                    eixo.hist(
                        grupo[coluna_x],
                        bins=15,
                        range=(df[coluna_x].min(), df[coluna_x].max()),
                        color=cor,
                        histtype="step",
                        linewidth=1.8,
                    )
                else:
                    eixo.scatter(
                        grupo[coluna_x],
                        grupo[coluna_y],
                        color=cor,
                        label=especie,
                        s=15,
                        edgecolors="white",
                        linewidths=0.4,
                    )
            if linha == len(medidas) - 1:
                eixo.set_xlabel(f"{coluna_x} (cm)")
            if coluna == 0:
                eixo.set_ylabel(f"{coluna_y} (cm)")
    marcadores, rotulos = eixos[0, 1].get_legend_handles_labels()
    figura.legend(marcadores, rotulos, loc="upper center", ncol=3, title="Espécie")
    figura.tight_layout(rect=(0, 0, 1, 0.95))
    figura.savefig(caminho_arquivo, dpi=150)
    plt.close(figura)


def grafico_boxplots(df: pd.DataFrame, caminho_arquivo: Path) -> None:
    medidas = df.drop(columns="especie").columns
    especies = list(CORES_ESPECIES)
    figura, eixos = plt.subplots(1, 4, figsize=(16, 4.5))
    for eixo, medida in zip(eixos, medidas):
        valores = [df.loc[df["especie"] == especie, medida] for especie in especies]
        caixas = eixo.boxplot(
            valores,
            patch_artist=True,
            tick_labels=[especie.removeprefix("Iris-") for especie in especies],
            medianprops={"color": "black"},
        )
        for caixa, especie in zip(caixas["boxes"], especies):
            caixa.set_facecolor(CORES_ESPECIES[especie])
        eixo.set_title(medida)
        eixo.set_ylabel("cm")
        eixo.grid(axis="y", alpha=0.3)
    figura.tight_layout()
    figura.savefig(caminho_arquivo, dpi=150)
    plt.close(figura)


def distribuicao_quatro_dimensoes(df: pd.DataFrame, pasta_saida: Path) -> None:
    print("\n**Distribuição nas quatro dimensões: ")
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho_matriz = pasta_saida / "matriz_dispersao.png"
    caminho_boxplots = pasta_saida / "boxplots_especies.png"
    grafico_matriz_dispersao(df, caminho_matriz)
    grafico_boxplots(df, caminho_boxplots)
    print(f"Gráficos salvos em {caminho_matriz} e {caminho_boxplots}")
