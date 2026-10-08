from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

VALORES_K = range(1, 30, 2)

def dividir_dados(df: pd.DataFrame, semente: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns="especie")
    y = df["especie"]
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=semente, stratify=y)
    return X_treino, X_teste, y_treino, y_teste


def padronizar(X_treino: pd.DataFrame, X_teste: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    padronizador = StandardScaler().fit(X_treino)
    X_treino_padronizado = pd.DataFrame(padronizador.transform(X_treino), columns=X_treino.columns, index=X_treino.index)
    X_teste_padronizado = pd.DataFrame(padronizador.transform(X_teste), columns=X_teste.columns, index=X_teste.index)
    return X_treino_padronizado, X_teste_padronizado


def validacao_cruzada_por_k(X_treino: pd.DataFrame, y_treino: pd.Series, semente: int) -> pd.DataFrame:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=semente)
    linhas = []
    for k in VALORES_K:
        pipeline = Pipeline(
            [
                ("padronizador", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=k)),
            ]
        )
        acuracias = cross_val_score(pipeline, X_treino, y_treino, cv=folds)
        linhas.append({"k": k, "media": acuracias.mean(), "desvio": acuracias.std()})
    return pd.DataFrame(linhas).set_index("k")


def grafico_escolha_k(resultados: pd.DataFrame, melhor_k: int, caminho_arquivo: Path) -> None:
    media = resultados["media"] * 100
    desvio = resultados["desvio"] * 100
    figura, eixo = plt.subplots(figsize=(8, 4.5))
    eixo.fill_between(
        resultados.index,
        media - desvio,
        media + desvio,
        color="#2a78d6",
        alpha=0.15,
        label="± 1 desvio padrão",
    )
    eixo.plot(
        resultados.index,
        media,
        color="#2a78d6",
        linewidth=2,
        marker="o",
        label="acurácia média",
    )
    eixo.axvline(
        melhor_k,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"melhor k = {melhor_k}",
    )
    eixo.set_xticks(list(resultados.index))
    eixo.set_xlabel("k (número de vizinhos)")
    eixo.set_ylabel("Acurácia na validação cruzada (%)")
    eixo.set_title("Escolha do k: validação cruzada com 5 folds no treino")
    eixo.grid(alpha=0.3)
    eixo.legend(loc="lower left")
    figura.tight_layout()
    figura.savefig(caminho_arquivo, dpi=150)
    plt.close(figura)


def escolher_k(X_treino: pd.DataFrame, y_treino: pd.Series, semente: int, pasta_saida: Path) -> int:
    resultados = validacao_cruzada_por_k(X_treino, y_treino, semente)
    melhor_k = int(resultados["media"].round(6).idxmax())

    media = resultados.loc[melhor_k, "media"] * 100
    desvio = resultados.loc[melhor_k, "desvio"] * 100
    print("\n**Escolha do k: ")
    print(f"Melhor k: {melhor_k} (acurácia média {media:.1f}% ± {desvio:.1f}%)")

    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho_arquivo = pasta_saida / "escolha_k.png"
    grafico_escolha_k(resultados, melhor_k, caminho_arquivo)
    print(f"Gráfico salvo em {caminho_arquivo}")
    return melhor_k


def treinar_knn(X_treino: pd.DataFrame, y_treino: pd.Series, k: int) -> KNeighborsClassifier:
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_treino, y_treino)
    return modelo


def grafico_matriz_confusao(y_teste: pd.Series, previsoes: pd.Series, k: int, caminho_arquivo: Path) -> None:
    figura, eixo = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_teste, previsoes, cmap="Blues", ax=eixo)
    eixo.set_xlabel("Espécie prevista")
    eixo.set_ylabel("Espécie real")
    eixo.set_title(f"Matriz de confusão do K-NN (k = {k})")
    figura.tight_layout()
    figura.savefig(caminho_arquivo, dpi=150)
    plt.close(figura)


def avaliar_modelo(modelo: KNeighborsClassifier, X_teste: pd.DataFrame, y_teste: pd.Series, pasta_saida: Path) -> pd.Series:
    previsoes = pd.Series(modelo.predict(X_teste), index=X_teste.index, name="prevista")
    print(f"\n**Avaliação do K-NN (k = {modelo.n_neighbors}): ")
    print(f"Acurácia: {accuracy_score(y_teste, previsoes) * 100:.1f}%")
    print(classification_report(y_teste, previsoes))

    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho_arquivo = pasta_saida / "matriz_confusao.png"
    grafico_matriz_confusao(y_teste, previsoes, modelo.n_neighbors, caminho_arquivo)
    print(f"Gráfico salvo em {caminho_arquivo}")
    return previsoes


def amostras_erradas(X_teste: pd.DataFrame, y_teste: pd.Series, previsoes: pd.Series) -> pd.DataFrame:
    erradas = X_teste.copy()
    erradas["real"] = y_teste
    erradas["prevista"] = previsoes
    return erradas[erradas["real"] != erradas["prevista"]]
