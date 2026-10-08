from pathlib import Path

import pandas as pd

NOMES_COLUNAS = [
    "comprimento_sepala",
    "largura_sepala",
    "comprimento_petala",
    "largura_petala",
    "especie",
]

def carregar_iris(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho, header=None, names=NOMES_COLUNAS)
