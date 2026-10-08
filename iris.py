from pathlib import Path

from analise import (
    analisar_relacao,
    distribuicao_quatro_dimensoes,
    estatisticas_descritivas,
    frequencia_categorias,
)
from carregamento import carregar_iris
from classificacao_knn import (
    amostras_erradas,
    avaliar_modelo,
    dividir_dados,
    escolher_k,
    padronizar,
    treinar_knn,
)
from componentes_principais import aplicar_pca, grafico_pca

CAMINHO_DADOS = Path("dados/iris.data")
PASTA_SAIDA = Path("saida")
SEMENTE = 42

def main() -> None:
    dados = carregar_iris(CAMINHO_DADOS)
    pasta_graficos = PASTA_SAIDA / "graficos"
    estatisticas_descritivas(dados)
    frequencia_categorias(dados)
    analisar_relacao(dados, "comprimento_sepala", "largura_sepala", "sepala", pasta_graficos)
    analisar_relacao(dados, "comprimento_petala", "largura_petala", "petala", pasta_graficos)
    distribuicao_quatro_dimensoes(dados, pasta_graficos)
    componentes, variancia = aplicar_pca(dados)
    grafico_pca(componentes, variancia, pasta_graficos)
 
    X_treino, X_teste, y_treino, y_teste = dividir_dados(dados, SEMENTE)
    X_treino_padronizado, X_teste_padronizado = padronizar(X_treino, X_teste)
    print("\n**Divisão treino/teste: ")
    print(f"Treino: {len(X_treino)} amostras")
    print(f"Teste: {len(X_teste)} amostras")
    print("\nEspécies no teste:")
    print(y_teste.value_counts())
    k = escolher_k(X_treino, y_treino, SEMENTE, pasta_graficos)
 
    modelo = treinar_knn(X_treino_padronizado, y_treino, k)
    previsoes = avaliar_modelo(modelo, X_teste_padronizado, y_teste, pasta_graficos)
    erradas = amostras_erradas(X_teste, y_teste, previsoes)
    print(f"\n**Amostras classificadas erradas: {len(erradas)} de {len(y_teste)}: ")
    print(erradas.to_string())
 
 
if __name__ == "__main__":
    main()
