"""Onde ficam os arquivos do projeto.

O código e o modelo final ficam no repositório. Os arquivos pesados (imagens de treino,
anotações baixadas e resultados brutos do treino) ficam na "pasta de dados", que não vai
para o GitHub. Por padrão ela é `dados/` dentro do repositório. Para usar outro lugar,
defina a variável de ambiente VISAO_DADOS ou grave o caminho no arquivo `.dados_local`.
"""

from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]  # raiz do repositório
MODELO_FINAL = RAIZ / "modelo" / "passo_livre.pt"
PASTA_SAIDA = RAIZ / "saida"  # alertas e capturas gerados ao rodar o detector
FOTOS_PROPRIAS = RAIZ / "fotos_proprias"  # opcional: fotos do grupo, já anotadas no formato YOLO


def pasta_dados() -> Path:
    variavel = os.environ.get("VISAO_DADOS", "").strip()
    if variavel:
        return Path(variavel)
    arquivo = RAIZ / ".dados_local"
    if arquivo.is_file():
        linhas = [linha.strip() for linha in arquivo.read_text(encoding="utf-8").splitlines() if linha.strip()]
        if linhas:
            return Path(linhas[0])
    return RAIZ / "dados"


def pasta_dataset() -> Path:
    return pasta_dados() / "dataset"


def pasta_anotacoes() -> Path:
    return pasta_dados() / "open_images"


def pasta_treinos() -> Path:
    return pasta_dados() / "treinos"
