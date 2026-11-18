"""Treino customizado: ajuste fino de um YOLO pré-treinado para as classes do produto.

Parte-se de um modelo que já sabe reconhecer formas gerais (pré-treinado no COCO) e
continua-se o treino com as imagens e as classes do Passo Livre. É a técnica de
transferência de aprendizado: exige muito menos imagens e tempo do que treinar do zero.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .caminhos import MODELO_FINAL, pasta_dados, pasta_dataset, pasta_treinos
from .detector import desligar_telemetria
from .montar_dataset import baixar_arquivo

URL_MODELO_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{arquivo}"
# arquivos do treino que valem como evidência e vão para o repositório
EVIDENCIAS = ("results.csv", "results.png", "args.yaml", "labels.jpg")


def treinar(
    epocas: int = 20,
    tamanho: int = 416,
    lote: int = 16,
    base: str = "yolo11n.pt",
    nome: str = "passo_livre",
    continuar: bool = False,
    fracao: float = 1.0,
    congelar: int = 10,
    dispositivo: str = "cpu",
) -> Path:
    from ultralytics import YOLO

    desligar_telemetria()
    dados = pasta_dataset() / "data.yaml"
    if not dados.is_file():
        raise FileNotFoundError(f"Conjunto de treino não encontrado em {dados}. Rode antes: python visao.py dataset")
    pasta = pasta_treinos() / nome

    if continuar:
        ultimo = pasta / "weights" / "last.pt"
        if not ultimo.is_file():
            raise FileNotFoundError(f"Não há treino interrompido em {pasta}")
        YOLO(str(ultimo)).train(resume=True)
    else:
        pesos = baixar_arquivo(URL_MODELO_BASE.format(arquivo=base), pasta_dados() / "pesos" / base)
        YOLO(str(pesos)).train(
            data=str(dados),
            epochs=epocas,
            imgsz=tamanho,
            batch=lote,
            device=dispositivo,  # "cpu" para o processador; "0" para a primeira placa de vídeo
            workers=4,
            cache=True,  # imagens pequenas: cabem na memória e aceleram o treino no processador
            patience=8,  # para antes se a validação não melhorar por 8 épocas
            fraction=fracao,
            freeze=congelar or None,  # camadas iniciais mantidas como vieram do modelo pré-treinado
            seed=42,
            project=str(pasta_treinos()),
            name=nome,
            exist_ok=True,
            plots=True,
        )
    return publicar(pasta)


def publicar(pasta_do_treino: Path) -> Path:
    """Copia o melhor modelo e as evidências do treino para a pasta `modelo/` do repositório."""
    melhor = pasta_do_treino / "weights" / "best.pt"
    if not melhor.is_file():
        raise FileNotFoundError(f"Treino sem modelo final em {melhor}")
    MODELO_FINAL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(melhor, MODELO_FINAL)
    evidencias = MODELO_FINAL.parent / "treino"
    evidencias.mkdir(exist_ok=True)
    for arquivo in EVIDENCIAS:
        if (pasta_do_treino / arquivo).is_file():
            shutil.copyfile(pasta_do_treino / arquivo, evidencias / arquivo)
    _limpar_caminhos_pessoais(evidencias / "args.yaml")
    resumo = pasta_dataset() / "resumo.json"
    if resumo.is_file():
        (MODELO_FINAL.parent / "dataset.json").write_text(resumo.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Modelo publicado em {MODELO_FINAL} ({MODELO_FINAL.stat().st_size / 1e6:.1f} MB)")
    return MODELO_FINAL


def _limpar_caminhos_pessoais(args_yaml: Path) -> None:
    """O arquivo de parâmetros do treino guarda caminhos do computador; no repositório ficam só os nomes."""
    if not args_yaml.is_file():
        return
    linhas = []
    for linha in args_yaml.read_text(encoding="utf-8").splitlines():
        chave, separador, valor = linha.partition(": ")
        if separador and ("\\" in valor or "/" in valor) and chave in ("model", "data", "project", "save_dir"):
            valor = Path(valor.strip().strip("'\"")).name
            linha = f"{chave}: {valor}"
        linhas.append(linha)
    args_yaml.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def historico(results_csv: Path) -> list[dict[str, float]]:
    """Lê o histórico do treino (uma linha por época) com nomes de coluna sem espaços."""
    import csv

    with results_csv.open(encoding="utf-8", newline="") as arquivo:
        return [{chave.strip(): float(valor) for chave, valor in linha.items() if valor.strip()} for linha in csv.DictReader(arquivo)]


if __name__ == "__main__":
    print(json.dumps({"modelo": str(treinar())}))
