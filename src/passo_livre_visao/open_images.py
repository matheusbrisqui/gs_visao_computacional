"""Monta o conjunto de treino a partir do Open Images (banco aberto de imagens anotadas do Google).

Este módulo cuida só da lógica: ler as anotações, escolher as imagens de forma equilibrada
entre as classes, separar treino, validação e teste, e converter as caixas para o formato YOLO.
Não acessa a rede; o download fica em `preparar_dataset.py`.

Formato das anotações do Open Images (arquivo CSV de caixas):
    ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
As coordenadas já vêm normalizadas de 0 a 1.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .classes import CLASSES, classe_de_origem


@dataclass(frozen=True)
class Caixa:
    classe: int  # índice da classe do produto
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def para_yolo(self) -> str:
        """Linha do arquivo de anotação YOLO: classe, centro x, centro y, largura, altura (tudo de 0 a 1)."""
        largura, altura = self.x_max - self.x_min, self.y_max - self.y_min
        return f"{self.classe} {self.x_min + largura / 2:.6f} {self.y_min + altura / 2:.6f} {largura:.6f} {altura:.6f}"


def ler_codigos_das_categorias(linhas: Iterable[list[str]]) -> dict[str, int]:
    """Do arquivo de descrições do Open Images (código, nome) para {código: índice da classe do produto}.

    Levanta ValueError se alguma categoria esperada não existir no arquivo, para o erro aparecer cedo.
    """
    origem = classe_de_origem()
    codigos: dict[str, int] = {}
    encontradas = set()
    for linha in linhas:
        if len(linha) < 2 or linha[0] == "LabelName":
            continue
        codigo, nome = linha[0], linha[1]
        if nome in origem:
            codigos[codigo] = origem[nome].indice
            encontradas.add(nome)
    faltando = sorted(set(origem) - encontradas)
    if faltando:
        raise ValueError(f"Categorias não encontradas no Open Images: {', '.join(faltando)}")
    return codigos


def ler_caixas(linhas: Iterable[dict[str, str]], codigos: dict[str, int]) -> dict[str, list[Caixa]]:
    """Agrupa por imagem as caixas das categorias de interesse.

    Imagens com caixa de grupo ("IsGroupOf", vários objetos em uma caixa só) ou de representação
    ("IsDepiction", desenho ou pintura) em uma categoria de interesse são descartadas por inteiro:
    elas ensinariam o modelo errado.
    """
    por_imagem: dict[str, list[Caixa]] = defaultdict(list)
    descartadas: set[str] = set()
    for linha in linhas:
        classe = codigos.get(linha["LabelName"])
        if classe is None:
            continue
        imagem = linha["ImageID"]
        if linha.get("IsGroupOf") == "1" or linha.get("IsDepiction") == "1":
            descartadas.add(imagem)
            continue
        x_min, x_max = _limitar(float(linha["XMin"])), _limitar(float(linha["XMax"]))
        y_min, y_max = _limitar(float(linha["YMin"])), _limitar(float(linha["YMax"]))
        if x_max <= x_min or y_max <= y_min:
            continue
        por_imagem[imagem].append(Caixa(classe, x_min, x_max, y_min, y_max))
    return {imagem: caixas for imagem, caixas in por_imagem.items() if imagem not in descartadas and caixas}


def _limitar(valor: float) -> float:
    return max(0.0, min(1.0, valor))


def _sorteio(texto: str) -> float:
    """Número estável entre 0 e 1 a partir de um texto: o mesmo resultado em qualquer máquina."""
    return int(hashlib.sha256(texto.encode()).hexdigest()[:12], 16) / 16**12


def selecionar_equilibrado(caixas_por_imagem: dict[str, list[Caixa]], maximo_por_classe: int) -> list[str]:
    """Escolhe imagens de modo que nenhuma classe domine o treino.

    Começa pela classe mais rara e segue até a mais comum. Para cada uma, inclui imagens até
    que haja `maximo_por_classe` imagens contendo aquela classe (contando as já escolhidas).
    A ordem é fixa (por sorteio estável), então o conjunto é reproduzível.
    """
    imagens_da_classe: dict[int, list[str]] = defaultdict(list)
    for imagem, caixas in caixas_por_imagem.items():
        for classe in {c.classe for c in caixas}:
            imagens_da_classe[classe].append(imagem)

    escolhidas: set[str] = set()
    contagem: Counter[int] = Counter()
    for classe in sorted(imagens_da_classe, key=lambda c: (len(imagens_da_classe[c]), c)):
        for imagem in sorted(imagens_da_classe[classe], key=_sorteio):
            if contagem[classe] >= maximo_por_classe:
                break
            if imagem in escolhidas:
                continue
            escolhidas.add(imagem)
            contagem.update({c.classe for c in caixas_por_imagem[imagem]})
    return sorted(escolhidas)


def particao(imagem: str, validacao: float = 0.1, teste: float = 0.1) -> str:
    """Treino, validação ou teste, decidido pelo identificador da imagem (estável entre execuções)."""
    sorteio = _sorteio("particao:" + imagem)
    if sorteio < teste:
        return "test"
    if sorteio < teste + validacao:
        return "val"
    return "train"


def fatores_de_repeticao(
    caixas_por_imagem: dict[str, list[Caixa]], imagens_de_treino: Iterable[str], alvo: int = 250, maximo: int = 4
) -> dict[str, int]:
    """Quantas vezes cada imagem de treino entra em uma época, para reforçar as classes raras.

    Uma classe com menos de `alvo` imagens tem as suas imagens repetidas até chegar perto do alvo
    (no máximo `maximo` vezes). Como o treino aplica variações aleatórias a cada passagem
    (recorte, espelhamento, mudança de cor), as repetições não são cópias idênticas.
    """
    imagens = list(imagens_de_treino)
    por_classe: Counter[int] = Counter()
    for imagem in imagens:
        por_classe.update({c.classe for c in caixas_por_imagem[imagem]})
    fator_da_classe = {classe: max(1, min(maximo, round(alvo / n))) for classe, n in por_classe.items()}
    return {imagem: max(fator_da_classe[c.classe] for c in caixas_por_imagem[imagem]) for imagem in imagens}


def resumo_por_classe(caixas_por_imagem: dict[str, list[Caixa]], imagens: Iterable[str]) -> dict[str, dict[str, int]]:
    """Por classe: em quantas imagens aparece e quantas caixas tem, no conjunto escolhido."""
    n_imagens: Counter[int] = Counter()
    n_caixas: Counter[int] = Counter()
    for imagem in imagens:
        caixas = caixas_por_imagem[imagem]
        n_imagens.update({c.classe for c in caixas})
        n_caixas.update(c.classe for c in caixas)
    return {c.nome: {"imagens": n_imagens[c.indice], "caixas": n_caixas[c.indice]} for c in CLASSES}


def conteudo_data_yaml(pasta_dataset: Path) -> str:
    """Arquivo de configuração lido pelo treino do YOLO."""
    nomes = "\n".join(f"  {c.indice}: {c.nome}" for c in CLASSES)
    return (
        f"path: {pasta_dataset.as_posix()}\n"
        "train: train.txt\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{nomes}\n"
    )


def abrir_csv_de_caixas(caminho: Path) -> Iterator[dict[str, str]]:
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        yield from csv.DictReader(arquivo)


def abrir_csv_simples(caminho: Path) -> Iterator[list[str]]:
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        yield from csv.reader(arquivo)
