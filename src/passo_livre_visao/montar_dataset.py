"""Baixa e monta o conjunto de treino: anotações e imagens do Open Images, no formato YOLO.

Etapas:
  1. baixa os arquivos de anotação (validação e teste do Open Images, cerca de 115 MB);
  2. escolhe as imagens das classes do produto, equilibrando as classes;
  3. baixa cada imagem do armazenamento oficial, reduz para no máximo 640 pixels e grava
     a imagem e o arquivo de caixas nas pastas de treino, validação e teste;
  4. soma as fotos do grupo, se existirem em `fotos_proprias/`;
  5. grava `data.yaml`, `resumo.json` e `ATRIBUICAO.csv` (autor e licença de cada imagem).

É seguro executar de novo: o que já foi baixado não é baixado outra vez.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import open_images as oi
from .caminhos import FOTOS_PROPRIAS, pasta_anotacoes, pasta_dataset
from .classes import CLASSES

BASE = "https://storage.googleapis.com/openimages"
URL_CATEGORIAS = f"{BASE}/v7/oidv7-class-descriptions-boxable.csv"
SUBCONJUNTOS = {
    "validation": {
        "caixas": f"{BASE}/v5/validation-annotations-bbox.csv",
        "imagens": f"{BASE}/2018_04/validation/validation-images-with-rotation.csv",
    },
    "test": {
        "caixas": f"{BASE}/v5/test-annotations-bbox.csv",
        "imagens": f"{BASE}/2018_04/test/test-images-with-rotation.csv",
    },
}
URL_IMAGEM = "https://open-images-dataset.s3.amazonaws.com/{subconjunto}/{imagem}.jpg"
AGENTE = "PassoLivre-Visao/1.0 (projeto academico)"
LADO_MAXIMO = 640
TENTATIVAS = 3


def _requisicao(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENTE}), timeout=60)


def baixar_arquivo(url: str, destino: Path) -> Path:
    """Baixa um arquivo grande em partes. Se já existir, não baixa de novo."""
    if destino.is_file() and destino.stat().st_size > 0:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + ".parte")
    with _requisicao(url) as resposta, parcial.open("wb") as arquivo:
        shutil.copyfileobj(resposta, arquivo, length=1024 * 1024)
    parcial.replace(destino)
    print(f"  baixado {destino.name} ({destino.stat().st_size / 1e6:.1f} MB)")
    return destino


def _baixar_imagem(url: str) -> bytes | None:
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            with _requisicao(url) as resposta:
                return resposta.read()
        except urllib.error.HTTPError as erro:
            if erro.code in (403, 404):
                return None  # imagem removida do banco: não adianta insistir
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1.5 * tentativa)
    return None


def _gravar_exemplo(imagem: str, subconjunto: str, caixas: list[oi.Caixa], pasta: Path) -> bool:
    """Baixa uma imagem, reduz o tamanho e grava imagem e anotação. Devolve False se não conseguiu."""
    from PIL import Image  # importado aqui para os testes da lógica não dependerem do Pillow

    parte = oi.particao(imagem)
    destino_imagem = pasta / "images" / parte / f"{imagem}.jpg"
    destino_caixas = pasta / "labels" / parte / f"{imagem}.txt"
    if not destino_imagem.is_file():
        dados = _baixar_imagem(URL_IMAGEM.format(subconjunto=subconjunto, imagem=imagem))
        if dados is None:
            return False
        try:
            # As caixas do Open Images referem-se à imagem como está gravada. Regravar sem os
            # metadados de orientação garante que o treino leia a imagem na mesma posição.
            foto = Image.open(io.BytesIO(dados)).convert("RGB")
            foto.thumbnail((LADO_MAXIMO, LADO_MAXIMO))
            foto.save(destino_imagem, "JPEG", quality=88)
        except OSError:
            return False
    destino_caixas.write_text("\n".join(c.para_yolo() for c in caixas) + "\n", encoding="utf-8")
    return True


def _ler_metadados(caminho: Path, interesse: set[str]) -> dict[str, dict[str, str]]:
    """Autor, licença e página de origem das imagens escolhidas (para a atribuição)."""
    metadados = {}
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        for linha in csv.DictReader(arquivo):
            if linha["ImageID"] in interesse:
                metadados[linha["ImageID"]] = linha
    return metadados


def _somar_fotos_proprias(pasta: Path) -> int:
    """Copia para o conjunto as fotos do grupo (`fotos_proprias/images` e `fotos_proprias/labels`)."""
    total = 0
    for foto in sorted((FOTOS_PROPRIAS / "images").glob("*.jpg")) if FOTOS_PROPRIAS.is_dir() else []:
        anotacao = FOTOS_PROPRIAS / "labels" / f"{foto.stem}.txt"
        if not anotacao.is_file():
            print(f"  aviso: {foto.name} sem arquivo de anotação; ignorada")
            continue
        parte = oi.particao("propria:" + foto.stem)
        shutil.copyfile(foto, pasta / "images" / parte / f"propria_{foto.name}")
        shutil.copyfile(anotacao, pasta / "labels" / parte / f"propria_{foto.stem}.txt")
        total += 1
    return total


def montar(maximo_por_classe: int = 400, trabalhadores: int = 16) -> dict:
    pasta, anotacoes = pasta_dataset(), pasta_anotacoes()
    for parte in ("train", "val", "test"):
        (pasta / "images" / parte).mkdir(parents=True, exist_ok=True)
        (pasta / "labels" / parte).mkdir(parents=True, exist_ok=True)

    print("1/5 Anotações do Open Images")
    categorias = baixar_arquivo(URL_CATEGORIAS, anotacoes / "categorias.csv")
    codigos = oi.ler_codigos_das_categorias(oi.abrir_csv_simples(categorias))

    caixas_por_imagem: dict[str, list[oi.Caixa]] = {}
    subconjunto_de: dict[str, str] = {}
    arquivos_metadados = {}
    for nome, urls in SUBCONJUNTOS.items():
        arquivo_caixas = baixar_arquivo(urls["caixas"], anotacoes / f"{nome}_caixas.csv")
        arquivos_metadados[nome] = baixar_arquivo(urls["imagens"], anotacoes / f"{nome}_imagens.csv")
        lidas = oi.ler_caixas(oi.abrir_csv_de_caixas(arquivo_caixas), codigos)
        caixas_por_imagem.update(lidas)
        subconjunto_de.update(dict.fromkeys(lidas, nome))

    print("2/5 Seleção equilibrada das imagens")
    candidatas = set(caixas_por_imagem)
    metadados: dict[str, dict[str, str]] = {}
    for nome, arquivo in arquivos_metadados.items():
        metadados.update(_ler_metadados(arquivo, {i for i in candidatas if subconjunto_de[i] == nome}))
    # imagens que o banco marca como giradas ficariam de lado em relação às caixas: ficam de fora
    giradas = {i for i, m in metadados.items() if m.get("Rotation", "").strip() not in ("", "0", "0.0")}
    disponiveis = {i: c for i, c in caixas_por_imagem.items() if i not in giradas}
    escolhidas = oi.selecionar_equilibrado(disponiveis, maximo_por_classe)
    print(f"  {len(candidatas)} imagens com as classes de interesse, {len(giradas)} giradas descartadas, {len(escolhidas)} escolhidas")

    print(f"3/5 Download de {len(escolhidas)} imagens")
    inicio = time.monotonic()

    def tarefa(imagem: str) -> bool:
        return _gravar_exemplo(imagem, subconjunto_de[imagem], disponiveis[imagem], pasta)

    gravadas = []
    with ThreadPoolExecutor(max_workers=trabalhadores) as executor:
        for n, (imagem, ok) in enumerate(zip(escolhidas, executor.map(tarefa, escolhidas), strict=True), 1):
            if ok:
                gravadas.append(imagem)
            if n % 250 == 0 or n == len(escolhidas):
                print(f"  {n}/{len(escolhidas)} ({time.monotonic() - inicio:.0f} s)")
    falhas = len(escolhidas) - len(gravadas)

    print("4/5 Fotos do grupo")
    proprias = _somar_fotos_proprias(pasta)
    print(f"  {proprias} foto(s) do grupo somada(s)")

    print("5/5 Arquivos de configuração e atribuição")
    (pasta / "data.yaml").write_text(oi.conteudo_data_yaml(pasta), encoding="utf-8")
    # lista de treino: imagens das classes raras entram mais de uma vez por época
    treino = [i for i in gravadas if oi.particao(i) == "train"]
    repeticoes = oi.fatores_de_repeticao(disponiveis, treino)
    linhas_de_treino = [f"./images/train/{imagem}.jpg" for imagem in treino for _ in range(repeticoes[imagem])]
    linhas_de_treino += [f"./images/train/{foto.name}" for foto in sorted((pasta / "images" / "train").glob("propria_*.jpg"))]
    (pasta / "train.txt").write_text("\n".join(linhas_de_treino) + "\n", encoding="utf-8")
    with (pasta / "ATRIBUICAO.csv").open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["imagem", "particao", "autor", "licenca", "pagina_de_origem"])
        for imagem in gravadas:
            m = metadados.get(imagem, {})
            escritor.writerow([imagem, oi.particao(imagem), m.get("Author", ""), m.get("License", ""), m.get("OriginalLandingURL", "")])
    por_particao = {parte: sum(1 for i in gravadas if oi.particao(i) == parte) for parte in ("train", "val", "test")}
    resumo = {
        "fonte": "Open Images V7 (subconjuntos de validação e teste), https://storage.googleapis.com/openimages/web/index.html",
        "licenca": "anotações CC BY 4.0; imagens CC BY 2.0 (autores em ATRIBUICAO.csv)",
        "lado_maximo_px": LADO_MAXIMO,
        "maximo_por_classe": maximo_por_classe,
        "imagens": len(gravadas),
        "falhas_de_download": falhas,
        "fotos_do_grupo": proprias,
        "por_particao": por_particao,
        "amostras_de_treino_por_epoca": len(linhas_de_treino),
        "por_classe": oi.resumo_por_classe(disponiveis, gravadas),
        "por_classe_e_particao": {
            parte: oi.resumo_por_classe(disponiveis, [i for i in gravadas if oi.particao(i) == parte]) for parte in por_particao
        },
        "classes": {c.nome: list(c.origem_open_images) for c in CLASSES},
    }
    (pasta / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Conjunto pronto em {pasta}: {len(gravadas)} imagens ({falhas} falha(s) de download)")
    return resumo
