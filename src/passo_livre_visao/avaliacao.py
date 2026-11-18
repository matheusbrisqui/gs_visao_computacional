"""Avaliação do modelo treinado no conjunto de teste (imagens que ele nunca viu no treino).

Gera, dentro do repositório:
  modelo/metricas.json            precisão, revocação e mAP gerais e por classe, e velocidade no processador
  assets/avaliacao_*.png          matriz de confusão e curvas
  assets/exemplo_*.jpg            fotos do conjunto de teste com as detecções desenhadas
  assets/CREDITOS.md              autor e licença das fotos de exemplo
"""

from __future__ import annotations

import csv
import json
import shutil
import statistics
import time
from pathlib import Path

from .caminhos import MODELO_FINAL, RAIZ, pasta_dataset, pasta_treinos
from .classes import CLASSES
from .detector import Detector, desligar_telemetria

GRAFICOS = {
    "confusion_matrix_normalized.png": "avaliacao_matriz_de_confusao.png",
    "BoxPR_curve.png": "avaliacao_curva_precisao_revocacao.png",
    "BoxF1_curve.png": "avaliacao_curva_f1.png",
}


def avaliar(modelo: str | None = None, tamanho: int = 416, exemplos_por_classe: int = 1) -> dict:
    from ultralytics import YOLO

    desligar_telemetria()
    caminho = Path(modelo) if modelo else MODELO_FINAL
    if not caminho.is_file():
        raise FileNotFoundError(f"Modelo não encontrado em {caminho}")
    dados = pasta_dataset() / "data.yaml"
    resultado = YOLO(str(caminho)).val(
        data=str(dados), split="test", imgsz=tamanho, device="cpu", workers=0, plots=True, verbose=False,
        project=str(pasta_treinos()), name="avaliacao", exist_ok=True,
    )
    caixa = resultado.box
    por_classe = {}
    for posicao, indice in enumerate(caixa.ap_class_index):
        por_classe[resultado.names[int(indice)]] = {
            "precisao": round(float(caixa.p[posicao]), 4),
            "revocacao": round(float(caixa.r[posicao]), 4),
            "map50": round(float(caixa.ap50[posicao]), 4),
            "map50_95": round(float(caixa.ap[posicao]), 4),
        }
    resumo_dataset = json.loads((pasta_dataset() / "resumo.json").read_text(encoding="utf-8"))
    velocidade = medir_velocidade(caminho, tamanho)
    metricas = {
        "modelo": caminho.name,
        "arquitetura": "YOLO11n (Ultralytics), ajuste fino a partir de pesos pré-treinados no COCO",
        "tamanho_da_imagem": tamanho,
        "conjunto": "teste",
        "imagens_de_teste": resumo_dataset["por_particao"]["test"],
        "geral": {
            "precisao": round(float(caixa.mp), 4),
            "revocacao": round(float(caixa.mr), 4),
            "map50": round(float(caixa.map50), 4),
            "map50_95": round(float(caixa.map), 4),
        },
        "por_classe": por_classe,
        "imagens_de_teste_por_classe": {nome: v["imagens"] for nome, v in resumo_dataset["por_classe_e_particao"]["test"].items()},
        "velocidade_no_processador": velocidade,
    }
    (MODELO_FINAL.parent / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    assets = RAIZ / "assets"
    assets.mkdir(exist_ok=True)
    for origem, destino in GRAFICOS.items():
        if (Path(resultado.save_dir) / origem).is_file():
            shutil.copyfile(Path(resultado.save_dir) / origem, assets / destino)
    gerar_exemplos(caminho, tamanho, assets, exemplos_por_classe)
    na_aplicacao = gerar_tela_de_tempo_real(caminho, tamanho, assets)
    if na_aplicacao:
        # taxa da aplicação completa (leitura do vídeo, detecção, regras, desenho e registro): é a que o usuário vê
        velocidade["quadros_por_segundo_na_aplicacao"] = na_aplicacao
        (MODELO_FINAL.parent / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    geral = metricas["geral"]
    print(f"Teste ({metricas['imagens_de_teste']} imagens): precisão {geral['precisao']:.3f} | revocação {geral['revocacao']:.3f} | "
          f"mAP50 {geral['map50']:.3f} | mAP50-95 {geral['map50_95']:.3f}")
    for nome, m in por_classe.items():
        print(f"  {nome:18s} precisão {m['precisao']:.3f} | revocação {m['revocacao']:.3f} | mAP50 {m['map50']:.3f}")
    print(f"Velocidade: detecção em {velocidade['ms_por_quadro']} ms por quadro; aplicação completa a "
          f"{velocidade.get('quadros_por_segundo_na_aplicacao', '?')} quadros por segundo")
    return metricas


def _imagens_de_teste() -> list[Path]:
    return sorted((pasta_dataset() / "images" / "test").glob("*.jpg"))


def medir_velocidade(modelo: Path, tamanho: int, quantidade: int = 40) -> dict:
    """Tempo de detecção por quadro, uma imagem por vez, como acontece na câmera."""
    import cv2

    detector = Detector(modelo, tamanho=tamanho)
    quadros = [cv2.imread(str(arquivo)) for arquivo in _imagens_de_teste()[:quantidade]]
    for quadro in quadros[:3]:  # aquecimento: as primeiras chamadas são mais lentas
        detector.detectar(quadro)
    tempos = []
    for quadro in quadros:
        inicio = time.perf_counter()
        detector.detectar(quadro)
        tempos.append((time.perf_counter() - inicio) * 1000)
    mediana = statistics.median(tempos)
    return {"ms_por_quadro": round(mediana, 1), "quadros_por_segundo": round(1000 / mediana, 1), "quadros_medidos": len(tempos)}


# Fotos de teste que não devem ilustrar o relatório (identificadores do Open Images)
EVITAR_COMO_EXEMPLO: frozenset[str] = frozenset({"7b010d3df34d9dc6", "2de6c1769ddbfcb9"})  # pessoas em primeiro plano


def gerar_exemplos(modelo: Path, tamanho: int, assets: Path, por_classe: int = 1) -> list[str]:
    """Grava fotos do conjunto de teste com as detecções do modelo e os créditos de cada foto.

    Para cada classe é escolhida a foto de teste em que o modelo detectou aquela classe com mais
    confiança (um acerto representativo). Também é gravado um exemplo de falha: uma foto em que
    nenhum dos objetos anotados foi encontrado. O arquivo `exemplos.json` descreve cada escolha.
    """
    import cv2

    from .classes import POR_NOME
    from .regras import Zona
    from .tempo_real import Pintor

    pasta = pasta_dataset()
    detector, pintor, zona = Detector(modelo, tamanho=tamanho), Pintor(), Zona()
    analisadas = []
    for arquivo in _imagens_de_teste():
        if arquivo.stem in EVITAR_COMO_EXEMPLO:
            continue
        anotacao = pasta / "labels" / "test" / f"{arquivo.stem}.txt"
        anotadas = {int(linha.split()[0]) for linha in anotacao.read_text(encoding="utf-8").splitlines() if linha.strip()}
        analisadas.append((arquivo, anotadas, detector.detectar(cv2.imread(str(arquivo)))))

    escolhidas: list[tuple[str, Path, list, dict]] = []
    usadas: set[Path] = set()
    for classe in CLASSES:
        candidatas = [
            (max((d.confianca for d in deteccoes if d.classe == classe.nome), default=0.0), arquivo, deteccoes)
            for arquivo, anotadas, deteccoes in analisadas
            if classe.indice in anotadas and arquivo not in usadas
        ]
        for confianca, arquivo, deteccoes in sorted(candidatas, key=lambda c: (-c[0], c[1].name))[:por_classe]:
            if confianca > 0:
                sufixo = "" if por_classe == 1 else f"_{arquivo.stem[:6]}"
                escolhidas.append((f"exemplo_{classe.nome}{sufixo}.jpg", arquivo, deteccoes, {"tipo": "acerto", "classe": classe.nome}))
                usadas.add(arquivo)
    for arquivo, anotadas, deteccoes in analisadas:
        detectadas = {POR_NOME[d.classe].indice for d in deteccoes if d.classe in POR_NOME}
        if arquivo not in usadas and not (anotadas & detectadas):
            nomes = sorted(c.nome for c in CLASSES if c.indice in anotadas)
            escolhidas.append(("exemplo_falha.jpg", arquivo, deteccoes, {"tipo": "falha", "nao_encontradas": nomes}))
            break

    creditos = {}
    with (pasta / "ATRIBUICAO.csv").open(encoding="utf-8", newline="") as arquivo_csv:
        for linha in csv.DictReader(arquivo_csv):
            creditos[linha["imagem"]] = linha
    linhas = [
        "# Créditos das fotos de exemplo",
        "",
        "Fotos do conjunto de teste do Open Images, usadas sob a licença indicada.",
        "As caixas e os rótulos foram desenhados pelo modelo deste projeto.",
        "",
        "| Arquivo | Autor | Licença | Origem |",
        "|---|---|---|---|",
    ]
    for antigo in assets.glob("exemplo_*.jpg"):
        antigo.unlink()
    descricoes = []
    for nome, arquivo, deteccoes, descricao in escolhidas:
        tela = pintor.desenhar(cv2.imread(str(arquivo)), deteccoes, zona, [], 0.0, detector.confianca)
        cv2.imwrite(str(assets / nome), tela)
        descricoes.append({"arquivo": nome, "imagem": arquivo.stem, **descricao})
        c = creditos.get(arquivo.stem, {})
        autor, licenca = c.get("autor") or "não informado", c.get("licenca") or "não informada"
        linhas.append(f"| {nome} | {autor} | {licenca} | {c.get('pagina_de_origem') or ''} |")
    (assets / "CREDITOS.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    (assets / "exemplos.json").write_text(json.dumps(descricoes, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return [d["arquivo"] for d in descricoes]


def gerar_tela_de_tempo_real(modelo: Path, tamanho: int, assets: Path, quadros_por_foto: int = 40, fotos: int = 20) -> float | None:
    """Roda a aplicação de tempo real em um vídeo montado com fotos de teste e guarda uma tela com alerta.

    Serve de evidência reproduzível do laço completo (leitura do vídeo, detecção, regras, alerta e
    registro) sem depender de câmera. Cada foto fica parada por `quadros_por_foto` quadros.
    """
    import cv2

    from .caminhos import pasta_dados
    from .tempo_real import executar

    trabalho = pasta_dados() / "tela_de_tempo_real"
    shutil.rmtree(trabalho, ignore_errors=True)
    trabalho.mkdir(parents=True)
    sequencia = [a for a in _imagens_de_teste() if a.stem not in EVITAR_COMO_EXEMPLO][:fotos]
    video = trabalho / "sequencia.mp4"
    gravador = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 480))
    for arquivo in sequencia:
        quadro = cv2.resize(cv2.imread(str(arquivo)), (640, 480))
        for _ in range(quadros_por_foto):
            gravador.write(quadro)
    gravador.release()

    inicio = time.perf_counter()
    executar(str(video), modelo=str(modelo), tamanho=tamanho, janela=False, pasta_saida=trabalho)
    quadros_por_segundo = round(len(sequencia) * quadros_por_foto / (time.perf_counter() - inicio), 1)  # inclui a carga do modelo
    eventos = []
    for registro in sorted(trabalho.glob("alertas_*.jsonl")):
        eventos += [json.loads(linha) for linha in registro.read_text(encoding="utf-8").splitlines() if linha.strip()]
    com_captura = [e for e in eventos if e.get("captura")]
    if not com_captura:
        return None
    # preferência: um bloqueio sem pessoa na cena, com a maior confiança
    escolhido = min(com_captura, key=lambda e: (e["nivel"] != "bloqueio", e["prioridade"] == "alta", -e["confianca"]))
    shutil.copyfile(trabalho / escolhido["captura"], assets / "tempo_real.jpg")
    origem = sequencia[(escolhido["quadro"] - 1) // quadros_por_foto].stem

    descricoes = json.loads((assets / "exemplos.json").read_text(encoding="utf-8"))
    descricoes.append({"arquivo": "tempo_real.jpg", "imagem": origem, "tipo": "tempo_real", "mensagem": escolhido["mensagem"],
                       "alertas_na_sequencia": len(eventos), "fotos_na_sequencia": len(sequencia)})
    (assets / "exemplos.json").write_text(json.dumps(descricoes, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with (pasta_dataset() / "ATRIBUICAO.csv").open(encoding="utf-8", newline="") as arquivo_csv:
        credito = next((linha for linha in csv.DictReader(arquivo_csv) if linha["imagem"] == origem), {})
    autor, licenca = credito.get("autor") or "não informado", credito.get("licenca") or "não informada"
    with (assets / "CREDITOS.md").open("a", encoding="utf-8") as creditos:
        creditos.write(f"| tempo_real.jpg | {autor} | {licenca} | {credito.get('pagina_de_origem') or ''} |\n")
    return quadros_por_segundo
