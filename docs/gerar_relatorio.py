"""Gera o documento em PDF da disciplina de Computer Vision (`docs/relatorio_visao_computacional.pdf`).

O texto fica neste arquivo. Os números vêm dos arquivos reais do treino e da avaliação
(modelo/metricas.json, modelo/dataset.json, modelo/treino/results.csv) e os dados da equipe
de `docs/equipe.json`. Campos não preenchidos saem em vermelho. O mesmo comando atualiza
no README o bloco de integrantes e links.

Uso:  python docs/gerar_relatorio.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "docs" / "relatorio_visao_computacional.pdf"
LARGURA_UTIL = A4[0] - 4 * cm

AZUL = colors.HexColor("#0b57d0")
CINZA = colors.HexColor("#f1f4f8")
BORDA = colors.HexColor("#c9d1dc")

base = getSampleStyleSheet()
E_TITULO = ParagraphStyle("titulo", parent=base["Title"], fontSize=21, leading=26, textColor=AZUL, spaceAfter=6)
E_SUB = ParagraphStyle("sub", parent=base["Normal"], fontSize=12, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#334155"))
E_H1 = ParagraphStyle("h1", parent=base["Heading1"], fontSize=14, leading=18, textColor=AZUL, spaceBefore=14, spaceAfter=6)
E_H2 = ParagraphStyle("h2", parent=base["Heading2"], fontSize=11, leading=14, spaceBefore=8, spaceAfter=4)
E_P = ParagraphStyle("p", parent=base["Normal"], fontSize=10, leading=14.5, spaceAfter=6, alignment=TA_JUSTIFY)
E_CEL = ParagraphStyle("cel", parent=base["Normal"], fontSize=8.5, leading=11)
E_CELB = ParagraphStyle("celb", parent=E_CEL, fontName="Helvetica-Bold")
E_MONO = ParagraphStyle("mono", parent=base["Normal"], fontName="Courier", fontSize=7.5, leading=10, backColor=CINZA, borderPadding=4, spaceAfter=8)
E_LEG = ParagraphStyle("leg", parent=base["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#475569"), alignment=TA_CENTER, spaceAfter=8)

ROTULOS = {
    "veiculo": "Veículo", "moto": "Moto", "bicicleta": "Bicicleta", "lixeira_entulho": "Lixeira ou entulho",
    "poste_placa": "Poste ou placa", "escada_degrau": "Escada ou degrau", "cadeira_de_rodas": "Pessoa em cadeira de rodas",
}


def p(texto: str, estilo=E_P) -> Paragraph:
    return Paragraph(texto, estilo)


def falta(o_que: str) -> str:
    return f'<font color="#b3261e"><b>[PREENCHER: {escape(o_que)}]</b></font>'


def ou_falta(valor: str | None, o_que: str) -> str:
    return escape(valor) if valor else falta(o_que)


def tabela(linhas: list[list], larguras: list[float], cabecalho: bool = True) -> Table:
    dados = [[c if not isinstance(c, str) else Paragraph(c, E_CELB if (cabecalho and i == 0) else E_CEL) for c in linha] for i, linha in enumerate(linhas)]
    t = Table(dados, colWidths=[w * LARGURA_UTIL for w in larguras], repeatRows=1 if cabecalho else 0)
    estilo = [("GRID", (0, 0), (-1, -1), 0.4, BORDA), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if cabecalho:
        estilo.append(("BACKGROUND", (0, 0), (-1, 0), CINZA))
    t.setStyle(TableStyle(estilo))
    return t


def mono(linhas: list[str]) -> Paragraph:
    return Paragraph("<br/>".join(escape(linha).replace(" ", "&nbsp;") for linha in linhas), E_MONO)


def figura(caminho: Path, largura: float, legenda: str, altura_maxima: float = 15 * cm) -> list:
    if not caminho.is_file():
        return [p(falta(f"imagem {caminho.name}; gere com python visao.py avaliar"))]
    img = Image(str(caminho))
    proporcao = img.imageHeight / img.imageWidth
    if largura * proporcao > altura_maxima:
        largura = altura_maxima / proporcao
    img.drawWidth, img.drawHeight = largura, largura * proporcao
    return [KeepTogether([img, p(legenda, E_LEG)])]


def grade_de_figuras(itens: list[tuple[Path, str]], colunas: int = 2, altura_maxima: float = 6.0 * cm) -> list:
    """Várias figuras lado a lado, cada uma com a sua legenda, para não gastar uma página por foto."""
    largura_da_celula = LARGURA_UTIL / colunas
    celulas = []
    for caminho, legenda in itens:
        img = Image(str(caminho))
        largura = largura_da_celula - 0.5 * cm
        proporcao = img.imageHeight / img.imageWidth
        if largura * proporcao > altura_maxima:
            largura = altura_maxima / proporcao
        img.drawWidth, img.drawHeight = largura, largura * proporcao
        celulas.append([img, Paragraph(legenda, E_LEG)])
    while len(celulas) % colunas:
        celulas.append("")
    linhas = [celulas[i:i + colunas] for i in range(0, len(celulas), colunas)]
    grade = Table(linhas, colWidths=[largura_da_celula] * colunas)
    grade.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                               ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return [grade] if itens else []


def carregar_json(caminho: Path, padrao):
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return padrao


def numero(valor: float, casas: int = 3) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def contar_palavras(texto: str) -> int:
    return len(texto.split())


def historico_do_treino() -> list[dict[str, float]]:
    arquivo = RAIZ / "modelo" / "treino" / "results.csv"
    if not arquivo.is_file():
        return []
    with arquivo.open(encoding="utf-8", newline="") as f:
        return [{k.strip(): float(v) for k, v in linha.items() if v.strip()} for linha in csv.DictReader(f)]


def parametros_do_treino() -> dict[str, str]:
    arquivo = RAIZ / "modelo" / "treino" / "args.yaml"
    if not arquivo.is_file():
        return {}
    pares = (linha.partition(": ") for linha in arquivo.read_text(encoding="utf-8").splitlines())
    return {chave.strip(): valor.strip() for chave, separador, valor in pares if separador}


def grafico_do_treino(historico: list[dict[str, float]]) -> Path | None:
    """Curvas de erro e de qualidade por época, desenhadas a partir do histórico real do treino."""
    if not historico:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epocas = [int(h["epoch"]) for h in historico]
    figura_, (esquerda, direita) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
    esquerda.plot(epocas, [h["train/box_loss"] for h in historico], label="caixa (treino)", color="#0b57d0")
    esquerda.plot(epocas, [h["train/cls_loss"] for h in historico], label="classe (treino)", color="#d97706")
    esquerda.plot(epocas, [h["val/box_loss"] for h in historico], label="caixa (validação)", color="#0b57d0", linestyle="--")
    esquerda.plot(epocas, [h["val/cls_loss"] for h in historico], label="classe (validação)", color="#d97706", linestyle="--")
    esquerda.set_title("Erro por época (quanto menor, melhor)")
    direita.plot(epocas, [h["metrics/mAP50(B)"] for h in historico], label="mAP50", color="#1b7f3b")
    direita.plot(epocas, [h["metrics/mAP50-95(B)"] for h in historico], label="mAP50-95", color="#1b7f3b", linestyle="--")
    direita.plot(epocas, [h["metrics/precision(B)"] for h in historico], label="precisão", color="#6b21a8", alpha=0.7)
    direita.plot(epocas, [h["metrics/recall(B)"] for h in historico], label="revocação", color="#b3261e", alpha=0.7)
    direita.set_title("Qualidade na validação (quanto maior, melhor)")
    direita.set_ylim(0, 1)
    for eixo in (esquerda, direita):
        eixo.set_xlabel("época")
        eixo.grid(alpha=0.3)
        eixo.legend(fontsize=8)
    figura_.tight_layout()
    destino = RAIZ / "assets" / "treino_curvas.png"
    destino.parent.mkdir(exist_ok=True)
    figura_.savefig(destino)
    plt.close(figura_)
    return destino


def rodar_testes() -> str:
    comando = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    resultado = subprocess.run(comando, cwd=RAIZ, capture_output=True, text=True, encoding="utf-8", errors="replace")
    linhas = [linha for linha in resultado.stdout.splitlines() if linha.strip()]
    return linhas[-1].strip("= ") if linhas else "suíte não executada"


def rodape(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(2 * cm, 1.2 * cm, "Passo Livre | Applied Computer Vision | FIAP Global Solution 2026.2")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Página {doc.page}")
    canvas.restoreState()


def montar() -> list:
    equipe = carregar_json(RAIZ / "docs" / "equipe.json", {})
    integrantes = [i for i in equipe.get("integrantes", []) if i.get("nome") or i.get("rm")]
    metricas = carregar_json(RAIZ / "modelo" / "metricas.json", {})
    dataset = carregar_json(RAIZ / "modelo" / "dataset.json", {})
    historico = historico_do_treino()
    args = parametros_do_treino()
    geral = metricas.get("geral", {})
    velocidade = metricas.get("velocidade_no_processador", {})
    por_classe = metricas.get("por_classe", {})
    tem_resultados = bool(geral and dataset)
    n_imagens = dataset.get("imagens", 0)
    particoes = dataset.get("por_particao", {})
    epocas_feitas = len(historico)
    tamanho = metricas.get("tamanho_da_imagem", 416)

    def m(chave: str, casas: int = 2) -> str:
        return numero(geral[chave], casas) if chave in geral else "[pendente]"

    fps_medido = velocidade.get("quadros_por_segundo_na_aplicacao") or velocidade.get("quadros_por_segundo")
    fps = numero(fps_medido, 0) if fps_medido else "[pendente]"

    resumo = (
        "Pessoas com deficiência, idosos e responsáveis com carrinho de bebê encontram calçadas tomadas por veículos, lixo, postes e "
        "degraus, sem aviso prévio. Este trabalho apresenta o componente de visão computacional "
        "do Passo Livre, uma CitizenTech de acessibilidade e mobilidade urbana: um detector de obstáculos que funciona em tempo real na "
        "câmera de um computador comum. Foi realizado o treinamento customizado da arquitetura YOLO11n, por ajuste fino a partir de pesos "
        "pré-treinados, com uma taxonomia própria de sete classes: veículo, moto, bicicleta, lixeira ou entulho, poste ou placa, escada "
        f"ou degrau e pessoa em cadeira de rodas. O conjunto de dados reúne {n_imagens or '[pendente]'} imagens do banco aberto Open Images, "
        "selecionadas de forma equilibrada entre as classes e divididas em treino, validação e teste. Sobre o modelo, um conjunto de regras "
        "transforma detecções em alertas úteis: considera só o que está na zona de passagem, exige que o objeto persista por vários "
        "quadros, classifica o obstáculo como atenção ou bloqueio conforme a largura ocupada e eleva a prioridade quando há uma pessoa em "
        f"cadeira de rodas na cena. No conjunto de teste, o modelo alcançou mAP50 de {m('map50')}, precisão de {m('precisao')} e revocação "
        f"de {m('revocacao')}, processando cerca de {fps} quadros por segundo apenas no processador, sem placa de vídeo. Os resultados "
        "mostram que é viável mapear obstáculos de calçada com equipamento de baixo custo e indicam que objetos finos e variados, como "
        "postes e placas, são o principal ponto a melhorar. Código, modelo treinado, testes automatizados e evidências estão no repositório público."
    )
    objetivo = (
        "Treinar de forma customizada uma rede de detecção de objetos para reconhecer, em tempo real e em equipamento comum, obstáculos "
        "que dificultam a passagem de pessoas com mobilidade reduzida nas calçadas urbanas, e transformar essas detecções em alertas e relatos "
        "que alimentem o mapa colaborativo de acessibilidade do Passo Livre."
    )
    print(f"resumo: {contar_palavras(resumo)} palavras | objetivo: {contar_palavras(objetivo)} palavras")

    h = []
    # ---- capa (título)
    h += [Spacer(1, 3.2 * cm), p("Passo Livre", E_TITULO),
          p("Detecção em tempo real de obstáculos de acessibilidade em calçadas com treinamento customizado de YOLO",
            ParagraphStyle("c1", parent=E_SUB, fontSize=15, leading=20)), Spacer(1, 1.2 * cm),
          p("Applied Computer Vision", E_SUB), p("CitizenTech de acessibilidade e mobilidade urbana", E_SUB), Spacer(1, 1.5 * cm),
          p("FIAP | Global Solution 2026.2 | Engenharia de Software | 4º ano | Turma 4ES", E_SUB),
          p("Professores: Arnaldo Viana e Paulo Rodrigues", E_SUB), p(f"{date.today():%d/%m/%Y}", E_SUB), PageBreak()]

    # ---- integrantes
    h.append(p("1. Integrantes", E_H1))
    linhas = [["Nome completo", "RM"]] + ([[escape(i.get("nome") or ""), escape(i.get("rm") or "")] for i in integrantes]
                                          or [[falta("nome completo de cada integrante em docs/equipe.json"), falta("RM")]])
    h.append(tabela(linhas, [0.75, 0.25]))

    # ---- resumo
    h.append(p("2. Resumo", E_H1))
    h.append(p(escape(resumo)))
    h.append(p("<b>Palavras-chave:</b> visão computacional; detecção de objetos; YOLO; acessibilidade; mobilidade urbana; CitizenTech."))

    # ---- introdução
    h.append(p("3. Introdução", E_H1))
    h.append(p("A calçada é o primeiro trecho de qualquer deslocamento, inclusive de quem usa transporte público. Para quem tem mobilidade "
               "reduzida, um carro estacionado sobre o passeio, um poste no meio da faixa livre, sacos de lixo ou um degrau sem rampa não são "
               "incômodos: são barreiras que obrigam a descer para a rua ou a desistir do trajeto. A norma brasileira de acessibilidade "
               "(ABNT NBR 9050) pede uma faixa livre contínua para circulação, mas o estado real das calçadas muda todos os dias e não "
               "existe um mapa atualizado dessas barreiras."))
    h.append(p("O <b>Passo Livre</b> é a CitizenTech proposta pelo grupo para a Global Solution: um mapa colaborativo de obstáculos em "
               "calçadas e pontos de ônibus, no qual o cidadão relata o problema e o poder público é acionado. O limite de um mapa feito só "
               "de relatos manuais é o esforço: a pessoa precisa parar, abrir o aplicativo, escolher o tipo de obstáculo e descrever. A visão "
               "computacional reduz esse esforço. Com a câmera apontada para o caminho, o sistema reconhece o obstáculo, avisa na hora e "
               "já deixa o relato pronto, com o tipo preenchido."))
    h.append(p("Este documento descreve a construção desse componente: a escolha das classes, a montagem do conjunto de imagens, o "
               "treinamento customizado da rede, a avaliação com imagens que ela nunca viu e a aplicação em tempo real. O trabalho se "
               "relaciona aos Objetivos de Desenvolvimento Sustentável 10 (redução das desigualdades) e 11 (cidades e comunidades "
               "sustentáveis)."))

    # ---- objetivo
    h.append(p("4. Objetivo", E_H1))
    h.append(p(escape(objetivo)))

    # ---- ferramentas
    h.append(p("5. Explicação técnica das ferramentas", E_H1))
    h.append(p("Como funciona a detecção de objetos com YOLO", E_H2))
    h.append(p("Detectar objetos é responder a duas perguntas ao mesmo tempo: <b>o que</b> há na imagem e <b>onde</b> está. A família YOLO "
               "(<i>You Only Look Once</i>) faz isso em uma única passagem da imagem pela rede neural, o que a torna rápida o bastante para "
               "vídeo. A rede tem três partes. A <b>espinha dorsal</b> (<i>backbone</i>) é uma sequência de camadas convolucionais que extrai "
               "características visuais, das mais simples (bordas, cores) às mais abstratas (rodas, degraus). O <b>pescoço</b> (<i>neck</i>) "
               "combina características de escalas diferentes, para que objetos pequenos e grandes sejam vistos. A <b>cabeça</b> (<i>head</i>) "
               "produz, para cada região da imagem, uma caixa delimitadora, a classe do objeto e um grau de confiança. Como várias regiões "
               "apontam para o mesmo objeto, uma etapa final chamada supressão de não máximos mantém só a melhor caixa de cada um."))
    h.append(p("Foi usada a versão <b>YOLO11n</b>, a menor da geração 11, com cerca de 2,6 milhões de parâmetros. A escolha foi deliberada: "
               "o produto precisa rodar no equipamento que o cidadão ou o agente público já tem, sem placa de vídeo."))
    h.append(p("Treinamento customizado por transferência de aprendizado", E_H2))
    h.append(p("Treinar uma rede do zero exige centenas de milhares de imagens. Na <b>transferência de aprendizado</b>, parte-se de uma rede "
               "já treinada em um grande banco genérico (COCO, com 80 categorias) e continua-se o treino com as imagens e as classes do "
               "problema. As camadas iniciais, que reconhecem formas gerais, são reaproveitadas; as camadas finais são reajustadas para a "
               "nova tarefa. A cabeça da rede é recriada para o número de classes do produto, e portanto o modelo final só conhece a "
               "taxonomia do Passo Livre."))
    h.append(p("Ferramentas", E_H2))
    h.append(tabela([
        ["Ferramenta", "Uso no projeto", "Por que foi escolhida"],
        ["Python 3.11+", "Linguagem de todo o projeto", "Padrão da área, com as bibliotecas necessárias e fácil de reproduzir."],
        ["PyTorch", "Motor de redes neurais: cálculo dos gradientes e execução do modelo", "Biblioteca aberta mais usada em pesquisa e base do YOLO atual. Roda em processador ou em placa de vídeo sem mudar o código."],
        ["Ultralytics YOLO11", "Arquitetura da rede, laço de treino, aumento de dados e cálculo das métricas", "Implementação de referência do YOLO, com pesos pré-treinados públicos e versão leve (n) adequada a tempo real em processador."],
        ["OpenCV", "Leitura da câmera e de vídeos, desenho das caixas e gravação", "Biblioteca clássica de visão computacional, com acesso direto a câmeras e arquivos de vídeo."],
        ["Open Images V7", "Fonte das imagens e das caixas anotadas", "Banco aberto do Google com caixas conferidas por pessoas e licença que permite o uso (CC BY). Tem categorias que os bancos mais comuns não têm, como escada, lixeira e cadeira de rodas."],
        ["Pillow", "Redução das imagens e escrita de textos com acento sobre o vídeo", "As fontes embutidas do OpenCV não têm acentuação."],
        ["pytest e ruff", "Testes automatizados das regras e da preparação dos dados; padronização do código", "Garantem que a lógica do produto funciona sem depender de câmera ou de treino."],
        ["Matplotlib e ReportLab", "Gráficos do treino e geração deste documento", "O documento é gerado a partir dos arquivos reais do treino, e não digitado à mão."],
    ], [0.19, 0.36, 0.45]))
    h.append(p("Métricas usadas", E_H2))
    h.append(tabela([
        ["Métrica", "O que mede"],
        ["IoU (interseção sobre união)", "Quanto a caixa prevista coincide com a caixa correta: área em comum dividida pela área total das duas. Vai de 0 a 1."],
        ["Precisão", "Das detecções que o modelo fez, que fração estava certa. Precisão baixa significa alarme falso."],
        ["Revocação", "Dos objetos que existiam, que fração o modelo encontrou. Revocação baixa significa obstáculo não visto."],
        ["mAP50", "Média, entre as classes, da área sob a curva de precisão por revocação, aceitando como acerto caixas com IoU de pelo menos 0,5. É o indicador principal de detectores."],
        ["mAP50-95", "A mesma média, calculada para exigências de IoU de 0,5 a 0,95. Mais rigorosa: cobra caixas bem ajustadas."],
    ], [0.25, 0.75]))

    # ---- metodologia
    h.append(p("6. Metodologia (passo a passo)", E_H1))
    h.append(p("Passo 1. Definição das classes", E_H2))
    h.append(p("As classes foram definidas a partir dos tipos de obstáculo do aplicativo Passo Livre, e não das categorias dos modelos "
               "prontos. Várias categorias do banco de imagens são agrupadas em uma classe do produto, porque para quem está passando o "
               "efeito é o mesmo. A sétima classe não é obstáculo: indica a presença de uma pessoa em cadeira de rodas e serve para elevar "
               "a prioridade do alerta."))
    origem = dataset.get("classes", {})
    tipos = {"veiculo": "carro na calçada", "moto": "carro na calçada", "bicicleta": "outro", "lixeira_entulho": "entulho ou lixo",
             "poste_placa": "poste ou obstáculo fixo", "escada_degrau": "degrau sem rampa", "cadeira_de_rodas": "(não é obstáculo)"}
    h.append(tabela([["Classe do modelo", "Categorias do Open Images agrupadas", "Tipo de relato no aplicativo"]]
                    + [[ROTULOS[c], escape(", ".join(origem.get(c, [])) or "-"), tipos[c]] for c in ROTULOS], [0.27, 0.45, 0.28]))

    h.append(p("Passo 2. Coleta e seleção das imagens", E_H2))
    h.append(p("As anotações dos subconjuntos de validação e teste do Open Images V7 foram baixadas e filtradas pelas categorias de "
               "interesse. Imagens com caixas de grupo (vários objetos em uma caixa só), com desenhos ou pinturas no lugar do objeto real, "
               "ou marcadas como giradas foram descartadas, porque ensinariam o modelo errado. Como havia muito mais imagens de carros do "
               f"que de lixeiras, a seleção foi <b>equilibrada</b>: no máximo {dataset.get('maximo_por_classe', 400)} imagens por classe, "
               "começando pelas classes mais raras. A ordem de escolha é fixa, então o mesmo comando reproduz o mesmo conjunto."))
    h.append(p("Passo 3. Preparação", E_H2))
    h.append(p(f"Cada imagem foi baixada do armazenamento oficial, reduzida para no máximo {dataset.get('lado_maximo_px', 640)} pixels no "
               "lado maior e regravada sem metadados. As caixas foram convertidas para o formato YOLO (classe, centro, largura e altura, "
               "em valores de 0 a 1). O conjunto foi dividido em treino, validação e teste na proporção aproximada de 80, 10 e 10 por "
               "cento, por um sorteio estável ligado ao identificador da imagem. O teste fica separado e só é usado na avaliação final."))
    if tem_resultados:
        linhas = [["Classe", "Imagens no treino", "Imagens na validação", "Imagens no teste", "Caixas no total"]]
        cp = dataset["por_classe_e_particao"]
        for c in ROTULOS:
            linhas.append([ROTULOS[c], str(cp["train"][c]["imagens"]), str(cp["val"][c]["imagens"]), str(cp["test"][c]["imagens"]),
                           str(dataset["por_classe"][c]["caixas"])])
        linhas.append(["<b>Total de imagens</b>", f"<b>{particoes.get('train', 0)}</b>", f"<b>{particoes.get('val', 0)}</b>",
                       f"<b>{particoes.get('test', 0)}</b>", ""])
        h.append(tabela(linhas, [0.32, 0.17, 0.17, 0.17, 0.17]))
        h.append(p("Uma imagem pode conter mais de uma classe, por isso a soma das linhas é maior que o total de imagens.", E_LEG))
        h.append(p("As classes continuaram desiguais, porque o banco tem poucas imagens de lixeiras, escadas e cadeiras de rodas. Para "
                   "compensar, as imagens das classes raras entram mais de uma vez em cada época do treino (até quatro vezes). Como o treino "
                   "aplica variações aleatórias a cada passagem, as repetições não são cópias idênticas. Com isso, cada época percorre "
                   f"{dataset.get('amostras_de_treino_por_epoca', 0)} amostras."))

    h.append(p("Passo 4. Treinamento", E_H2))
    h.append(p("O treino foi feito no processador de um notebook comum (Intel Core i7 de 12ª geração, sem placa de vídeo dedicada), o "
               "mesmo tipo de equipamento em que o produto deve rodar. Para caber nesse equipamento, as dez primeiras camadas da rede "
               "(a espinha dorsal) foram mantidas como vieram do modelo pré-treinado, e apenas as camadas seguintes foram reajustadas."))
    h.append(tabela([
        ["Parâmetro", "Valor", "Significado"],
        ["Modelo de partida", escape(args.get("model", "yolo11n.pt")), "Pesos pré-treinados no banco COCO"],
        ["Épocas", f"{args.get('epochs', '20')} previstas, {epocas_feitas or '?'} executadas", "Passagens completas pelo conjunto de treino. O treino para antes se a validação deixar de melhorar."],
        ["Tamanho da imagem", f"{args.get('imgsz', tamanho)} pixels", "Lado para o qual cada imagem é redimensionada na entrada da rede"],
        ["Lote", args.get("batch", "16"), "Imagens processadas a cada ajuste dos pesos"],
        ["Camadas congeladas", args.get("freeze", "10"), "Camadas iniciais que não são reajustadas"],
        ["Otimizador", f"{args.get('optimizer', 'auto')} (escolha automática: AdamW)", "Algoritmo que ajusta os pesos a partir do erro"],
        ["Aumento de dados", "mosaico, espelhamento horizontal, variação de cor, escala e translação", "Variações aleatórias aplicadas às imagens, para o modelo não decorar o conjunto"],
        ["Paciência", args.get("patience", "8"), "Épocas sem melhora na validação antes de interromper"],
        ["Semente", args.get("seed", "42"), "Fixa os sorteios para o treino ser reproduzível"],
    ], [0.22, 0.33, 0.45]))
    grafico = grafico_do_treino(historico)
    if grafico:
        h += figura(grafico, LARGURA_UTIL, "Figura 1. Evolução do treino por época: erro no treino e na validação (esquerda) e métricas na validação (direita).")
    else:
        h.append(p(falta("histórico do treino; rode python visao.py treinar")))

    h.append(p("Passo 5. Avaliação", E_H2))
    h.append(p("O melhor modelo do treino (o de maior qualidade na validação) foi avaliado no conjunto de <b>teste</b>, com imagens que não "
               "participaram do treino nem da escolha do modelo. Foram calculadas precisão, revocação, mAP50 e mAP50-95, no geral e por "
               "classe, além da matriz de confusão e do tempo de detecção por quadro."))

    h.append(p("Passo 6. Das detecções aos alertas", E_H2))
    h.append(p("O modelo informa apenas que há um objeto em certa posição. O que o cidadão precisa saber é se aquilo atrapalha a passagem. "
               "Essa decisão fica em um conjunto de regras explícitas, separadas do modelo e cobertas por testes automatizados:"))
    h.append(tabela([
        ["Regra", "Como funciona"],
        ["Zona de passagem", "Só interessa o objeto cujo ponto de apoio (centro da base da caixa) cai na faixa central e inferior da imagem, que corresponde ao trecho da calçada logo à frente."],
        ["Confiança mínima", "Detecções abaixo de 40% de confiança são ignoradas."],
        ["Persistência", "O obstáculo precisa aparecer em pelo menos 8 dos últimos 12 quadros. Evita alerta por engano em um quadro isolado e tolera falhas curtas de detecção."],
        ["Nível", "Se a caixa cobre metade ou mais da largura da zona de passagem, o nível é bloqueio. Abaixo disso, atenção."],
        ["Prioridade", "Se há uma pessoa em cadeira de rodas na cena, o alerta sai com prioridade alta."],
        ["Intervalo", "O mesmo tipo de obstáculo não repete alerta em menos de 8 segundos."],
    ], [0.22, 0.78]))

    h.append(p("Passo 7. Aplicação em tempo real", E_H2))
    h.append(p("O programa lê a câmera (ou um vídeo gravado, ou uma câmera de rede), passa cada quadro pelo modelo, aplica as regras e "
               "mostra o resultado na tela: as caixas com o nome e a confiança, a zona de passagem, a taxa de quadros por segundo e a faixa "
               "de alerta. Cada alerta é gravado em um arquivo de eventos, com a foto do momento. Opcionalmente, o alerta cria um relato no "
               "aplicativo Passo Livre pela API, com o tipo de obstáculo já preenchido. Nenhuma imagem é enviada para fora do computador."))
    h.append(p("Passo 8. Testes automatizados", E_H2))
    h.append(p(f"As regras de alerta, a preparação dos dados e o registro dos eventos têm testes automatizados, que rodam sem câmera e sem "
               f"o modelo. Resultado da execução feita na geração deste documento: <b>{escape(rodar_testes())}</b>."))

    # ---- resultados
    h.append(p("7. Exemplos de resultados", E_H1))
    if tem_resultados:
        h.append(p(f"Resultados no conjunto de teste ({metricas.get('imagens_de_teste', '?')} imagens que o modelo não viu durante o treino), "
                   f"com imagens de {tamanho} pixels."))
        linhas = [["Classe", "Imagens de teste", "Precisão", "Revocação", "mAP50", "mAP50-95"]]
        teste_por_classe = metricas.get("imagens_de_teste_por_classe", {})
        for c in ROTULOS:
            if c in por_classe:
                mc = por_classe[c]
                linhas.append([ROTULOS[c], str(teste_por_classe.get(c, "")), numero(mc["precisao"]), numero(mc["revocacao"]),
                               numero(mc["map50"]), numero(mc["map50_95"])])
        linhas.append(["<b>Todas as classes</b>", f"<b>{metricas.get('imagens_de_teste', '')}</b>", f"<b>{numero(geral['precisao'])}</b>",
                       f"<b>{numero(geral['revocacao'])}</b>", f"<b>{numero(geral['map50'])}</b>", f"<b>{numero(geral['map50_95'])}</b>"])
        h.append(tabela(linhas, [0.30, 0.16, 0.13, 0.14, 0.13, 0.14]))
        melhores = sorted(por_classe, key=lambda c: -por_classe[c]["map50"])
        h.append(p(f"<b>Leitura dos resultados.</b> As classes com melhor desempenho foram {ROTULOS[melhores[0]].lower()} e "
                   f"{ROTULOS[melhores[1]].lower()}; as de pior desempenho, {ROTULOS[melhores[-1]].lower()} e {ROTULOS[melhores[-2]].lower()}. "
                   "A quantidade de imagens não explica sozinha essa ordem. Lixeira e cadeira de rodas têm poucas imagens, mas aparência "
                   "bem definida, e foram bem. Poste ou placa tem mais imagens, porém reúne cinco categorias de aparência muito diferente "
                   "(poste de luz, placas, parquímetro, hidrante), em geral objetos finos ou pequenos na foto, e foi a pior. Escadas variam "
                   "muito e não têm contorno claro. Vale a ressalva de que lixeira, escada e cadeira de rodas têm entre 11 e 14 imagens de "
                   "teste cada, o que torna os números dessas classes pouco estáveis."))
        h.append(p("<b>Precisão alta, revocação moderada.</b> Quando o modelo aponta um obstáculo, ele acerta na maior parte das vezes. "
                   "Em compensação, deixa de ver parte dos objetos, sobretudo os pequenos ou distantes. Para o uso proposto esse é o lado "
                   "mais seguro do compromisso, porque alarme falso em excesso faria o cidadão ignorar os avisos, mas a revocação é o "
                   "principal número a melhorar."))
        h.append(p(f"<b>Velocidade.</b> No processador, sem placa de vídeo, a detecção leva {numero(velocidade.get('ms_por_quadro', 0), 0)} "
                   f"milissegundos por quadro, medidos uma imagem por vez. A aplicação completa, que além de detectar lê o vídeo, aplica as regras, "
                   f"desenha a tela e grava os alertas, rodou a cerca de {fps} quadros por segundo. O valor varia com a carga do "
                   "computador, mas é suficiente para uso em tempo real por uma pessoa caminhando."))
        h += figura(RAIZ / "assets" / "avaliacao_matriz_de_confusao.png", 0.72 * LARGURA_UTIL,
                    "Figura 2. Matriz de confusão normalizada no conjunto de teste. A última linha e a última coluna (background) indicam "
                    "objetos não encontrados e detecções onde não havia objeto anotado.")
        h += figura(RAIZ / "assets" / "avaliacao_curva_precisao_revocacao.png", 0.72 * LARGURA_UTIL,
                    "Figura 3. Curva de precisão por revocação, por classe. A área sob cada curva é o AP da classe; a média é o mAP50.")
        h.append(p("Detecções em imagens de teste", E_H2))
        h.append(p("As figuras a seguir são fotos do conjunto de teste com as caixas e os rótulos produzidos pelo modelo treinado. O retângulo "
                   "azul é a zona de passagem usada pelas regras de alerta. Autores e licenças das fotos estão em assets/CREDITOS.md."))
        n = 4
        exemplos = []
        for c in ROTULOS:
            if (RAIZ / "assets" / f"exemplo_{c}.jpg").is_file():
                exemplos.append((RAIZ / "assets" / f"exemplo_{c}.jpg", f"Figura {n}. {ROTULOS[c]}."))
                n += 1
        descricoes = carregar_json(RAIZ / "assets" / "exemplos.json", [])
        falha = next((e for e in descricoes if e.get("tipo") == "falha"), None)
        if falha and (RAIZ / "assets" / falha["arquivo"]).is_file():
            perdidas = ", ".join(ROTULOS.get(c, c).lower() for c in falha.get("nao_encontradas", []))
            exemplos.append((RAIZ / "assets" / falha["arquivo"], f"Figura {n}. Exemplo de falha: a foto contém {perdidas}, e o modelo não encontrou."))
            n += 1
        h += grade_de_figuras(exemplos)
        h.append(p("As figuras de cada classe mostram a foto de teste em que o modelo teve mais confiança naquela classe. Para dar a "
                   "medida real do modelo, a última figura mostra um caso em que ele errou."))
        tempo_real = RAIZ / "assets" / "tempo_real.jpg"
        if tempo_real.is_file():
            sequencia = next((e for e in descricoes if e.get("tipo") == "tempo_real"), {})
            h.append(p("Aplicação em tempo real", E_H2))
            h.append(p("Para ter uma evidência reproduzível sem depender de câmera, a aplicação de tempo real foi executada sobre um vídeo "
                       f"montado com {sequencia.get('fotos_na_sequencia', 'várias')} fotos de teste em sequência. O laço completo (leitura do "
                       f"vídeo, detecção, regras, alerta e registro) emitiu {sequencia.get('alertas_na_sequencia', 'vários')} alertas. O uso "
                       "com câmera ao vivo está no vídeo de demonstração."))
            h += figura(tempo_real, 0.62 * LARGURA_UTIL, f"Figura {n}. Tela da aplicação de tempo real durante o processamento do vídeo de teste, "
                        "com a zona de passagem, a taxa de quadros por segundo e a faixa de alerta.", 9 * cm)
    else:
        h.append(p(falta("resultados; rode python visao.py treinar e python visao.py avaliar")))

    # ---- conclusão
    h.append(p("8. Conclusão", E_H1))
    h.append(p("O trabalho cumpriu o que a disciplina propõe: uma arquitetura de visão computacional foi treinada de forma customizada para "
               "um problema real e roda em tempo real. O principal resultado prático é que isso foi feito inteiramente em um notebook sem "
               "placa de vídeo, do treino à execução, o que aproxima a solução do equipamento disponível em prefeituras, associações e na "
               "mão do próprio cidadão."))
    h.append(p("Dois aprendizados se destacam. O primeiro é que o modelo sozinho não entrega valor ao cidadão: foram as regras de zona de "
               "passagem, persistência e prioridade que transformaram caixas na tela em um aviso compreensível. O segundo é que a definição "
               "das classes pesa tanto quanto a quantidade de imagens: a classe que agrupa objetos de aparência muito diferente (poste ou "
               "placa) teve o pior resultado, mesmo com mais imagens do que classes bem definidas, como lixeira e cadeira de rodas."))
    h.append(p("<b>Limitações.</b> O conjunto vem de um banco internacional, e não de calçadas brasileiras, de modo que o desempenho em campo "
               "tende a ser inferior ao medido. Não há classe para buraco, que é uma das barreiras mais comuns, porque o banco aberto usado "
               "não tem essa categoria. A zona de passagem é fixa e supõe a câmera apontada para a frente, na altura do peito. O modelo "
               "detecta o objeto, mas não mede a largura livre em centímetros."))
    h.append(p("<b>Próximos passos.</b> Coletar e anotar fotos de calçadas das cidades atendidas, incluindo buracos e rampas, e somá-las ao "
               "conjunto (a estrutura do projeto já prevê isso). Treinar por mais épocas em placa de vídeo, com a rede inteira liberada. "
               "Estimar a largura livre com apoio de um sensor de distância, integrando com o robô do projeto. Levar o modelo para o celular."))
    h.append(p("<b>Uso responsável.</b> As imagens captadas pela câmera ficam no computador de quem usa. O sistema não identifica pessoas: a "
               "classe de cadeira de rodas serve apenas para priorizar o alerta e não é associada a nenhum indivíduo. O envio ao aplicativo "
               "leva só o tipo do obstáculo e a posição informada."))

    # ---- links
    h.append(p("9. Links e reprodução", E_H1))
    h.append(tabela([
        ["Item", "Endereço"],
        ["Repositório GitHub", ou_falta(equipe.get("repositorio"), "link do repositório em docs/equipe.json")],
        ["Vídeo de demonstração", ou_falta(equipe.get("video"), "link do vídeo em docs/equipe.json")],
    ], [0.3, 0.7]))
    h.append(p("Como reproduzir", E_H2))
    h.append(mono([
        "pip install -r requirements.txt      # instala as bibliotecas",
        "python visao.py camera               # detecção em tempo real pela webcam (usa o modelo já treinado)",
        "python visao.py video ARQUIVO.mp4    # detecção em um vídeo gravado",
        "python visao.py dataset              # baixa e monta o conjunto de imagens",
        "python visao.py treinar              # treina o modelo",
        "python visao.py avaliar              # mede no conjunto de teste e gera gráficos e exemplos",
        "python -m pytest -q                  # testes automatizados",
    ]))
    h.append(p("Referências", E_H2))
    for referencia in (
        "KUZNETSOVA, A. et al. The Open Images Dataset V4. International Journal of Computer Vision, 2020. Imagens sob licença CC BY 2.0 e anotações sob CC BY 4.0. https://storage.googleapis.com/openimages/web/index.html",
        "JOCHER, G.; QIU, J. Ultralytics YOLO11. 2024. Licença AGPL-3.0. https://github.com/ultralytics/ultralytics",
        "REDMON, J. et al. You Only Look Once: Unified, Real-Time Object Detection. CVPR, 2016.",
        "LIN, T.-Y. et al. Microsoft COCO: Common Objects in Context. ECCV, 2014.",
        "ABNT. NBR 9050: Acessibilidade a edificações, mobiliário, espaços e equipamentos urbanos. 2020.",
    ):
        h.append(p(escape(referencia), ParagraphStyle("ref", parent=E_P, fontSize=8.5, leading=11, alignment=0)))
    return h


INICIO, FIM = "<!-- equipe:inicio -->", "<!-- equipe:fim -->"
R_INICIO, R_FIM = "<!-- resultados:inicio -->", "<!-- resultados:fim -->"


def _atualizar_resultados(texto: str) -> str:
    """Reescreve no README a tabela de resultados, a partir de modelo/metricas.json."""
    metricas = carregar_json(RAIZ / "modelo" / "metricas.json", {})
    dataset = carregar_json(RAIZ / "modelo" / "dataset.json", {})
    if R_INICIO not in texto or R_FIM not in texto or not metricas.get("geral") or not dataset:
        return texto
    geral, velocidade, particoes = metricas["geral"], metricas.get("velocidade_no_processador", {}), dataset.get("por_particao", {})
    linhas = [
        f"Modelo YOLO11n ajustado com {dataset.get('imagens', '?')} imagens ({particoes.get('train', '?')} de treino, "
        f"{particoes.get('val', '?')} de validação e {particoes.get('test', '?')} de teste). Números medidos no conjunto de teste, "
        f"com imagens de {metricas.get('tamanho_da_imagem', '?')} pixels.",
        "",
        "| Classe | Precisão | Revocação | mAP50 | mAP50-95 |",
        "|---|---|---|---|---|",
    ]
    for classe, rotulo in ROTULOS.items():
        if classe in metricas.get("por_classe", {}):
            m = metricas["por_classe"][classe]
            linhas.append(f"| {rotulo} | {numero(m['precisao'])} | {numero(m['revocacao'])} | {numero(m['map50'])} | {numero(m['map50_95'])} |")
    linhas.append(f"| **Todas as classes** | **{numero(geral['precisao'])}** | **{numero(geral['revocacao'])}** | "
                  f"**{numero(geral['map50'])}** | **{numero(geral['map50_95'])}** |")
    if velocidade:
        na_aplicacao = velocidade.get("quadros_por_segundo_na_aplicacao") or velocidade["quadros_por_segundo"]
        linhas += ["", f"Velocidade no processador, sem placa de vídeo: detecção em {numero(velocidade['ms_por_quadro'], 0)} ms por quadro; "
                       f"aplicação completa a cerca de {numero(na_aplicacao, 0)} quadros por segundo."]
    bloco = R_INICIO + "\n" + "\n".join(linhas) + "\n" + R_FIM
    return texto[: texto.index(R_INICIO)] + bloco + texto[texto.index(R_FIM) + len(R_FIM):]


def atualizar_readme() -> None:
    """Reescreve no README o bloco de integrantes e links, a partir de docs/equipe.json."""
    readme = RAIZ / "README.md"
    if not readme.is_file():
        return
    texto = readme.read_text(encoding="utf-8")
    if INICIO not in texto or FIM not in texto:
        return
    equipe = carregar_json(RAIZ / "docs" / "equipe.json", {})
    integrantes = [i for i in equipe.get("integrantes", []) if i.get("nome") or i.get("rm")]
    pendente = "_preencher em docs/equipe.json_"
    linhas = ["| Nome completo | RM |", "|---|---|"]
    linhas += [f"| {i.get('nome') or pendente} | {i.get('rm') or pendente} |" for i in integrantes] or [f"| {pendente} | {pendente} |"]
    linhas += ["", f"- **Vídeo de demonstração:** {equipe.get('video') or pendente}", f"- **Repositório:** {equipe.get('repositorio') or pendente}"]
    bloco = INICIO + "\n" + "\n".join(linhas) + "\n" + FIM
    novo = texto[: texto.index(INICIO)] + bloco + texto[texto.index(FIM) + len(FIM):]
    novo = _atualizar_resultados(novo)
    if novo != texto:
        readme.write_text(novo, encoding="utf-8", newline="\n")
        print(f"atualizado: {readme.name}")


def main() -> None:
    doc = SimpleDocTemplate(str(SAIDA), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Passo Livre - Detecção de obstáculos em calçadas", author="Equipe Passo Livre")
    doc.build(montar(), onFirstPage=rodape, onLaterPages=rodape)
    print(f"gerado: docs/{SAIDA.name}")
    atualizar_readme()


if __name__ == "__main__":
    main()
