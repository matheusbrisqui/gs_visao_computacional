"""Detecção em tempo real: lê a câmera ou um vídeo, detecta, aplica as regras e mostra o resultado."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

from .caminhos import PASTA_SAIDA
from .classes import POR_NOME
from .registro import RegistroDeAlertas, enviar_relato
from .regras import Alerta, Deteccao, MonitorDeAlertas, Parametros, Zona, no_caminho

TEMPO_DO_AVISO_S = 3.0  # quanto tempo a faixa de alerta fica na tela
COR_ZONA = (208, 87, 11)  # azul do produto, no formato BGR
COR_ALERTA = {"bloqueio": (30, 38, 179), "atencao": (0, 106, 178)}


class Pintor:
    """Desenha caixas, zona de passagem e avisos sobre o quadro.

    As formas são desenhadas com o OpenCV. Os textos usam o Pillow, porque as fontes embutidas
    do OpenCV não têm acentos e os rótulos são em português.
    """

    def __init__(self) -> None:
        from PIL import ImageFont

        self._fontes: dict[int, object] = {}
        self._arquivo = None
        for candidato in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"):
            try:
                ImageFont.truetype(candidato, 12)
                self._arquivo = candidato
                break
            except OSError:
                continue

    def _fonte(self, tamanho: int):
        from PIL import ImageFont

        if tamanho not in self._fontes:
            self._fontes[tamanho] = ImageFont.truetype(self._arquivo, tamanho) if self._arquivo else ImageFont.load_default()
        return self._fontes[tamanho]

    def desenhar(self, quadro, deteccoes: list[Deteccao], zona: Zona, avisos: list[Alerta], fps: float, confianca_minima: float):
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw

        altura, largura = quadro.shape[:2]
        escala = max(0.8, largura / 800)
        tela = quadro.copy()

        # zona de passagem: preenchimento translúcido e contorno
        z1, z2 = (int(zona.x1 * largura), int(zona.y1 * altura)), (int(zona.x2 * largura) - 1, int(zona.y2 * altura) - 1)
        camada = tela.copy()
        cv2.rectangle(camada, z1, z2, COR_ZONA, -1)
        cv2.addWeighted(camada, 0.10, tela, 0.90, 0, tela)
        cv2.rectangle(tela, z1, z2, COR_ZONA, max(1, round(2 * escala)))

        textos = [("Zona de passagem", (z1[0] + 6, z1[1] + 4), round(15 * escala), COR_ZONA, None)]
        for d in sorted(deteccoes, key=lambda d: d.confianca):
            classe = POR_NOME.get(d.classe)
            cor = classe.cor_bgr if classe else (90, 90, 90)
            destaque = classe is not None and classe.tipo_relato is not None and no_caminho(d, zona) and d.confianca >= confianca_minima
            p1, p2 = (int(d.x1 * largura), int(d.y1 * altura)), (int(d.x2 * largura), int(d.y2 * altura))
            cv2.rectangle(tela, p1, p2, cor, max(2, round((4 if destaque else 2) * escala)))
            rotulo = f"{classe.rotulo if classe else d.classe} {d.confianca:.0%}"
            textos.append((rotulo, (p1[0], max(0, p1[1] - round(22 * escala))), round(16 * escala), (255, 255, 255), cor))

        # faixa superior: desempenho; faixa inferior: avisos
        if fps > 0:  # em fotos não há taxa de quadros a mostrar
            textos.append((f"{fps:4.1f} quadros por segundo", (10, 8), round(16 * escala), (255, 255, 255), (40, 40, 40)))
        y_aviso = altura - round(46 * escala) * max(1, len(avisos)) - 8
        for aviso in avisos:
            prioridade = " | PRIORIDADE ALTA" if aviso.prioridade == "alta" else ""
            texto_do_aviso, tamanho_do_aviso = f"ALERTA: {aviso.mensagem}{prioridade}", round(24 * escala)
            while tamanho_do_aviso > 10 and self._fonte(tamanho_do_aviso).getlength(texto_do_aviso) > largura - 32:
                tamanho_do_aviso -= 1  # o aviso nunca passa da largura da tela
            textos.append((texto_do_aviso, (10, y_aviso), tamanho_do_aviso, (255, 255, 255), COR_ALERTA[aviso.nivel]))
            y_aviso += round(46 * escala)

        imagem = Image.fromarray(cv2.cvtColor(tela, cv2.COLOR_BGR2RGB))
        pincel = ImageDraw.Draw(imagem)
        for texto, (x, y), tamanho, cor_texto, cor_fundo in textos:
            fonte = self._fonte(tamanho)
            if cor_fundo is not None:
                _, _, x2, y2 = pincel.textbbox((x + 5, y + 2), texto, font=fonte)
                pincel.rectangle((x, y, x2 + 5, y2 + 4), fill=cor_fundo[::-1])
                pincel.text((x + 5, y + 2), texto, font=fonte, fill=cor_texto[::-1])
            else:
                pincel.text((x, y), texto, font=fonte, fill=cor_texto[::-1], stroke_width=2, stroke_fill=(255, 255, 255))
        return cv2.cvtColor(np.asarray(imagem), cv2.COLOR_RGB2BGR)


def executar(
    fonte: int | str,
    modelo: str | None = None,
    confianca: float = 0.40,
    tamanho: int = 416,
    janela: bool = True,
    gravar: str | None = None,
    limite: int = 0,
    enviar_para: str | None = None,
    posicao: tuple[float, float] | None = None,
    pasta_saida: Path = PASTA_SAIDA,
) -> int:
    """Laço principal. `fonte` é o número da câmera, um arquivo de vídeo ou o endereço de uma câmera de rede."""
    import cv2

    from .detector import Detector

    detector = Detector(modelo, confianca=confianca, tamanho=tamanho)
    eh_camera = isinstance(fonte, int)
    captura = cv2.VideoCapture(fonte, cv2.CAP_DSHOW) if eh_camera else cv2.VideoCapture(str(fonte))
    if not captura.isOpened():
        print(f"Não foi possível abrir a fonte de vídeo: {fonte}")
        return 2
    fps_da_fonte = captura.get(cv2.CAP_PROP_FPS) or 0.0
    eh_arquivo = not eh_camera and Path(str(fonte)).is_file()

    monitor = MonitorDeAlertas(Parametros(confianca_minima=confianca))
    registro = RegistroDeAlertas(pasta_saida)
    pintor = Pintor()
    gravador = None
    avisos: list[tuple[float, Alerta]] = []
    duracoes: deque[float] = deque(maxlen=30)
    titulo = "Passo Livre - detector de obstaculos (Q ou Esc para sair)"
    print(f"Modelo: {detector.caminho.name} | fonte: {fonte} | confiança mínima: {confianca:.2f} | tamanho: {tamanho}")

    quadros, inicio = 0, time.monotonic()
    try:
        while True:
            antes = time.monotonic()
            ok, quadro = captura.read()
            if not ok:
                break
            quadros += 1
            # relógio das regras: tempo do próprio vídeo para arquivos, tempo real para câmeras
            instante = quadros / fps_da_fonte if eh_arquivo and fps_da_fonte > 0 else time.monotonic() - inicio

            deteccoes = detector.detectar(quadro)
            novos = monitor.processar(deteccoes, instante)
            avisos = [(t, a) for t, a in avisos if instante - t < TEMPO_DO_AVISO_S] + [(instante, a) for a in novos]
            fps = len(duracoes) / sum(duracoes) if duracoes else 0.0
            tela = pintor.desenhar(quadro, deteccoes, monitor.parametros.zona, [a for _, a in avisos], fps, confianca)

            for alerta in novos:
                evento = registro.registrar(alerta, tela)
                texto = f"[{instante:7.2f} s] ALERTA: {alerta.mensagem} | nível {alerta.nivel} | prioridade {alerta.prioridade}"
                if enviar_para and posicao:
                    numero = enviar_relato(enviar_para, alerta, *posicao)
                    texto += f" | relato #{numero} criado no aplicativo" if numero else " | aplicativo indisponível, alerta só registrado"
                print(texto + (f" | {evento['captura']}" if evento["captura"] else ""))

            if gravar:
                if gravador is None:
                    Path(gravar).parent.mkdir(parents=True, exist_ok=True)
                    altura, largura = tela.shape[:2]
                    ritmo = fps_da_fonte if eh_arquivo and fps_da_fonte > 0 else 15.0
                    gravador = cv2.VideoWriter(gravar, cv2.VideoWriter_fourcc(*"mp4v"), ritmo, (largura, altura))
                gravador.write(tela)
            if janela:
                cv2.imshow(titulo, tela)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
            duracoes.append(time.monotonic() - antes)
            if limite and quadros >= limite:
                break
    except KeyboardInterrupt:
        pass
    finally:
        captura.release()
        if gravador is not None:
            gravador.release()
        if janela:
            cv2.destroyAllWindows()

    total = time.monotonic() - inicio
    print(f"Fim: {quadros} quadros em {total:.1f} s ({quadros / total if total else 0:.1f} quadros por segundo), {registro.total} alerta(s)")
    if registro.total:
        print(f"Alertas gravados em {registro.arquivo}")
    return 0 if quadros else 2


def detectar_em_imagem(arquivo: str, modelo: str | None = None, confianca: float = 0.40, tamanho: int = 416, saida: str | None = None) -> int:
    """Detecta em uma foto, grava a foto anotada e lista o que foi encontrado."""
    import cv2

    from .detector import Detector

    quadro = cv2.imread(arquivo)
    if quadro is None:
        print(f"Não foi possível ler a imagem: {arquivo}")
        return 2
    detector = Detector(modelo, confianca=confianca, tamanho=tamanho)
    deteccoes = detector.detectar(quadro)
    zona = Zona()
    tela = Pintor().desenhar(quadro, deteccoes, zona, [], 0.0, confianca)
    destino = Path(saida) if saida else PASTA_SAIDA / f"{Path(arquivo).stem}_detectado.jpg"
    destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), tela)
    for d in sorted(deteccoes, key=lambda d: -d.confianca):
        situacao = "no caminho" if no_caminho(d, zona) else "fora da zona de passagem"
        print(f"  {POR_NOME[d.classe].rotulo if d.classe in POR_NOME else d.classe}: {d.confianca:.0%} ({situacao})")
    print(f"{len(deteccoes)} objeto(s) detectado(s). Imagem gravada em {destino}")
    return 0
