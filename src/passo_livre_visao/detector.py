"""Carrega o modelo treinado e devolve as detecções de um quadro no formato usado pelas regras."""

from __future__ import annotations

from pathlib import Path

from .caminhos import MODELO_FINAL
from .regras import Deteccao


def desligar_telemetria() -> None:
    """A biblioteca Ultralytics envia estatísticas de uso anônimas por padrão. Aqui isso fica desligado."""
    try:
        from ultralytics import settings

        if settings.get("sync", True):
            settings.update({"sync": False})
    except Exception:
        pass


class Detector:
    def __init__(self, modelo: str | Path | None = None, confianca: float = 0.40, tamanho: int = 416) -> None:
        caminho = Path(modelo) if modelo else MODELO_FINAL
        if not caminho.is_file():
            raise FileNotFoundError(
                f"Modelo não encontrado em {caminho}. Treine com `python visao.py treinar` ou informe --modelo."
            )
        from ultralytics import YOLO  # importado aqui: carregar o PyTorch leva alguns segundos

        desligar_telemetria()
        self.caminho = caminho
        self.confianca = confianca
        self.tamanho = tamanho
        self._modelo = YOLO(str(caminho))
        self.nomes: dict[int, str] = dict(self._modelo.names)

    def detectar(self, quadro) -> list[Deteccao]:
        """Recebe uma imagem no formato do OpenCV (matriz BGR) e devolve as caixas encontradas."""
        resultado = self._modelo.predict(quadro, imgsz=self.tamanho, conf=self.confianca, device="cpu", verbose=False)[0]
        caixas = resultado.boxes
        deteccoes = []
        for (x1, y1, x2, y2), confianca, classe in zip(
            caixas.xyxyn.tolist(), caixas.conf.tolist(), caixas.cls.tolist(), strict=True
        ):
            deteccoes.append(Deteccao(self.nomes[int(classe)], round(float(confianca), 4), x1, y1, x2, y2))
        return deteccoes
