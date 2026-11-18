"""Registro dos alertas: um arquivo de eventos por execução e a foto do momento de cada alerta.

Tudo fica na pasta `saida/`, só no computador de quem roda. Nada é enviado para fora, a não ser
que o envio ao aplicativo Passo Livre seja ligado de propósito (opção --enviar-para).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .regras import Alerta


class RegistroDeAlertas:
    def __init__(self, pasta: Path, execucao: str | None = None) -> None:
        self.execucao = execucao or datetime.now().strftime("%Y%m%dT%H%M%S")
        self.pasta = pasta
        self.arquivo = pasta / f"alertas_{self.execucao}.jsonl"
        self.pasta_capturas = pasta / "capturas"
        self.total = 0

    def registrar(self, alerta: Alerta, quadro_anotado=None) -> dict:
        """Grava o alerta e, se receber o quadro, a foto correspondente. Devolve o evento gravado."""
        self.pasta.mkdir(parents=True, exist_ok=True)
        evento = {"execucao": self.execucao, "momento": datetime.now().isoformat(timespec="seconds"), **asdict(alerta),
                  "mensagem": alerta.mensagem, "captura": None}
        if quadro_anotado is not None:
            import cv2  # importado aqui para os testes do registro não dependerem do OpenCV

            self.pasta_capturas.mkdir(parents=True, exist_ok=True)
            captura = self.pasta_capturas / f"{self.execucao}_{alerta.quadro:06d}_{alerta.classe}.jpg"
            if cv2.imwrite(str(captura), quadro_anotado):
                evento["captura"] = f"capturas/{captura.name}"
        with self.arquivo.open("a", encoding="utf-8") as arquivo:
            arquivo.write(json.dumps(evento, ensure_ascii=False) + "\n")
        self.total += 1
        return evento


def enviar_relato(url_api: str, alerta: Alerta, lat: float, lon: float, tempo_limite_s: float = 5.0) -> int | None:
    """Cria um relato no aplicativo Passo Livre (POST /relatos). Devolve o número do relato ou None se falhar.

    Vai apenas o tipo do obstáculo, a posição informada e uma descrição gerada. Nenhuma imagem é enviada,
    e o perfil de quem relata fica sem preenchimento: a câmera não deduz dado pessoal de ninguém.
    """
    corpo = {
        "tipo": alerta.tipo_relato,
        "lat": lat,
        "lon": lon,
        "descricao": f"Detectado pela câmera: {alerta.mensagem.lower()} (confiança {alerta.confianca:.0%})",
    }
    requisicao = urllib.request.Request(
        url_api.rstrip("/") + "/relatos",
        data=json.dumps(corpo).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(requisicao, timeout=tempo_limite_s) as resposta:
            return json.loads(resposta.read().decode("utf-8")).get("id")
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
        return None
