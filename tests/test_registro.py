"""Registro dos alertas em arquivo e envio opcional ao aplicativo."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from passo_livre_visao.registro import RegistroDeAlertas, enviar_relato
from passo_livre_visao.regras import Alerta

ALERTA = Alerta(classe="poste_placa", rotulo="Poste ou placa", tipo_relato="poste_obstaculo", nivel="bloqueio",
                prioridade="alta", confianca=0.82, ocupacao=0.61, quadro=37, instante_s=3.7)


def test_alerta_e_gravado_como_uma_linha_json(tmp_path):
    registro = RegistroDeAlertas(tmp_path, execucao="20261112T100000")
    registro.registrar(ALERTA)
    registro.registrar(ALERTA)
    linhas = (tmp_path / "alertas_20261112T100000.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(linhas) == registro.total == 2
    evento = json.loads(linhas[0])
    assert evento["classe"] == "poste_placa"
    assert evento["tipo_relato"] == "poste_obstaculo"
    assert evento["nivel"] == "bloqueio" and evento["prioridade"] == "alta"
    assert evento["mensagem"] == "Poste ou placa bloqueando a passagem"
    assert evento["captura"] is None  # sem quadro, não há foto


def test_nada_e_gravado_sem_alerta(tmp_path):
    RegistroDeAlertas(tmp_path / "saida")
    assert not (tmp_path / "saida").exists()


@pytest.fixture
def aplicativo_falso():
    """Servidor HTTP mínimo que imita o POST /relatos do aplicativo e guarda o que recebeu."""
    recebidos = []

    class Tratador(BaseHTTPRequestHandler):
        def do_POST(self):
            corpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            recebidos.append((self.path, corpo))
            resposta = json.dumps({"id": 42, **corpo}).encode()
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resposta)))
            self.end_headers()
            self.wfile.write(resposta)

        def log_message(self, *args):
            pass

    servidor = HTTPServer(("127.0.0.1", 0), Tratador)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{servidor.server_port}", recebidos
    servidor.shutdown()
    servidor.server_close()


def test_envio_cria_relato_no_aplicativo_sem_imagem_e_sem_dado_pessoal(aplicativo_falso):
    url, recebidos = aplicativo_falso
    assert enviar_relato(url + "/", ALERTA, -23.5613, -46.6558) == 42
    caminho, corpo = recebidos[0]
    assert caminho == "/relatos"
    assert corpo["tipo"] == "poste_obstaculo"
    assert (corpo["lat"], corpo["lon"]) == (-23.5613, -46.6558)
    assert "bloqueando a passagem" in corpo["descricao"]
    assert set(corpo) == {"tipo", "lat", "lon", "descricao"}  # nem foto, nem perfil, nem identificação


def test_aplicativo_fora_do_ar_nao_derruba_a_deteccao():
    assert enviar_relato("http://127.0.0.1:9", ALERTA, -23.56, -46.65, tempo_limite_s=1) is None
