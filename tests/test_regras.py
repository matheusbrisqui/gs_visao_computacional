"""Regras que transformam detecções em alertas: zona de passagem, persistência, nível, prioridade e intervalo."""

import pytest

from passo_livre_visao.regras import (
    Deteccao,
    MonitorDeAlertas,
    Parametros,
    Zona,
    nivel_do_obstaculo,
    no_caminho,
    obstaculos_no_caminho,
    ocupacao_da_passagem,
)

ZONA = Zona()  # x de 0,2 a 0,8; y de 0,45 a 1,0


def poste(confianca=0.8, x1=0.45, x2=0.55, y2=0.9) -> Deteccao:
    return Deteccao("poste_placa", confianca, x1, 0.2, x2, y2)


def carro_largo(confianca=0.9) -> Deteccao:
    return Deteccao("veiculo", confianca, 0.25, 0.5, 0.75, 0.95)


def cadeira(confianca=0.85) -> Deteccao:
    return Deteccao("cadeira_de_rodas", confianca, 0.05, 0.5, 0.18, 0.9)


def quadros(monitor: MonitorDeAlertas, deteccoes, n: int, inicio_s: float = 0.0, fps: float = 10.0):
    """Alimenta n quadros iguais e devolve todos os alertas emitidos."""
    alertas = []
    for i in range(n):
        alertas += monitor.processar(deteccoes, inicio_s + i / fps)
    return alertas


# ---------------------------------------------------------------- zona de passagem
def test_objeto_no_centro_inferior_esta_no_caminho():
    assert no_caminho(poste(), ZONA)


def test_objeto_na_lateral_nao_esta_no_caminho():
    assert not no_caminho(poste(x1=0.02, x2=0.12), ZONA)


def test_objeto_distante_no_alto_da_imagem_nao_esta_no_caminho():
    assert not no_caminho(poste(y2=0.30), ZONA)


def test_ocupacao_e_a_fracao_da_largura_da_passagem():
    assert ocupacao_da_passagem(carro_largo(), ZONA) == pytest.approx(0.5 / 0.6)
    assert ocupacao_da_passagem(poste(), ZONA) == pytest.approx(0.1 / 0.6)
    assert ocupacao_da_passagem(poste(x1=0.0, x2=1.0), ZONA) == 1.0  # nunca passa de 100%
    assert ocupacao_da_passagem(poste(x1=0.0, x2=0.1), ZONA) == 0.0  # fora da zona


def test_nivel_muda_no_limite_de_bloqueio():
    assert nivel_do_obstaculo(0.49, 0.5) == "atencao"
    assert nivel_do_obstaculo(0.50, 0.5) == "bloqueio"


def test_filtro_ignora_baixa_confianca_fora_da_zona_e_quem_nao_e_obstaculo():
    deteccoes = [poste(), poste(confianca=0.2), poste(x1=0.02, x2=0.1), Deteccao("cadeira_de_rodas", 0.9, 0.4, 0.5, 0.6, 0.9)]
    assert obstaculos_no_caminho(deteccoes, Parametros()) == [poste()]


# ---------------------------------------------------------------- persistência
def test_deteccao_isolada_nao_gera_alerta():
    monitor = MonitorDeAlertas()
    assert quadros(monitor, [poste()], 7) == []


def test_alerta_sai_quando_o_obstaculo_persiste():
    monitor = MonitorDeAlertas()
    alertas = quadros(monitor, [poste()], 8)
    assert len(alertas) == 1
    alerta = alertas[0]
    assert (alerta.classe, alerta.tipo_relato) == ("poste_placa", "poste_obstaculo")
    assert (alerta.nivel, alerta.prioridade) == ("atencao", "normal")
    assert alerta.quadro == 8
    assert alerta.confianca == pytest.approx(0.8)
    assert alerta.mensagem == "Poste ou placa no caminho"


def test_falhas_curtas_de_deteccao_nao_impedem_o_alerta():
    """O modelo perde o objeto em alguns quadros; a janela (8 de 12) tolera isso."""
    monitor = MonitorDeAlertas()
    alertas = []
    for i in range(12):
        visto = i not in (2, 5, 9)
        alertas += monitor.processar([poste()] if visto else [], i / 10)
    assert len(alertas) == 1


def test_objeto_que_sumiu_nao_gera_alerta_atrasado():
    monitor = MonitorDeAlertas()
    quadros(monitor, [poste()], 7)
    assert quadros(monitor, [], 5, inicio_s=0.7) == []


# ---------------------------------------------------------------- nível e prioridade
def test_obstaculo_largo_e_classificado_como_bloqueio():
    alertas = quadros(MonitorDeAlertas(), [carro_largo()], 8)
    assert alertas[0].nivel == "bloqueio"
    assert alertas[0].tipo_relato == "carro_na_calcada"
    assert alertas[0].mensagem == "Veículo bloqueando a passagem"


def test_pessoa_em_cadeira_de_rodas_eleva_a_prioridade():
    alertas = quadros(MonitorDeAlertas(), [poste(), cadeira()], 8)
    assert [a.prioridade for a in alertas] == ["alta"]
    assert all(a.classe != "cadeira_de_rodas" for a in alertas)  # a pessoa nunca vira obstáculo


def test_cadeira_de_rodas_com_baixa_confianca_nao_eleva_a_prioridade():
    alertas = quadros(MonitorDeAlertas(), [poste(), cadeira(confianca=0.2)], 8)
    assert alertas[0].prioridade == "normal"


def test_bloqueio_vem_antes_de_atencao_quando_ha_dois_obstaculos():
    alertas = quadros(MonitorDeAlertas(), [poste(), carro_largo()], 8)
    assert [a.nivel for a in alertas] == ["bloqueio", "atencao"]


# ---------------------------------------------------------------- intervalo entre alertas
def test_mesmo_obstaculo_nao_repete_alerta_dentro_do_intervalo():
    monitor = MonitorDeAlertas()
    alertas = quadros(monitor, [poste()], 60)  # 6 segundos a 10 quadros por segundo
    assert len(alertas) == 1


def test_alerta_repete_depois_do_intervalo():
    monitor = MonitorDeAlertas()
    alertas = quadros(monitor, [poste()], 100)  # 10 segundos
    assert len(alertas) == 2
    assert alertas[1].instante_s - alertas[0].instante_s >= 8.0


def test_intervalo_e_por_tipo_de_obstaculo():
    monitor = MonitorDeAlertas()
    primeiro = quadros(monitor, [poste()], 8)
    segundo = quadros(monitor, [poste(), carro_largo()], 8, inicio_s=0.8)
    assert [a.classe for a in primeiro] == ["poste_placa"]
    assert [a.classe for a in segundo] == ["veiculo"]


def test_parametros_configuraveis():
    rigoroso = MonitorDeAlertas(Parametros(confianca_minima=0.9))
    assert quadros(rigoroso, [poste(confianca=0.8)], 20) == []
    rapido = MonitorDeAlertas(Parametros(janela_quadros=3, minimo_na_janela=2))
    assert len(quadros(rapido, [poste()], 2)) == 1
