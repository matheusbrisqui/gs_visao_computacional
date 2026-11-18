"""Classes que o modelo aprende e como cada uma vira informação para o cidadão.

O modelo é treinado com uma taxonomia própria do produto, e não com as classes genéricas
dos modelos prontos: várias categorias do banco de imagens são agrupadas em uma única classe
de obstáculo, porque para quem anda de cadeira de rodas tanto faz se o que bloqueia a calçada
é um carro ou uma van.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Classe:
    indice: int
    nome: str  # nome usado no modelo e nos arquivos de anotação
    rotulo: str  # texto exibido na tela e falado no alerta
    tipo_relato: str | None  # tipo de obstáculo no aplicativo Passo Livre; None = não é obstáculo
    origem_open_images: tuple[str, ...]  # categorias do Open Images agrupadas nesta classe
    cor_bgr: tuple[int, int, int]  # cor da caixa na tela (formato do OpenCV: azul, verde, vermelho)


CLASSES: tuple[Classe, ...] = (
    Classe(0, "veiculo", "Veículo", "carro_na_calcada", ("Car", "Van", "Taxi", "Truck"), (30, 38, 179)),
    Classe(1, "moto", "Moto", "carro_na_calcada", ("Motorcycle",), (6, 119, 217)),
    Classe(2, "bicicleta", "Bicicleta", "outro", ("Bicycle",), (153, 138, 15)),
    Classe(3, "lixeira_entulho", "Lixeira ou entulho", "entulho_lixo", ("Waste container",), (3, 74, 124)),
    Classe(4, "poste_placa", "Poste ou placa", "poste_obstaculo",
           ("Street light", "Traffic sign", "Stop sign", "Parking meter", "Fire hydrant"), (168, 33, 107)),
    Classe(5, "escada_degrau", "Escada ou degrau", "degrau_sem_rampa", ("Stairs",), (144, 116, 14)),
    # Não é obstáculo: indica que há uma pessoa em cadeira de rodas na cena e eleva a prioridade do alerta.
    Classe(6, "cadeira_de_rodas", "Pessoa em cadeira de rodas", None, ("Wheelchair",), (59, 127, 27)),
)

NOMES: tuple[str, ...] = tuple(c.nome for c in CLASSES)
POR_NOME: dict[str, Classe] = {c.nome: c for c in CLASSES}
POR_INDICE: dict[int, Classe] = {c.indice: c for c in CLASSES}
CLASSE_PRIORITARIA = "cadeira_de_rodas"


def classe_de_origem() -> dict[str, Classe]:
    """Mapa: categoria do Open Images -> classe do produto."""
    return {origem: classe for classe in CLASSES for origem in classe.origem_open_images}


def eh_obstaculo(nome: str) -> bool:
    classe = POR_NOME.get(nome)
    return classe is not None and classe.tipo_relato is not None
