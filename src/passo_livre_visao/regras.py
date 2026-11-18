"""Da detecção ao alerta: decide quando um objeto visto pela câmera é um obstáculo no caminho.

O modelo só diz "há um poste aqui, com 80% de certeza". Estas regras transformam isso em
informação útil para quem tem mobilidade reduzida:

1. Zona de passagem: só interessa o que está no trecho da calçada por onde a pessoa vai passar.
2. Persistência: o objeto precisa aparecer em vários quadros seguidos, para não alertar por engano.
3. Nível: quanto da largura da passagem o obstáculo ocupa (atenção ou bloqueio).
4. Prioridade: se há uma pessoa em cadeira de rodas na cena, o alerta sai com prioridade alta.
5. Intervalo: o mesmo tipo de obstáculo não gera alertas repetidos em sequência.

Todas as funções são puras ou dependem só do próprio estado, para serem testadas sem câmera.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .classes import CLASSE_PRIORITARIA, POR_NOME, eh_obstaculo


@dataclass(frozen=True)
class Deteccao:
    """Uma caixa devolvida pelo modelo. Coordenadas normalizadas de 0 a 1 (x cresce para a direita, y para baixo)."""

    classe: str
    confianca: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def largura(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def ponto_de_apoio(self) -> tuple[float, float]:
        """Centro da base da caixa: onde o objeto toca o chão."""
        return ((self.x1 + self.x2) / 2, self.y2)


@dataclass(frozen=True)
class Zona:
    """Trecho da imagem por onde a pessoa vai passar: a faixa central e inferior do quadro."""

    x1: float = 0.2
    y1: float = 0.45
    x2: float = 0.8
    y2: float = 1.0

    @property
    def largura(self) -> float:
        return self.x2 - self.x1

    def contem(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass(frozen=True)
class Parametros:
    confianca_minima: float = 0.40
    janela_quadros: int = 12  # quantos quadros recentes são observados
    minimo_na_janela: int = 8  # em quantos deles o obstáculo precisa aparecer
    ocupacao_bloqueio: float = 0.50  # fração da largura da passagem a partir da qual é bloqueio
    intervalo_s: float = 8.0  # tempo mínimo entre dois alertas do mesmo tipo
    zona: Zona = field(default_factory=Zona)


@dataclass(frozen=True)
class Alerta:
    classe: str
    rotulo: str
    tipo_relato: str
    nivel: str  # "atencao" ou "bloqueio"
    prioridade: str  # "alta" ou "normal"
    confianca: float  # média nos quadros em que apareceu
    ocupacao: float  # fração da largura da passagem ocupada
    quadro: int
    instante_s: float

    @property
    def mensagem(self) -> str:
        situacao = "bloqueando a passagem" if self.nivel == "bloqueio" else "no caminho"
        return f"{self.rotulo} {situacao}"


def no_caminho(deteccao: Deteccao, zona: Zona) -> bool:
    """O objeto está no caminho quando o ponto em que toca o chão cai dentro da zona de passagem."""
    return zona.contem(*deteccao.ponto_de_apoio)


def ocupacao_da_passagem(deteccao: Deteccao, zona: Zona) -> float:
    """Fração (0 a 1) da largura da zona de passagem coberta pela caixa do objeto."""
    sobreposicao = min(deteccao.x2, zona.x2) - max(deteccao.x1, zona.x1)
    if zona.largura <= 0:
        return 0.0
    return max(0.0, min(1.0, sobreposicao / zona.largura))


def nivel_do_obstaculo(ocupacao: float, limite_bloqueio: float) -> str:
    return "bloqueio" if ocupacao >= limite_bloqueio else "atencao"


def obstaculos_no_caminho(deteccoes: list[Deteccao], parametros: Parametros) -> list[Deteccao]:
    """Filtra o que interessa em um quadro: obstáculo, com confiança suficiente e dentro da zona."""
    return [
        d
        for d in deteccoes
        if eh_obstaculo(d.classe) and d.confianca >= parametros.confianca_minima and no_caminho(d, parametros.zona)
    ]


class MonitorDeAlertas:
    """Acompanha os quadros em sequência e decide quando emitir um alerta."""

    def __init__(self, parametros: Parametros | None = None) -> None:
        self.parametros = parametros or Parametros()
        # por classe: histórico dos últimos quadros (None = não apareceu; senão, (confiança, ocupação))
        self._historico: dict[str, deque[tuple[float, float] | None]] = {}
        self._ultimo_alerta_s: dict[str, float] = {}
        self._quadro = 0

    def processar(self, deteccoes: list[Deteccao], instante_s: float) -> list[Alerta]:
        """Recebe as detecções de um quadro e devolve os alertas disparados nele (normalmente nenhum)."""
        p = self.parametros
        self._quadro += 1
        no_quadro: dict[str, tuple[float, float]] = {}
        for d in obstaculos_no_caminho(deteccoes, p):
            ocupacao = ocupacao_da_passagem(d, p.zona)
            atual = no_quadro.get(d.classe)
            # se houver mais de um objeto da mesma classe, vale o que mais ocupa a passagem
            if atual is None or ocupacao > atual[1]:
                no_quadro[d.classe] = (d.confianca, ocupacao)

        ha_pessoa_prioritaria = any(
            d.classe == CLASSE_PRIORITARIA and d.confianca >= p.confianca_minima for d in deteccoes
        )

        alertas = []
        for classe in set(self._historico) | set(no_quadro):
            historico = self._historico.setdefault(classe, deque(maxlen=p.janela_quadros))
            historico.append(no_quadro.get(classe))
            presentes = [h for h in historico if h is not None]
            if classe not in no_quadro or len(presentes) < p.minimo_na_janela:
                continue
            ultimo = self._ultimo_alerta_s.get(classe)
            if ultimo is not None and instante_s - ultimo < p.intervalo_s:
                continue
            self._ultimo_alerta_s[classe] = instante_s
            ocupacao = max(o for _, o in presentes)
            definicao = POR_NOME[classe]
            alertas.append(
                Alerta(
                    classe=classe,
                    rotulo=definicao.rotulo,
                    tipo_relato=definicao.tipo_relato or "outro",
                    nivel=nivel_do_obstaculo(ocupacao, p.ocupacao_bloqueio),
                    prioridade="alta" if ha_pessoa_prioritaria else "normal",
                    confianca=round(sum(c for c, _ in presentes) / len(presentes), 3),
                    ocupacao=round(ocupacao, 3),
                    quadro=self._quadro,
                    instante_s=round(instante_s, 2),
                )
            )
        return sorted(alertas, key=lambda a: (a.nivel != "bloqueio", -a.ocupacao))
