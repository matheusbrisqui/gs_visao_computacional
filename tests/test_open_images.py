"""Montagem do conjunto de treino: leitura das anotações, seleção equilibrada, partição e formato YOLO."""

from collections import Counter
from pathlib import Path

import pytest

from passo_livre_visao import open_images as oi
from passo_livre_visao.classes import CLASSES, NOMES, POR_NOME, classe_de_origem, eh_obstaculo

# Códigos fictícios no formato do Open Images, um para cada categoria usada
CODIGOS_FICTICIOS = [[f"/m/{i:04d}", nome] for i, nome in enumerate(sorted(classe_de_origem()))]
CODIGO = {nome: codigo for codigo, nome in CODIGOS_FICTICIOS}


def linha(imagem, categoria, x_min=0.1, x_max=0.5, y_min=0.2, y_max=0.8, grupo="0", representacao="0"):
    return {"ImageID": imagem, "LabelName": CODIGO.get(categoria, "/m/outra"), "XMin": str(x_min), "XMax": str(x_max),
            "YMin": str(y_min), "YMax": str(y_max), "IsGroupOf": grupo, "IsDepiction": representacao}


@pytest.fixture
def codigos():
    return oi.ler_codigos_das_categorias([["LabelName", "DisplayName"], *CODIGOS_FICTICIOS, ["/m/outra", "Cat"]])


# ---------------------------------------------------------------- taxonomia
def test_indices_das_classes_sao_sequenciais_e_nomes_unicos():
    assert [c.indice for c in CLASSES] == list(range(len(CLASSES)))
    assert len(set(NOMES)) == len(NOMES)


def test_cada_categoria_de_origem_pertence_a_uma_unica_classe():
    todas = [origem for c in CLASSES for origem in c.origem_open_images]
    assert len(todas) == len(set(todas))


def test_tipos_de_relato_pertencem_ao_dominio_do_aplicativo():
    tipos_do_aplicativo = {"buraco", "carro_na_calcada", "entulho_lixo", "poste_obstaculo", "degrau_sem_rampa", "outro"}
    assert {c.tipo_relato for c in CLASSES if c.tipo_relato} <= tipos_do_aplicativo


def test_cadeira_de_rodas_nao_e_obstaculo():
    assert not eh_obstaculo("cadeira_de_rodas")
    assert eh_obstaculo("veiculo")
    assert not eh_obstaculo("classe_inexistente")


# ---------------------------------------------------------------- leitura
def test_categorias_sao_agrupadas_na_classe_do_produto(codigos):
    assert codigos[CODIGO["Car"]] == codigos[CODIGO["Van"]] == POR_NOME["veiculo"].indice
    assert codigos[CODIGO["Wheelchair"]] == POR_NOME["cadeira_de_rodas"].indice
    assert "/m/outra" not in codigos


def test_categoria_ausente_no_banco_gera_erro_claro():
    sem_escada = [c for c in CODIGOS_FICTICIOS if c[1] != "Stairs"]
    with pytest.raises(ValueError, match="Stairs"):
        oi.ler_codigos_das_categorias(sem_escada)


def test_caixas_sao_agrupadas_por_imagem_e_outras_categorias_ignoradas(codigos):
    caixas = oi.ler_caixas([linha("a", "Car"), linha("a", "Stairs"), linha("a", "Cat"), linha("b", "Bicycle")], codigos)
    assert sorted(caixas) == ["a", "b"]
    assert sorted(c.classe for c in caixas["a"]) == [POR_NOME["veiculo"].indice, POR_NOME["escada_degrau"].indice]


def test_imagem_com_caixa_de_grupo_ou_desenho_e_descartada_por_inteiro(codigos):
    linhas = [linha("a", "Car"), linha("a", "Car", grupo="1"), linha("b", "Stairs", representacao="1"), linha("c", "Stairs")]
    assert sorted(oi.ler_caixas(linhas, codigos)) == ["c"]


def test_caixa_invalida_e_ignorada_e_coordenadas_ficam_entre_0_e_1(codigos):
    linhas = [linha("a", "Car", x_min=0.5, x_max=0.5), linha("a", "Car", x_min=-0.1, x_max=1.2)]
    (caixa,) = oi.ler_caixas(linhas, codigos)["a"]
    assert (caixa.x_min, caixa.x_max) == (0.0, 1.0)


# ---------------------------------------------------------------- formato YOLO
def test_conversao_para_o_formato_yolo():
    assert oi.Caixa(3, 0.1, 0.5, 0.2, 0.8).para_yolo() == "3 0.300000 0.500000 0.400000 0.600000"


def test_data_yaml_lista_todas_as_classes_na_ordem():
    yaml = oi.conteudo_data_yaml(Path("C:/dados/dataset"))
    assert "path: C:/dados/dataset" in yaml
    assert "train: train.txt" in yaml
    for classe in CLASSES:
        assert f"  {classe.indice}: {classe.nome}" in yaml


# ---------------------------------------------------------------- seleção e partição
def conjunto_desequilibrado(codigos):
    linhas = [linha(f"carro{i}", "Car") for i in range(300)]
    linhas += [linha(f"escada{i}", "Stairs") for i in range(40)]
    linhas += [linha(f"misto{i}", "Wheelchair") for i in range(20)] + [linha(f"misto{i}", "Car") for i in range(20)]
    return oi.ler_caixas(linhas, codigos)


def test_selecao_limita_a_classe_dominante_e_mantem_as_raras(codigos):
    caixas = conjunto_desequilibrado(codigos)
    escolhidas = oi.selecionar_equilibrado(caixas, maximo_por_classe=50)
    resumo = oi.resumo_por_classe(caixas, escolhidas)
    assert resumo["veiculo"]["imagens"] == 50  # conta também os carros das imagens mistas
    assert resumo["escada_degrau"]["imagens"] == 40  # classe rara entra inteira
    assert resumo["cadeira_de_rodas"]["imagens"] == 20
    assert len(escolhidas) == len(set(escolhidas)) == 90


def test_selecao_e_reproduzivel(codigos):
    caixas = conjunto_desequilibrado(codigos)
    assert oi.selecionar_equilibrado(caixas, 50) == oi.selecionar_equilibrado(dict(reversed(list(caixas.items()))), 50)


def test_particao_e_estavel_e_respeita_as_proporcoes():
    imagens = [f"img{i:05d}" for i in range(5000)]
    partes = Counter(oi.particao(i) for i in imagens)
    assert all(oi.particao(i) == oi.particao(i) for i in imagens[:50])
    assert set(partes) == {"train", "val", "test"}
    assert 0.77 < partes["train"] / 5000 < 0.83
    assert 0.08 < partes["val"] / 5000 < 0.12
    assert 0.08 < partes["test"] / 5000 < 0.12


# ---------------------------------------------------------------- reforço das classes raras
def test_imagens_de_classes_raras_sao_repetidas_no_treino(codigos):
    caixas = conjunto_desequilibrado(codigos)
    fatores = oi.fatores_de_repeticao(caixas, list(caixas), alvo=120, maximo=4)
    assert fatores["carro0"] == 1  # classe comum: 320 imagens, sem repetição
    assert fatores["escada0"] == 3  # 40 imagens: 120 / 40
    assert fatores["misto0"] == 4  # cadeira de rodas: 20 imagens dariam 6, limitado a 4
    assert set(fatores.values()) <= {1, 2, 3, 4}
