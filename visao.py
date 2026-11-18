"""Passo Livre, detector de obstáculos em calçadas: comando único do projeto.

    python visao.py camera            detecção em tempo real pela webcam
    python visao.py video ARQUIVO     detecção em tempo real em um vídeo gravado
    python visao.py imagem ARQUIVO    detecção em uma foto
    python visao.py dataset           baixa e monta o conjunto de treino
    python visao.py treinar           treina o modelo
    python visao.py avaliar           mede o modelo no conjunto de teste e gera os resultados

Use `python visao.py COMANDO --help` para ver as opções de cada comando.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # funciona sem instalar o pacote


def _opcoes_de_deteccao(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--modelo", help="arquivo .pt do modelo (padrão: modelo/passo_livre.pt)")
    parser.add_argument("--confianca", type=float, default=0.40, help="confiança mínima de 0 a 1 (padrão: 0.40)")
    parser.add_argument("--tamanho", type=int, default=416, help="lado da imagem analisada, em pixels (padrão: 416)")
    parser.add_argument("--sem-janela", action="store_true", help="não abre janela; útil para gravar ou testar")
    parser.add_argument("--gravar", metavar="ARQUIVO.mp4", help="grava o vídeo com as caixas e os alertas")
    parser.add_argument("--limite", type=int, default=0, help="para depois de N quadros (0 = sem limite)")
    parser.add_argument("--enviar-para", metavar="URL", help="cria um relato no aplicativo Passo Livre a cada alerta")
    parser.add_argument("--lat", type=float, help="latitude do local filmado (usada com --enviar-para)")
    parser.add_argument("--lon", type=float, help="longitude do local filmado (usada com --enviar-para)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    comandos = parser.add_subparsers(dest="comando", required=True)

    camera = comandos.add_parser("camera", help="detecção em tempo real pela webcam")
    camera.add_argument("--indice", type=int, default=0, help="número da câmera (padrão: 0)")
    _opcoes_de_deteccao(camera)

    video = comandos.add_parser("video", help="detecção em tempo real em um vídeo gravado ou endereço de câmera")
    video.add_argument("fonte", help="arquivo de vídeo ou endereço (http/rtsp) de uma câmera")
    _opcoes_de_deteccao(video)

    imagem = comandos.add_parser("imagem", help="detecção em uma foto")
    imagem.add_argument("arquivo")
    imagem.add_argument("--modelo")
    imagem.add_argument("--confianca", type=float, default=0.40)
    imagem.add_argument("--tamanho", type=int, default=416)
    imagem.add_argument("--saida", help="onde gravar a foto com as caixas (padrão: saida/)")

    dataset = comandos.add_parser("dataset", help="baixa e monta o conjunto de treino")
    dataset.add_argument("--maximo-por-classe", type=int, default=400, help="imagens por classe (padrão: 400)")

    treinar = comandos.add_parser("treinar", help="treina o modelo")
    treinar.add_argument("--epocas", type=int, default=20)
    treinar.add_argument("--tamanho", type=int, default=416)
    treinar.add_argument("--congelar", type=int, default=10, help="camadas iniciais que não são reajustadas (0 = treina tudo)")
    treinar.add_argument("--dispositivo", default="cpu", help="cpu (padrão) ou 0 para treinar na placa de vídeo")
    treinar.add_argument("--lote", type=int, default=16)
    treinar.add_argument("--base", default="yolo11n.pt", help="modelo pré-treinado de partida")
    treinar.add_argument("--nome", default="passo_livre", help="nome da pasta do treino")
    treinar.add_argument("--continuar", action="store_true", help="retoma um treino interrompido")
    treinar.add_argument("--fracao", type=float, default=1.0, help="fração das imagens de treino usada (teste rápido)")

    avaliar = comandos.add_parser("avaliar", help="mede o modelo no conjunto de teste")
    avaliar.add_argument("--modelo")
    avaliar.add_argument("--tamanho", type=int, default=416)

    args = parser.parse_args(argv)

    if args.comando == "dataset":
        from passo_livre_visao.montar_dataset import montar

        montar(maximo_por_classe=args.maximo_por_classe)
        return 0
    if args.comando == "treinar":
        from passo_livre_visao.treino import treinar as executar_treino

        executar_treino(epocas=args.epocas, tamanho=args.tamanho, lote=args.lote, base=args.base, nome=args.nome,
                        continuar=args.continuar, fracao=args.fracao, congelar=args.congelar, dispositivo=args.dispositivo)
        return 0
    if args.comando == "avaliar":
        from passo_livre_visao.avaliacao import avaliar as executar_avaliacao

        executar_avaliacao(modelo=args.modelo, tamanho=args.tamanho)
        return 0
    if args.comando == "imagem":
        from passo_livre_visao.tempo_real import detectar_em_imagem

        return detectar_em_imagem(args.arquivo, modelo=args.modelo, confianca=args.confianca, tamanho=args.tamanho, saida=args.saida)

    from passo_livre_visao.tempo_real import executar

    fonte = args.indice if args.comando == "camera" else args.fonte
    posicao = (args.lat, args.lon) if args.lat is not None and args.lon is not None else None
    if args.enviar_para and posicao is None:
        parser.error("--enviar-para exige --lat e --lon: a câmera não sabe onde está")
    return executar(fonte, modelo=args.modelo, confianca=args.confianca, tamanho=args.tamanho, janela=not args.sem_janela,
                    gravar=args.gravar, limite=args.limite, enviar_para=args.enviar_para, posicao=posicao)


if __name__ == "__main__":
    sys.exit(main())
