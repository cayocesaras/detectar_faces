# Reconhecimento Facial com Streamlit

Este projeto é uma aplicação web simples para **detecção e reconhecimento facial** utilizando Python, Streamlit, MTCNN e FaceNet.

Você pode fazer upload de uma imagem contendo rostos, e o sistema detecta todas as faces presentes, reconhecendo os rostos conhecidos previamente cadastrados. O reconhecimento exibe na caixa de detecção o nome da pessoa junto com a porcentagem de similaridade (mesmo valores baixos são exibidos).

## Funcionalidades

- Detecção de múltiplos rostos em imagens usando MTCNN.
- Extração de embeddings faciais com FaceNet.
- Reconhecimento comparando os embeddings das faces detectadas com os embeddings de rostos conhecidos.
- Exibição da caixa com o nome do rosto mais parecido e percentual de similaridade.
- Interface web simples e intuitiva com Streamlit.
- Upload de imagens para testes diretamente pela interface.

## Como usar

### 1. Instalação das dependências

` pip install streamlit opencv-python-headless mtcnn keras-facenet scikit-learn pillow numpy `

### 2. Preparação da pasta de rostos conhecidos

1. Crie uma pasta chamada conhecidos/ na raiz do projeto.
2. Insira fotos (jpg, jpeg ou png) com rostos conhecidos.
3. O nome do arquivo será usado como o nome da pessoa (exemplo: maria.jpg → nome: "maria").
4. Certifique-se que cada imagem contenha um rosto detectável.

### 3. Rodar a aplicação

` streamlit run app.py `

### 4. Uso

- Acesse a URL local fornecida pelo Streamlit.
- Faça upload de uma imagem contendo um ou mais rostos.
- A aplicação mostrará as caixas de reconhecimento com nomes e porcentagens.

## Estrutura do projeto: 
```
├── app.py                  # Código principal da aplicação
├── conhecidos/             # Fotos dos rostos conhecidos
│   ├── pessoa1.jpg
│   ├── pessoa2.png
│   └── ...
├── README.md               # Este arquivo
└── requirements.txt        # (Opcional) arquivo com dependências
```

## Tecnologias Utilizadas

- [Streamlit](https://streamlit.io/) — para interface web interativa e rápida.
- [MTCNN](https://mtcnn.readthedocs.io/en/latest/) — para detecção de rostos.
- [Keras FaceNet](https://github.com/nyoki-mtl/keras-facenet) — para extração de embeddings faciais.
- [scikit-learn](https://scikit-learn.org/stable/) — para cálculo de similaridade de cosseno.
- [OpenCV](https://opencv.org/) — para manipulação e desenho nas imagens.
- [Pillow](https://python-pillow.org/) — para manipulação de imagens no Streamlit.

## Como funciona o reconhecimento facial?

1. O sistema detecta todos os rostos na imagem de entrada com MTCNN.
2. Para cada rosto detectado, extrai um vetor de características (embedding) com FaceNet.
3. Compara o embedding extraído com os embeddings dos rostos conhecidos carregados da pasta conhecidos/.
4. Calcula a similaridade de cosseno entre os embeddings.
5. Exibe o nome do rosto conhecido mais parecido e a porcentagem de similaridade na caixa ao redor da face detectada.
6. Caso não haja rosto conhecido suficientemente parecido, exibe "Desconhecido" com a porcentagem obtida.

## Contato

Dúvidas, sugestões ou contribuições são bem-vindas!
Entre em contato: cayo.cesar.as@gmail.com
