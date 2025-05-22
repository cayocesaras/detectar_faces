import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from keras_facenet import FaceNet

# Inicializa detector e modelo de reconhecimento
detector = MTCNN()
embedder = FaceNet()

# Carrega rostos conhecidos da pasta
def load_known_faces(path='conhecidos'):
    known_embeddings = {}
    for file in os.listdir(path):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(file)[0]
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            faces = detector.detect_faces(img_rgb)
            if not faces:
                st.warning(f"Nenhum rosto detectado em {file}.")
                continue

            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)
            face_crop = img_rgb[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            embedding = embedder.embeddings([face_crop])[0]
            known_embeddings[name] = embedding
    return known_embeddings

# Reconhecimento facial sempre mostrando melhor resultado com score
def recognize_faces(image_bgr, known_embeddings):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    st.write(f"🔍 Rostos detectados: {len(faces)}")

    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        face_crop = image_rgb[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue

        emb = embedder.embeddings([face_crop])[0]
        name = "Desconhecido"
        highest_score = -1

        for known_name, known_emb in known_embeddings.items():
            score = cosine_similarity([emb], [known_emb])[0][0]
            st.write(f"→ Similaridade com **{known_name}**: {score:.2f}")
            if score > highest_score:
                name = known_name
                highest_score = score

        # Converte similaridade para percentual
        percent = int(highest_score * 100)

        # Desenha retângulo ao redor do rosto
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Texto com nome e percentual
        text = f"{name} - {percent}%"

        # Medidas do texto para o fundo
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Posição do retângulo do texto abaixo da caixa (ajustando se ultrapassar a imagem)
        text_bottom_y = y + h + text_height + 10
        if text_bottom_y > image_bgr.shape[0]:
            # Se passar do limite da imagem, coloca o texto dentro da caixa, no canto inferior
            text_bottom_y = y + h - 5
            text_origin = (x + 5, text_bottom_y)
            rect_top_left = (x, y + h - text_height - 10)
            rect_bottom_right = (x + text_width + 10, y + h)
        else:
            # Texto abaixo da caixa
            text_origin = (x + 5, y + h + text_height + 5)
            rect_top_left = (x, y + h)
            rect_bottom_right = (x + text_width + 10, y + h + text_height + 10)

        # Fundo verde do texto
        cv2.rectangle(image_bgr, rect_top_left, rect_bottom_right, (0, 255, 0), -1)

        # Texto do nome em preto
        cv2.putText(image_bgr, text, text_origin,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return image_bgr

# Interface Streamlit
def main():
    st.title("🧠 Reconhecimento Facial com Streamlit")
    st.markdown("Faça upload de uma imagem contendo **rostos** para teste.")

    uploaded_file = st.file_uploader("Upload de imagem", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        st.image(image, caption="Imagem enviada", use_column_width=True)

        st.info("🧬 Carregando rostos conhecidos...")
        known_embeddings = load_known_faces()

        if known_embeddings:
            st.success(f"{len(known_embeddings)} rostos conhecidos carregados com sucesso.")
            result = recognize_faces(image_bgr, known_embeddings)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Resultado", use_column_width=True)
        else:
            st.warning("⚠️ Nenhum rosto conhecido foi carregado!")

if __name__ == "__main__":
    main()