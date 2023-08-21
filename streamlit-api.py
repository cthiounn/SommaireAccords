import layoutparser as lp
import cv2
import numpy as np
import pdf2image
import os
import streamlit as st
from io import StringIO
import tempfile
import extra_streamlit_components as stx

tab = stx.tab_bar(data=[
    stx.TabBarItemData(id="titre", title="✄ Détection des titres", description="Détecter les contours des titres"),
    stx.TabBarItemData(id="sommaire", title="✍ Extraction d'un sommaire", description="Extraire tous les titres")])


TESSERACT_LANGUAGE="fra"
if tab=="titre":
    SHOW_IMAGE=True
    PRINT_DETECTED_TITLE=False
else:
    SHOW_IMAGE=False
    PRINT_DETECTED_TITLE=True
    
def pdf_to_img(pdf_file):
    return pdf2image.convert_from_path(
        pdf_file, grayscale=False, thread_count=os.cpu_count()
    )

modele = st.radio(
    "Modèle d'inférence de la Dares",
    ('Modèle 1 : 7000', 'Modèle 2 : 900'),horizontal=True)
base_threshold = st.slider('Seuil de tolérance pour le modèle de base', 0, 100, 50) / 100
modele_threshold = st.slider('Seuil de tolérance pour le modèle Dares', 0, 100, 50) / 100
    
model = lp.models.Detectron2LayoutModel(
    config_path ="config.yaml",
    model_path ="model_final.pth",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", base_threshold],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
)

model1 = lp.models.Detectron2LayoutModel(
    config_path ="config_dares.yaml",
    model_path ="model_final_dares_1.pth",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", modele_threshold],
    label_map={3: "Text", 4: "Title", 1: "List", 2: "Table", 0: "Figure"},
)

model2 = lp.models.Detectron2LayoutModel(
    config_path ="config_dares.yaml",
    model_path ="model_final_dares_2.pth",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", modele_threshold],
    label_map={3: "Text", 4: "Title", 1: "List", 2: "Table", 0: "Figure"},
)
uploaded_file = st.file_uploader("Choisir un PDF")
col1, col2 = st.columns(2)
if uploaded_file is not None:
    file_name="tempfilename"
    temp_local_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_local_dir, file_name)
    pdf=uploaded_file.getvalue()
    with open(file_path, "wb") as binary_file:
        binary_file.write(pdf)
        
    all_images = pdf_to_img(file_path)
    list_of_titles=[]
    with col1:
        st.write("Modèle de base")

    with col2:
        st.write(f"{modele}")
    for one_image in all_images:
        one_image_np = np.asarray(one_image)
        layout = model.detect(one_image_np)
        layout1 = model1.detect(one_image_np)
        layout2 = model2.detect(one_image_np)
        if SHOW_IMAGE:
            with col1:
                st.image(lp.draw_box(one_image_np, layout, box_width=3, show_element_type=True), caption="Titres détectés", use_column_width=True)
            with col2:
                if modele=='Modèle 1 : 7000':
                    st.image(lp.draw_box(one_image_np, layout1, box_width=3, show_element_type=True), caption="Titres détectés", use_column_width=True)
                else:
                    st.image(lp.draw_box(one_image_np, layout2, box_width=3, show_element_type=True), caption="Titres détectés", use_column_width=True)
        title_blocks = lp.Layout([b for b in layout if b.type == "Title"])
        title_blocks1 = lp.Layout([b for b in layout1 if b.type == "Title"])
        title_blocks2 = lp.Layout([b for b in layout2 if b.type == "Title"])
        ocr_agent = lp.TesseractAgent(languages=TESSERACT_LANGUAGE)
        title_blocks_dares= title_blocks1 if modele=='Modèle 1 : 7000' else title_blocks2
        with col1:
            for block in title_blocks.sort(key=lambda x: x.coordinates[1]):
                segment_image = block.pad(left=5, right=5, top=5, bottom=5).crop_image(one_image_np)
                # add padding in each image segment can help
                # improve robustness
                text = ocr_agent.detect(segment_image)
                if PRINT_DETECTED_TITLE:
                    st.write(text)
                block.set(text=text, inplace=True)
        with col2:
            for block in title_blocks_dares.sort(key=lambda x: x.coordinates[1]):
                segment_image = block.pad(left=5, right=5, top=5, bottom=5).crop_image(one_image_np)
                # add padding in each image segment can help
                # improve robustness
                text = ocr_agent.detect(segment_image)
                if PRINT_DETECTED_TITLE:
                    st.write(text)
                block.set(text=text, inplace=True)

    if os.path.exists(file_path):
        os.remove(file_path) # Delete file