import layoutparser as lp
import cv2
import numpy as np
import pdf2image
import os
import streamlit as st
from io import StringIO
import tempfile
import extra_streamlit_components as stx
from MemSum.src.summarizer import MemSum
from tqdm import tqdm
from rouge_score import rouge_scorer
import json
import numpy as np
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
        r"/usr/bin/tesseract"
    )
POPPLER_PATH = r"/usr/share/poppler"
# https://stackoverflow.com/questions/64048828/pytesseract-gives-an-error-permissionerror-winerror-5-access-is-denied
def pdf_to_img(pdf_file):
    return pdf2image.convert_from_path(pdf_file,
        grayscale=True,thread_count=os.cpu_count())

def ocr_core(file,lang,config):
    if lang:
        text = pytesseract.image_to_string(file, lang=lang,config=config)
    else:
        text = pytesseract.image_to_string(file,config=config)
    return text

def print_pages(pdf_file,lang, config):
    txt = ""
    images = pdf_to_img(pdf_file)
    for pg, img in enumerate(images):
        txt += ocr_core(img,lang, config)
    return txt

def read_and_print_pdf_with_ocr(pdf_file_path: str,lang:str=None) -> None:
    custom_oem_psm_config = r"-c tessedit_do_invert=0"
    return print_pages(pdf_file_path,lang, custom_oem_psm_config)

tab = stx.tab_bar(data=[
    stx.TabBarItemData(id="titre", title="✄ Détection des titres", description="Détecter les contours des titres"),
    stx.TabBarItemData(id="sommaire", title="✍ Extraction d'un sommaire", description="Extraire tous les titres"),
    stx.TabBarItemData(id="sommaire_MemSum", title="🖨️ Extraction d'un sommaire (MemSum)", description="Extraire tous les titres")])

FILENAME_MODEL="model_batch_5600.pt"
PATH_TO_MODEL="."
FILENAME_VOCAB="vocabulary_200dim.pkl"
PATH_TO_VOCAB="."
TESSERACT_LANGUAGE="fra"
memsum_custom_data = MemSum( os.path.join(PATH_TO_MODEL,FILENAME_MODEL) , 
                  os.path.join(PATH_TO_VOCAB,FILENAME_VOCAB), 
                  gpu = 0 ,  max_doc_len = 500  )

def extract_summary(document_of_list_shape):
    list_of_results=[]
    extracted_summary = memsum_custom_data.extract( [ document_of_list_shape ], 
                                       p_stop_thres = 0.6, 
                                       max_extracted_sentences_per_document = 50
                                      )[0]
    for element in document_of_list_shape:
        if element in extracted_summary:
            list_of_results.append(element)
    return list_of_results

if tab=="titre":
    SHOW_IMAGE=True
    PRINT_DETECTED_TITLE=False
else:
    SHOW_IMAGE=False
    PRINT_DETECTED_TITLE=True

if tab!="sommaire_MemSum":
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
else:
    uploaded_file = st.file_uploader("Choisir un PDF")
    col1, col2 = st.columns(2)
    if uploaded_file is not None:
        file_name="tempfilename"
        temp_local_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_local_dir, file_name)
        pdf=uploaded_file.getvalue()
        with open(file_path, "wb") as binary_file:
            binary_file.write(pdf)
            
        all_text = read_and_print_pdf_with_ocr(file_path,lang=TESSERACT_LANGUAGE).split("\n")
        with col1:
            text="\n".join(all_text)
            st.write(f"{text}")
        with col2:
            text="\n\n".join(extract_summary(all_text))
            st.write(f"{text}")

        if os.path.exists(file_path):
            os.remove(file_path) # Delete file