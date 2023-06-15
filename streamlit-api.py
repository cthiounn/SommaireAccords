import layoutparser as lp
import cv2
import numpy as np
import pdf2image
import os
import streamlit as st
import pandas as pd
from io import StringIO



PATH_OF_THE_PDF="./"
NAME_OF_THE_PDF_TO_READ="Accord-denteprise-ITG-DD2_0.pdf"
TESSERACT_LANGUAGE="fra"
SHOW_IMAGE=True
PRINT_DETECTED_TITLE=False

def pdf_to_img(pdf_file):
    return pdf2image.convert_from_path(
        pdf_file, grayscale=False, thread_count=os.cpu_count()
    )
    
model = lp.models.Detectron2LayoutModel(
    "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
)
    


uploaded_file = st.file_uploader("Choisir un PDF")
if uploaded_file is not None:
    all_images = pdf_to_img(uploaded_file)
    list_of_titles=[]
    for one_image in all_images:
        one_image_np = np.asarray(one_image)
        layout = model.detect(one_image_np)
        if SHOW_IMAGE:
            st.image(lp.draw_box(one_image_np, layout, box_width=3, show_element_type=True), caption="Titres détectés", use_column_width=True)
        title_blocks = lp.Layout([b for b in layout if b.type == "Title"])
        ocr_agent = lp.TesseractAgent(languages=TESSERACT_LANGUAGE)
        for block in title_blocks:
            segment_image = block.pad(left=5, right=5, top=5, bottom=5).crop_image(one_image_np)
            # add padding in each image segment can help
            # improve robustness
        
            text = ocr_agent.detect(segment_image)
            block.set(text=text, inplace=True)

        for txt in title_blocks.get_texts():
            list_of_titles.append(txt)
            if PRINT_DETECTED_TITLE:
                print(txt, end="\n---\n")