import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as bs
import requests
import cachetools as ct
import PyPDF2 as pdf
import re
import csv
import os
from transformers import pipeline

from scrapping import *

cacheIntro = ct.TTLCache(maxsize=float('inf'), ttl=86400)  
cache = ct.TTLCache(maxsize=float('inf'), ttl=86400)
def introducao():
    if 'dados' in cacheIntro:
        st.write(cacheIntro['dados'])
    else:
        st.title("AI JobSearch - Programa de auxílio na procura de vagas de emprego")

        # Problema
        st.header("Cansado de perder seu tempo procurando vagas de emprego?")
        st.write("Use a inteligência artificial para encontrar vagas de emprego de acordo com suas habilidades e interesses de forma eficiente.")

        st.image("./images/bg.jpg")

        st.write('''Como usar a ferramenta:
            \n1. Insira seu currículo na aba 'Upload para análise individual';
            \n2. Selecione o cargo e a região na aba 'Scrapping';
            \n3. Clique no botão 'Executar o scraping';
            \n4. Clique no botão 'Download' para salvar os dados;
            \n5. Selecione uma coluna e uma opção na aba 'Analise dinâmica';
            \n6. Clique no botão 'Executar a analise dinâmica';
            \n7. Clique no botão 'Download' para salvar os dados;''')


# -----

def upload():
    st.header("Upload para análise individual")
    caminhoMeu = st.file_uploader("Selecione seu currículo Linkedin em pdf", type="pdf")



    if caminhoMeu is not None:
        # salva o pdf  na pasta arquivosGerados
        with open(f"./arquivosGerados/Profile.pdf", "wb") as f:
            f.write(caminhoMeu.getbuffer())

        caminhoMeu = f"./arquivosGerados/Profile.pdf"
        st.write(f"Arquivo carregado!")

        def read_pdf(caminho):
            str_texto = ""
            texto=[]
            with open(caminho, "rb") as arquivo_pdf:
                reader = pdf.PdfReader(arquivo_pdf)

                for page in reader.pages:
                    page.extract_text()
                    texto.append(page.extract_text())
                
                for i in texto:
                    str_texto = str_texto+i
                
                str_texto = str_texto.replace("\n", " ")

                return str_texto


        def acha_colunas(text):
            # Regex
            padrao_email = r"\S+@\S+"
            padrao_numero = r"\(\d{2}\)\s\d{5}-\d{4}"
            padrao_link = r"www\.linkedin\.com/in/[\w-]+"

            emails = re.findall(padrao_email, text)
            numero = re.findall(padrao_numero, text)
            link = re.findall(padrao_link, text)

            for i in range(len(emails)):
                if re.search(r"com", emails[i]):
                    pass
                else:
                    emails[i] = emails[i]+"com"

            # Salvar em CSV
            nome_arquivo = './/arquivosGerados//1_saidaPDF.csv'
            if not os.path.exists(nome_arquivo):
                with open(nome_arquivo, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Email", "Numero", "Link"])  # Cabeçalho

            with open(nome_arquivo, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for i in range(len(emails)):
                    writer.writerow([emails[i], numero[i], link[i]])

            






            
            dict = {"Email": emails, "Numero": numero, "Link": link, "Texto Completo": read_pdf(caminhoMeu)}
            

            print(f"Dados salvos em {nome_arquivo}")
            return dict




        st.write(f"\nInformacoes do PDF:")
        st.write(acha_colunas(read_pdf(caminhoMeu)))
    


def LLM(texto):
    context = texto

    model_name = 'pierreguillou/bert-base-cased-squad-v1.1-portuguese'
    nlp = pipeline("question-answering", model=model_name)

    question = "A vaga possui qualquer benefício?"

    result = nlp(question=question, context=context)

    print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")

    return result


def scrapping(): 
    st.header("Scrapping")

    st.write("Selecione o cargo e a região:")

    # digite o cargo e a região
    cargo = st.text_input('Cargo')

    regiao = st.text_input('Região')

    num_pags = st.number_input('Quantidade de paginas', min_value=1, max_value=10)

    # button para executar o scraping
    if st.button('Executar o scraping'):
        num_pags = int(num_pags)
        if cargo == '':
            cargo = 'Analista de Dados'
        if regiao == '':
            regiao = 'São Paulo'
        executar_scrapping(cargo, regiao, num_pags)

    df = pd.read_json('./arquivosGerados/8_vagasIndeed.json')
    st.dataframe(df)

# Usa um sidebar para fazer a paginação
page = st.sidebar.selectbox('Selecione uma opção', ['Introdução', 'Upload para análise individual', 'Scrapping'])

if page == 'Introdução':
    introducao()
elif page == 'Upload para análise individual':
    upload()
elif page == 'Scrapping':
    scrapping()



