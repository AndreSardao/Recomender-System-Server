#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import sympy as sp
import re
import mysql.connector
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from collections import Counter

import json
import requests
import copy
import random

from flask import Flask
from flask import request
import pymysql.cursors

global connection_pool
global module_primary

import os
os.environ["OPENAI_API_KEY"] = "open_AI_Key"

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

import matplotlib.pyplot as plt
from io import BytesIO


# In[2]:


# puxa tabelas do banco SGP: dasafios, infos de uso 
def pega_perguntas(tabela):
    config = {
    'user': 'root',          
    'password': '??????',        
    'host': 'localhost',            
    'database': '????' 
    }

    conn = mysql.connector.connect(**config)

    query = "SELECT * FROM {}".format(tabela)
    df = pd.read_sql_query(query, conn)

    #print(df.tail(50))

    conn.close()
    
    return df

# puxa tabelas do banco básico
def pega_data(tabela):
    config = {
    'user': 'root',          
    'password': '????',        
    'host': 'localhost',            
    'database': '????' 
    }

    conn = mysql.connector.connect(**config)

    query = "SELECT * FROM {}".format(tabela)
    df = pd.read_sql_query(query, conn)

    # Fechar a conexão com o banco de dados
    conn.close()
    
    return df 

# Só a parte q interessa
def filtra_para_interesses(df, ano, materia):
    df = df.loc[df['year_id'] == ano]
    df = df.loc[df['subject_id'] == materia]
    df = df.loc[df['active'] == 1]
    return df

# Queremos questões sem imagem
def devolve_id_com_imagem(df,df_im_qt):
    id_qt = df['id'].values
    
    # Quais são as questões que têm imagem? OBS: Queremos as que não tem
    questoes_imagem = list(df_im_qt['question_id'].values)
    
    # Filtrando os id das questoes de matematica do quinto ano que não tem imagem
    id_sem_imagem = []
    for q in id_qt:
        if q not in questoes_imagem:
            id_sem_imagem.append(q)
    
    return id_sem_imagem

# tive que dividir essa função em duas, para usar com o DF da alternativas
def devolve_qtoes_sem_imagem(df,id_sem_imagem):
    
    # As questões que nos interessam...
    df_de_interess = df[df["id"].isin(id_sem_imagem)]
    
    return df_de_interess

def devolve_qtoes_sem_imagem_alt(df,id_sem_imagem):
    
    # As questões que nos interessam...
    df_de_interess = df[df["question_id"].isin(id_sem_imagem)]
    
    return df_de_interess

# some com linha
def dropando_row(df, idx):
    df = df.drop(df[df.id == idx].index)
    return df

# some com linha
def dropando_row_alt(df, idx):
    df = df.drop(df[df.question_id == idx].index)
    return df

# limpa os html nas alternativas, enuciados etc..
def limpa_html(html_string):
    if type(html_string) == str:
        
        soup = BeautifulSoup(html_string, "html.parser")
        texto_limpo = soup.get_text(separator=" ", strip=True)

        return texto_limpo

# vira de LaTex em linguagem de gente
def decode_latex(texto):
    # Verificar se a string tem LaTeX (assumindo casos de LaTeX que tenham barras invertidas \)
    if isinstance(texto, str) and '\\' in texto:
        # Padrão (\frac{num}{den}) e detectar números mistos. Se aparecer outros casos, tenho que aumentar aqui
        latex_pattern = re.compile(r'(\d*)\\frac\{(\d+)\}\{(\d+)\}')
        
        def converter_parte_latex(match):
            inteiro = match.group(1)  
            numerador = match.group(2)  
            denominador = match.group(3) 
            
            # Se houver parte inteira, combinar com a fração
            if inteiro:
                return f'{inteiro} {numerador}/{denominador}'  
            else:
                return f'{numerador}/{denominador}'  
            
        texto_convertido = latex_pattern.sub(converter_parte_latex, texto)
        return texto_convertido
    return texto

# Constroi mapeamento entre duas lista, aqui é usado para relacionar duas colunas de um dataframe, notadamente o challenges_vs_skills
def dict_maker(list1, list2):
    dct = {}
    for k,v in zip(list1, list2):
        dct[k]=v
        
    return dct

# Em um dicionário onde tanto chave quanto valores sejam do tipo string, float ou int, esta função inverte keys e values. 
# Se algum elemento não for desses tipos, retorna uma mensagem de erro.
# Observação: Caso haja valores repetidos, a inversão resultará na perda de alguns dados.
def dict_inverter(dcto):
    
    tipos = ['str', 'int', 'float']
    
    # Verifica se todas as chaves e todos os valores são de tipos permitidos.
    if all(type(k).__name__ in tipos for k in dcto.keys()) and all(type(v).__name__ in tipos for v in dcto.values()):
        dct_invertido = {}
        for k, v in dcto.items():
            dct_invertido[v] = k
        return dct_invertido
    else:
        return "Esta função não funciona para este dicionário"

# Inverte dicionários que têm valores repetidos em chaves diferentes.
def double_value_inverter(dicto):
    lista_aux = copy.deepcopy(list(dicto.keys()))
    dictoo = {}
    for k in dicto.keys():
        if k in lista_aux:
            lista_aux.remove(k)
        for ke in lista_aux:
            if dicto[k] == dicto[ke]:
                dictoo[dicto[k]] = [k,ke]
                lista_aux.remove(ke)
        if dicto[k] not in dictoo.keys():
            dictoo[dicto[k]] = [k]
    return dictoo
    
# Primeira função a ser usada qdo recebemos o challenge data id do 
def get_student_and_chall(chall_dta_id):
    a = list(df_challenges[df_challenges['id'] == chall_dta_id]['data_student_id'].values)
    b = list(df_challenges[df_challenges['id'] == chall_dta_id]['challenge_id'].values)
    return a[0], b[0]

# Recebe um data_student e devolve o student id correspondente.
def devolve_student_id(dta_student):
    df_student_id = df_data_students.loc[df_data_students['id'] == dta_student]
    listenha = list(df_student_id["student_id"].values)
    return listenha[0]

# Tem um parâmetro apontando para matemática, tem que mudar no futuro
def from_student_get_challenges(stu_dta_id):
    df = df_challenges[df_challenges['data_student_id'] == stu_dta_id]
    df = df[df['subject_id'] == 2]
    lista_data_challenges = list(df['id'].values)
    lista_challenges = list(df['challenge_id'].values)
    lista_de_notas = list(df['performance'].values)
    return lista_data_challenges, lista_challenges, lista_de_notas 

# Retorna exatamente as questões 
def get_errors(data_chall_id):
    erradas = []
    df_aux = df_data_challenge_answers[df_data_challenge_answers['data_challenge_id']== data_chall_id]
    qts = list(df_aux['question_id'].values)
    crt = list(df_aux['correct'].values)
    for q,r in zip(qts,crt):
        if r == 0:
            erradas.append(q)
    return erradas

def get_errors_completo(data_chall_id):
    erradas = {}
    certas = {} 
    tempo_erradas = {}
    tempo_certas = {}
    
    df_aux = df_data_challenge_answers[df_data_challenge_answers['data_challenge_id']== data_chall_id]
    qts = list(df_aux['question_id'].values)
    crt = list(df_aux['correct'].values)
    aws = list(df_aux['answer'].values)
    tme = list(df_aux['time'].values)
    for q,r,a,t in zip(qts,crt,aws,tme):
        
        if r == 0:
            erradas[q] = a
            tempo_erradas[q] = t
        else:
            certas[q] = a
            tempo_certas[q] = t
        
    return erradas, certas, tempo_erradas, tempo_certas

def get_errors_completo_aws(data_chall_id):
    erradas = {}
    certas = {} 
    tempo_erradas = {}
    tempo_certas = {}
    
    df_aux = df_data_challenge_answers[df_data_challenge_answers['data_challenge_id']== data_chall_id]
    qts = list(df_aux['question_id'].values)
    crt = list(df_aux['correct'].values)
    aws = list(df_aux['answer'].values)
    tme = list(df_aux['time'].values)
    for q,r,a,t in zip(qts,crt,aws,tme):
        
        if r == 0:
            erradas[q] = a
            tempo_erradas[q] = t
        else:
            certas[q] = a
            tempo_certas[q] = t
        
    return erradas

#

"""
    Classifica quais erros são sistemáticos com base na frequência dos erros.
    
    Critérios sugeridos:
    - (1) Se o total de erros for 3: Considera sistemático se houver apenas um tipo de erro.
    
    - (2) Se o total de erros for 4 ou 5: Considera sistemático se algum tipo de erro representar, por exemplo, 60% ou mais dos erros.
    - (3) Se o total de erros for maior que 5: Considera sistemático se algum tipo de erro ocorrer em 40% ou mais do total.
    
    Parâmetros:error_counts (dict): Dicionário com a distribuição dos erros. Ex.: {'Erro de composição': 5, 'Erro Posicional': 2, ...}
                             
    Retorna:
        systematic_errors (list): Lista dos tipos de erro classificados como sistemáticos.
"""
def classifica_erros_sistematicos(error_counts):

    total = sum(error_counts.values())
    systematic_errors = []
    
    if total == 3:
        # Se o aluno cometeu exatamente 3 erros,
        # o único erro sistemático ocorre somente se houver apenas um tipo de erro.
        if len(error_counts) == 1:
            systematic_errors = list(error_counts.keys())
    elif total in [4, 5]:
        # Se o total for 4 ou 5, usamos um limiar de 60%
        for error, count in error_counts.items():
            if count / total >= 0.6:
                systematic_errors.append(error)
    else:
        # Para total de erros maior que 5, usamos um limiar de 40%
        for error, count in error_counts.items():
            if count / total >= 0.4:
                systematic_errors.append(error)
                
    return systematic_errors


# Gera relatórios a partir de um template.
def gena_relatorios_template(error_counts, erros_sist, vid, anim):
    """
    Retorna micro-relatórios (aluno, professor, pais) usando apenas f-strings.
    """
    # cálculo de distribuição percentual
    total = sum(error_counts.values()) or 1
    distribution = ", ".join(f"{err}: {count/total:.0%}" 
                             for err, count in error_counts.items())
    dominant = ", ".join(erros_sist) if erros_sist else "Nenhum"
    vid_str = "📺 Vídeos disponíveis." if vid else ""
    anim_str = "🎞️ Animações disponíveis." if anim else ""

    # Aluno
    student = (
        f"📊 Progresso no tópico\n"
        f"Distribuição de erros: {distribution}\n"
        f"Erro principal: {dominant}\n"
        f"{vid_str} {anim_str}".strip()
    )

    # Professor
    teacher = (
        f"👩‍🏫 Relatório Turma\n"
        f"Distribuição de erros: {distribution}\n"
        f"Erros sistemáticos: {dominant}\n"
        f"{vid_str} {anim_str}".strip()
    )

    # Pais
    parent = (
        f"📌 Evolução do aluno\n"
        f"Distribuição de erros: {distribution}\n"
        f"Principal dificuldade: {dominant}\n"
        f"{vid_str} {anim_str}".strip()
    )

    return {"student": student, "teacher": teacher, "parent": parent}

def plota_erros_pdf(error_counts) -> bytes:
    """
    Gera um histograma de erros e retorna o gráfico em PDF (bytes).
    
    Parâmetros:
        error_counts (dict): contagem absoluta de cada tipo de erro.
        
    Retorna:
        pdf_bytes (bytes): conteúdo do PDF gerado.
    """
    # Cria a figura
    plt.figure(figsize=(8, 6))
    bars = plt.bar(error_counts.keys(), error_counts.values(),
                   color='skyblue', edgecolor='black')

    # Título e rótulos
    plt.title('Histograma de Erros', fontsize=16, fontweight='bold')
    plt.xlabel('Tipos de Erro', fontsize=14)
    plt.ylabel('Frequência', fontsize=14)

    # Grade no eixo y
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Rótulos de valor
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 yval + 0.1, int(yval),
                 ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.show()

    # Salva em buffer PDF
    buf = BytesIO()
    plt.savefig(buf, format='pdf')
    plt.close()
    buf.seek(0)
    return buf.getvalue()


def gena_relatorios_template_grafico(error_counts: dict,
                              erros_sist: list,
                              vid: bool,
                              anim: bool):
    """
    Gera três micro-relatórios (aluno, professor, pais) e um PDF do gráfico de erros.

    Parâmetros:
        error_counts (dict): contagens absolutas de erros.
        erros_sist (list): lista de erros dominantes.
        vid (bool): há vídeos disponíveis?
        anim (bool): há animações disponíveis?

    Retorna:
        tuple:
          - reports (dict): {'student': str, 'teacher': str, 'parent': str}
          - chart_pdf (bytes): bytes do PDF gerado pelo plota_erros.
    """
    # --- passo 1: preparar texto ---
    total = sum(error_counts.values()) or 1
    distribution = ", ".join(
        f"{err}: {count/total:.0%}" for err, count in error_counts.items()
    )
    dominant = ", ".join(erros_sist) if erros_sist else "Nenhum"
    vid_str = "📺 Vídeos disponíveis." if vid else "Sem vídeos recomendados."
    anim_str = "🎞️ Animações disponíveis." if anim else "Sem animações recomendadas."

    # Micro-relatório para o aluno
    student = (
        f"📊 Progresso no tópico\n"
        f"Distribuição de erros: {distribution}\n"
        f"Erro principal: {dominant}\n"
        f"{vid_str} {anim_str}".strip()
    )

    # Micro-relatório para o professor
    teacher = (
        f"👩‍🏫 Relatório Turma\n"
        f"Distribuição de erros: {distribution}\n"
        f"Erros sistemáticos: {dominant}\n"
        f"{vid_str} {anim_str}".strip()
    )

    # Micro-relatório para os pais
    parent = (
        f"📌 Evolução do aluno\n"
        f"Distribuição de erros: {distribution}\n"
        f"Principal dificuldade: {dominant}\n"
        f"{vid_str} {anim_str}".strip()
    )

    reports = {
        "student": student,
        "teacher": teacher,
        "parent": parent
    }

    # --- passo 2: gerar o PDF do gráfico ---
    chart_pdf = plota_erros_pdf(error_counts)

    return reports, chart_pdf


'''
# Descobre se o aluno fez algum desafio de alguma habilidade ligada à habilidade mãe
def pesca_hab(lista_challs, dicto, hab_mae):
    for hb in 
''' 
def get_mysql_pool():
    global connection_pool
    connection_pool = pymysql.connect(host='localhost',
                                      database='explorando_dump',
                                      user='root',
                                      password='Bimbo020916@',
                                      cursorclass=pymysql.cursors.DictCursor)

# A função que faz a requisição
def puxa(id):
    event = get_module(id)
    return event

def get_module_primary(module_id):
    get_mysql_pool()
    global module_primary

    with connection_pool.cursor() as cursor:
        sql_m = "select `m`.id, `m`.year_id, `m`.active " \
                "from `modules` `m` " \
                "where `m`.`deleted_at` is null and `m`.id=%s"
        cursor.execute(sql_m, (module_id,))
        module_primary = cursor.fetchone()

def get_module(module_id):
    get_module_primary(module_id)

    module = {
        'id': module_primary['id'],
        'active': module_primary['active'],
        'year': {'id': module_primary['year_id']},
        'previous_modules': []
    }

    with connection_pool.cursor() as cursor:
        sql = "select `mp`.previous_module_id as id, `m`.year_id, `m`.active " \
              "from `module_previous_module` `mp` " \
              "join `modules` `m` on `m`.`id` = `mp`.`previous_module_id` " \
              "where `m`.`deleted_at` is null and `mp`.module_id=%s"
        cursor.execute(sql, (module_id,))
        results = cursor.fetchall()

    for result in results:
        module['previous_modules'].append({
            'id': result['id'],
            'active': result['active'],
            'year': {'id': result['year_id']}
        })

    return module

# Para ser usada no contexto dos módulos, pega as habilidade na coluna 'descriptions'
def destaca_bncc(texto):
    
    if ((texto[8] == ' ') or (texto[8] == '\t')):
        texto = texto[0:8]
    elif texto[8] != ' ':
        texto = texto[0:9]
        
    return texto

def destaca_bncc_ajeita(sok,dok,dex,df):    
    for s,d,i in zip(sok,dok,dex):
        if s != None:
            if s[8] == '_':
                df.loc[i,'description'] = s[0:11] 
            else:
                df.loc[i,'description'] = s[0:10]


# In[3]:


df_questions = pega_perguntas('questions')


# In[4]:


df_math_5_act = filtra_para_interesses(df_questions, 5, 2)


# In[5]:


df_math_4_act = filtra_para_interesses(df_questions, 4, 2)


# In[6]:


df_im_qt = pega_perguntas('image_question')


# In[7]:


# As questões que nos interessam...
aux_5 = devolve_id_com_imagem(df_math_5_act,df_im_qt)
aux_4 = devolve_id_com_imagem(df_math_4_act,df_im_qt)


# In[8]:


df_de_interess_5 = devolve_qtoes_sem_imagem(df_math_5_act,aux_5)
df_de_interess_4 = devolve_qtoes_sem_imagem(df_math_4_act,aux_4)


# In[9]:


# Puxando as alternativas
df_alternatives = pega_perguntas('alternatives')


# In[10]:


# DF só com as alternativas que nos interessam
df_de_interess_alternativas_5 = devolve_qtoes_sem_imagem_alt(df_alternatives,aux_5)
df_de_interess_alternativas_4 = devolve_qtoes_sem_imagem_alt(df_alternatives,aux_4)


# In[11]:


# Derrubando questões F ou V
df_de_interess_alternativas_5 = dropando_row_alt(df_de_interess_alternativas_5, 4427)
df_de_interess_5 = dropando_row(df_de_interess_5, 4427)


# In[12]:


df_de_interess_4[df_de_interess_4['id'] == 12266]['title'].values


# In[13]:


qst_id_4 = list(df_de_interess_alternativas_4['question_id'].values)
d = Counter(qst_id_4)


# In[14]:


alt_3 = []
alt_2 = []
for k in d.keys():
    if d[k] == 3:
        alt_3.append(k)
    if d[k] == 2:
        alt_2.append(k)   


# In[15]:


auxx = alt_2 + alt_3
for q_id in auxx:
    df_de_interess_alternativas_4 = df_de_interess_alternativas_4.drop(df_de_interess_alternativas_4[df_de_interess_alternativas_4.question_id == q_id].index)
    df_de_interess_4 = df_de_interess_4.drop(df_de_interess_4[df_de_interess_4.id == q_id].index)


# In[16]:


# Faxina geral de html e LaTex no quinto ano
df_de_interess_5['title'] = df_de_interess_5['title'].apply(limpa_html)
df_de_interess_5['support'] = df_de_interess_5['support'].apply(limpa_html)
df_de_interess_5['command'] = df_de_interess_5['command'].apply(limpa_html)
df_de_interess_5['title'] = df_de_interess_5['title'].apply(decode_latex)
df_de_interess_5['support'] = df_de_interess_5['support'].apply(decode_latex)
df_de_interess_5['command'] = df_de_interess_5['command'].apply(decode_latex)


# In[17]:


# Faxina geral de html e LaTex no quarto ano
df_de_interess_4['title'] = df_de_interess_4['title'].apply(limpa_html)
df_de_interess_4['support'] = df_de_interess_4['support'].apply(limpa_html)
df_de_interess_4['command'] = df_de_interess_4['command'].apply(limpa_html)
df_de_interess_4['title'] = df_de_interess_4['title'].apply(decode_latex)
df_de_interess_4['support'] = df_de_interess_4['support'].apply(decode_latex)
df_de_interess_4['command'] = df_de_interess_4['command'].apply(decode_latex)


# In[18]:


df_de_interess_alternativas_5['content'] = df_de_interess_alternativas_5['content'].apply(limpa_html)
df_de_interess_alternativas_5['content'] = df_de_interess_alternativas_5['content'].apply(decode_latex)


# In[19]:


df_de_interess_alternativas_4['content'] = df_de_interess_alternativas_4['content'].apply(limpa_html)
df_de_interess_alternativas_4['content'] = df_de_interess_alternativas_4['content'].apply(decode_latex)


# In[20]:


# list_maker 5 ano
aux_alt_id_5 = list(df_de_interess_alternativas_5['question_id'].values)
aux_alt_5 = list(df_de_interess_alternativas_5['content'].values)
aux_alt_correct_5 = list(df_de_interess_alternativas_5['correct'].values)
aux_alt_id_das_alt_5 = list(df_de_interess_alternativas_5['id'].values)

aux_interess_5_id = list(df_de_interess_5['id'].values)
aux_interess_5_title = list(df_de_interess_5['title'].values)
aux_interess_5_support = list(df_de_interess_5['support'].values)
aux_interess_5_command = list(df_de_interess_5['command'].values)
aux_interess_5_ano = list(df_de_interess_5['year_id'].values)
aux_interess_5_hab = list(df_de_interess_5['skill_id'].values)

aux_alt_id_teste_5 = list(df_de_interess_alternativas_5['id'].values)


# In[21]:


# list_maker 4 ano
aux_alt_id_4 = list(df_de_interess_alternativas_4['question_id'].values)
aux_alt_4 = list(df_de_interess_alternativas_4['content'].values)
aux_alt_correct_4 = list(df_de_interess_alternativas_4['correct'].values)
aux_alt_id_das_alt_4 = list(df_de_interess_alternativas_4['id'].values)


aux_interess_4_id = list(df_de_interess_4['id'].values)
aux_interess_4_title = list(df_de_interess_4['title'].values)
aux_interess_4_support = list(df_de_interess_4['support'].values)
aux_interess_4_command = list(df_de_interess_4['command'].values)
aux_interess_4_ano = list(df_de_interess_4['year_id'].values)
aux_interess_4_hab = list(df_de_interess_4['skill_id'].values)

aux_alt_id_teste_4 = list(df_de_interess_alternativas_4['id'].values)


# In[22]:


df_habs = pega_perguntas('skills')


# In[23]:


df_habs = df_habs.loc[df_habs['active'] == 1]


# In[24]:


aux_habs_ids = list(df_habs['id'].values)
aux_habs_bncc = list(df_habs['bncc_code'].values)


# In[25]:


aux_interess_5_hab.append(200)


# In[157]:


BNCC_5 = {}
for ide, hab in zip(aux_habs_ids,aux_habs_bncc):
    if ide in aux_interess_5_hab:
        BNCC_5[ide] = hab
#print(BNCC_5)
BNCC_5[193] = 'EF05MA02'


BNCC_5_inv = {}
for ide, hab in zip(aux_habs_ids,aux_habs_bncc):
    if ide in aux_interess_5_hab:
        BNCC_5_inv[hab] = ide
        
BNCC_5_inv['EF05MA02'] = 193
bncc_5 = list(BNCC_5.values())


# In[161]:


BNCC_4 = {}
for ide, hab in zip(aux_habs_ids,aux_habs_bncc):
    if ide in aux_interess_4_hab:
        BNCC_4[ide] = hab
        
BNCC_4[155] = 'EF04MA09'
#BNCC_4
bncc_4 = list(BNCC_4.values())
BNCC_4_inv = {}
for ide, hab in zip(aux_habs_ids,aux_habs_bncc):
    if ide in aux_interess_4_hab:
        BNCC_4_inv[hab] = ide
BNCC_4_inv['EF04MA09'] = 155
#BNCC_4_inv


# In[28]:


df_challs = pega_data('challenges')


# In[29]:


df_challs = df_challs.loc[df_challs['active'] == 1]


# In[30]:


challs_act = list(df_challs['id'].values)


# In[31]:


df_activity_chapters = pega_data('activity_chapters')
#df_activity_chapters.head()


# In[32]:


df_mod = pega_data('chapters')
#df_mod.head()


# In[33]:


df_mod_mat = df_mod[df_mod['subject_id'] == 2]


# In[34]:


df_mod_mat_5 = df_mod_mat[df_mod_mat['grade_id'] == 5]


# In[35]:


df_mod_mat_4 = df_mod_mat[df_mod_mat['grade_id'] == 4]


# In[36]:


aux_chall = list(df_challs['id'].values)
aux_acttiv = list(df_challs['activity_id'].values)

chall_2_activity = dict_maker(aux_chall, aux_acttiv)

aux_actt = list(df_activity_chapters['activity_id'].values)
aux_chap = list(df_activity_chapters['chapter_id'].values)

activity_2_chapter = dict_maker(aux_actt, aux_chap)

aux_chap_1 = list(df_mod['id'].values)
aux_hab = list(df_mod['skill_id'].values)

aux_chap_1_5 = list(df_mod_mat_5['id'].values)
aux_hab_mat_5 = list(df_mod_mat_5['skill_id'].values)

aux_chap_1_4 = list(df_mod_mat_4['id'].values)
aux_hab_mat_4 = list(df_mod_mat_4['skill_id'].values)

chapter_2_hab = dict_maker(aux_chap_1, aux_hab)

from_chall_2_skill = {}

for ch in aux_chall:
    if not np.isnan(chapter_2_hab[activity_2_chapter[chall_2_activity[ch]]]):
        #print(ch)
        #print(type(chapter_2_hab[activity_2_chapter[chall_2_activity[ch]]]))
        from_chall_2_skill[ch] = int(chapter_2_hab[activity_2_chapter[chall_2_activity[ch]]])


# In[37]:


mod_2_hab_5_ = dict_maker(aux_chap_1_5, aux_hab_mat_5)
mod_2_hab_4_ = dict_maker(aux_chap_1_4, aux_hab_mat_4)

hab_2_mod_5_ = dict_maker(aux_hab_mat_5, aux_chap_1_5)
hab_2_mod_4_ = dict_maker(aux_hab_mat_4, aux_chap_1_4)


# In[38]:


mod_2_hab_5 = {k: v for k, v in mod_2_hab_5_.items() if not np.isnan(v)}
mod_2_hab_4 = {k: v for k, v in mod_2_hab_4_.items() if not np.isnan(v)}

hab_2_mod_5_ = {k: v for k, v in hab_2_mod_5_.items() if not np.isnan(v)}
hab_2_mod_4_ = {k: v for k, v in hab_2_mod_4_.items() if not np.isnan(v)}


# In[39]:


def check_unique(lista):
    aux = list(set(lista))
    aux_remov = lista.copy()
    for x in aux:
        aux_remov.remove(x)
    return aux_remov


# In[40]:


oie = list(from_chall_2_skill.values())
opa = check_unique(oie)


# In[41]:


retem = []
for ide in opa:
    dfe = df_habs[df_habs['id']== ide]
    listy = list(dfe['bncc_code'].values)
    if len(listy) != 0:
        retem.append(listy[0])
#retem


# In[42]:


cimiterio = []
for k in from_chall_2_skill.keys():
    a = from_chall_2_skill[k]
    if a in opa:
        cimiterio.append(k)
#cimiterio


# In[43]:


#print("Antes: ", len(from_chall_2_skill))
for k in cimiterio:
    del from_chall_2_skill[k]   
#print("Depois: ", len(from_chall_2_skill))


# In[44]:


from_skill_2_chall = dict_maker(from_chall_2_skill.values(),from_chall_2_skill.keys())


# In[45]:


#from_skill_2_chall.keys()


# In[46]:


from_skill_2_chall[155] = 201068


# In[47]:


#from_skill_2_chall


# In[48]:


df_data_students = pega_data('data_students')


# In[49]:


df_challenges = pega_data('data_challenges')


# In[50]:


aux_data_challenge_id = list(df_challenges['id'].values)


# In[51]:


aux_challenge_id = list(df_challenges['challenge_id'].values)


# In[52]:


# Dicionário que traduz de data_challenges para challenges
from_data_2_challs = dict_maker(aux_data_challenge_id, aux_challenge_id)


# In[53]:


df_mod_5 = df_mod.loc[df_mod['grade_id'] == 5]
df_mod_math_5 = df_mod_5.loc[df_mod_5['subject_id'] == 2]
df_mod_math_5_act = df_mod_math_5.loc[df_mod_math_5['active'] == 1]

df_mod_4 = df_mod.loc[df_mod['grade_id'] == 4]
df_mod_math_4 = df_mod_4.loc[df_mod_4['subject_id'] == 2]
df_mod_math_4_act = df_mod_math_4.loc[df_mod_math_4['active'] == 1]


# In[54]:


# Cortando fora simulados
df_mod_math_5_act = df_mod_math_5_act.drop(df_mod_math_5_act[df_mod_math_5_act.id == 20729].index)
df_mod_math_5_act = df_mod_math_5_act.drop(df_mod_math_5_act[df_mod_math_5_act.id == 20747].index)
#sd = list(df_mod_math_5_act['specificity'].values)
ds = list(df_mod_math_5_act['description'].values)
idex = list(df_mod_math_5_act.index)

#sd_4 = list(df_mod_math_4_act['specificity'].values)
ds_4 = list(df_mod_math_4_act['description'].values)
idex_4 = list(df_mod_math_4_act.index)


# In[55]:


df_mod_math_5_act['description'] = df_mod_math_5_act['description'].apply(destaca_bncc)
#destaca_bncc_ajeita(sd,ds,idex,df_mod_math_5_act)


# In[56]:


df_mod_math_4_act['description'] = df_mod_math_4_act['description'].apply(destaca_bncc)
#destaca_bncc_ajeita(sd_4,ds_4,idex_4,df_mod_math_4_act)


# In[57]:


#df_mod_math_4_act.head(25)


# In[58]:


aux_mod_description_5 = list(df_mod_math_5_act['description'])
aux_mod_id_5 = list(df_mod_math_5_act['id'])

aux_mod_description_4 = list(df_mod_math_4_act['description'])
aux_mod_id_4 = list(df_mod_math_4_act['id'])


# In[59]:


aux_skill_id = list(df_habs['id'].values)
aux_bncc = list(df_habs['bncc_code'].values)
bncc_2_skill_id = dict_maker(aux_bncc, aux_skill_id)

aux_chap_1_5 = list(df_mod_math_5_act['id'].values)
aux_hab_5 = list(df_mod_math_5_act['skill_id'].values)
skill_2_mod_5 = dict_maker(aux_hab_5, aux_chap_1_5)
mod_2_skill_5 = dict_maker(aux_chap_1_5, aux_hab_5)

aux_chap_1_4 = list(df_mod_math_4_act['id'].values)
aux_hab_4 = list(df_mod_math_4_act['skill_id'].values)
skill_2_mod_4 = dict_maker(aux_hab_4, aux_chap_1_4)
mod_2_skill_4 = dict_maker(aux_chap_1_4, aux_hab_4)

#skill_2_mod_5 = dict_maker(aux_hab_5, aux_chap_1_5)
#skill_2_mod_4 = dict_maker(aux_hab_4, aux_chap_1_4)

skill_2_modchap = {**skill_2_mod_5, **skill_2_mod_4}


# In[60]:


hab_2_mod_5 = {}
for code in aux_bncc:    
    if bncc_2_skill_id[code] in aux_hab_5: 
        hab_2_mod_5[code] = skill_2_mod_5[bncc_2_skill_id[code]]
    

hab_2_mod_4 = {}
for code in aux_bncc:    
    if bncc_2_skill_id[code] in aux_hab_4: 
        hab_2_mod_4[code] = skill_2_mod_4[bncc_2_skill_id[code]]


# In[61]:


aux_mod_description_5 = list(df_mod_math_5_act['description'])
aux_mod_id_5 = list(df_mod_math_5_act['id'])

aux_mod_description_4 = list(df_mod_math_4_act['description'])
aux_mod_id_4 = list(df_mod_math_4_act['id'])


# In[62]:


df_data_challenge_answers = pega_data('data_challenge_answers')


# In[63]:


def from_chall_2_mod(chall):
    hab_mae_id = from_chall_2_skill[chall]
    hab_mae = BNCC_5[hab_mae_id]
    mod_mae = hab_2_mod_5[hab_mae]
    return mod_mae 

def mod_to_chall(mod):
    if mod in mod_2_hab_5.keys(): 
        hab = mod_2_hab_5[mod]
    print("HHAABB",hab)
    if hab in BNCC_5_inv: 
        skill = BNCC_5_inv[hab]
    print("SSKKIILLLL",skill)
    chall = from_skill_2_chall[skill]
    return chall

def from_chall_2_mod_4(chall):
    hab_mae_id = from_chall_2_skill[chall]
    hab_mae = BNCC_4[hab_mae_id]
    mod_mae = hab_2_mod_4[hab_mae]
    return mod_mae 

def mod_to_chall_4(mod):
    hab = mod_2_hab_4[mod]
    skill = BNCC_4_inv[hab]
    chall = from_skill_2_chall[skill]
    return chall



# In[64]:


df_data_students = pega_data('data_students')


# In[65]:


#df_data_students.head()


# In[66]:


df_videos = pega_data("videos")


# In[181]:


#df_videos


# In[68]:


# valores bizarros
dict_perdido = {851.0: 4,193.0: 3,664.0: 3,220.0: 3,
         867.0: 3,
         871.0: 3,
         865.0: 3,
         866.0: 3,
         624.0: 3,
         632.0: 3,
         638.0: 3,
         822.0: 3,
         825.0: 3,
         837.0: 3,
         957.0: 3,
         66.0: 2,
         242.0: 2,
         533.0: 2,
         245.0: 2,
         36.0: 2,
         492.0: 2,
         1991.0: 2,
         240.0: 2,
         75.0: 2,
         1243.0: 2,
         169.0: 2,
         173.0: 2,
         1352.0: 2,
         190.0: 2,
         356.0: 2,
         272.0: 2,
         689.0: 2,
         852.0: 2,
         973.0: 2,
         853.0: 2,
         974.0: 2,
         854.0: 2,
         857.0: 2,
         874.0: 2,
         978.0: 2,
         491.0: 2,
         637.0: 2,
         634.0: 2,
         219.0: 2,
         640.0: 2,
         631.0: 2,
         651.0: 2,
         652.0: 2,
         656.0: 2,
         659.0: 2,
         526.0: 2,
         927.0: 2,
         943.0: 2,
         946.0: 2,
         363.0: 2,
         384.0: 2,
         389.0: 2,
         406.0: 2,
         407.0: 2,
         409.0: 2,
         980.0: 2,
         841.0: 2,
         870.0: 2,
         855.0: 2,
         849.0: 2,
         972.0: 2,
         882.0: 2,
         433.0: 2,
         307.0: 2,
         312.0: 2,
         313.0: 2,
         316.0: 2,
         320.0: 2,
         394.0: 2,
         1220.0: 2,
         1410.0: 2,
         2000.0: 2}


# In[69]:


chapter_2_hab_II = dict(chapter_2_hab)
repetidos = list(dict_perdido.keys())
chapter_2_hab_III = {k: int(v) for k, v in chapter_2_hab_II.items() if (not np.isnan(v)) and (v not in repetidos)}


# In[70]:


#chap_2_activity = dict_maker(aux_chap, aux_actt)

activity_aux = list(df_videos['activity_id'].values)
vid_aux = list(df_videos['id'].values)
activity_2_vid = dict_maker(activity_aux, vid_aux)


# In[71]:


acttt_doest_matters = []
for ativ in aux_actt:
    if ativ not in activity_aux:
        acttt_doest_matters.append(ativ)


# In[72]:


for k in acttt_doest_matters:
    del activity_2_chapter[k]


# In[73]:


chap_2_acttt = double_value_inverter(activity_2_chapter)


# In[74]:


hab_2_chapter = {v:k for k,v in chapter_2_hab_III.items()}


# In[78]:


# if not np.isnan(activity_2_vid[chap_2_acttt[hab_2_chapter[hb]]]):
def skillid_2_videoid0(hb):  
    video_s = []
    if not np.isnan(chap_2_acttt[hab_2_chapter[hb]]):
            lista_aux = int(chap_2_acttt[hab_2_chapter[hb]])
            for x in lista_aux:
                if x in activity_aux:
                    video_s.append(activity_2_vid[x])
            if len(video_s) != 0:
                return video_s
            else: 
                return "Atualmente este capítulo não possui vídeos"
    
                  


# In[79]:


import numpy as np

def skillid_2_videoid(hb):
    # Pega o capítulo correspondente à skill
    chap = hab_2_chapter.get(hb)
    if chap is None:
        return "Habilidade não mapeada em hab_2_chapter"

    # Pega a "inversa forçada" (lista ou NaN)
    raw = chap_2_acttt.get(chap, np.nan)

    # se for escalar (float) e NaN -> não há vídeos
    if isinstance(raw, float) and np.isnan(raw):
        return "Atualmente este capítulo não possui vídeos"

    # Montra a lista de activities
    # se raw for lista/array, usá-la direta;
    # se raw for um único número, colocá-lo numa lista
    if isinstance(raw, (list, np.ndarray)):
        lista_aux = list(raw)
    else:
        # pode ser um float (não-NaN) ou int
        lista_aux = [int(raw)]

    # Mapeamento cada activity ao vídeo
    video_s = []
    for x in lista_aux:
        if x in activity_aux:             # activity_aux é a lista de activities válidas
            video_s.append(activity_2_vid[x])  # activity_2_vid mapeia activity -> video

    # dando o resultado
    if video_s:
        return video_s
    else:
        return "Atualmente este capítulo não possui vídeos"


# In[167]:


def back_bone(data_challenge):

    dta_chall = data_challenge
    
    # Pega o aluno e o desafio correspondente
    dta_student, chall = get_student_and_chall(dta_chall)
    
    std_id  = devolve_student_id(dta_student)
    
    
    # Filtra os data students relacionados ao desafio mãe
    #df_chall = df_challenges[(df_challenges['challenge_id'] == chall)]
    
    # Pega as ações ligadas ao aluno: todos os challenges e notas daquele aluno 
    todos_dta_challs, todos_challs, todas_notas = from_student_get_challenges(dta_student)
    

    # Pega o id da habilidade principal (mãe)
    hab_mae_id = from_chall_2_skill[chall]
    
    # Pega a bncc em si
    hab_mae = BNCC_5[hab_mae_id]
    
    # O nome da hbilidade mãe não é colocado no mesmo saco das outras habilidades
    csv_1 = hab_mae
    
    # O 'saco' onde colocamos o código BNCC das habilidades pré-requisito
    habs_para_csv = []
    
    # Peda o módulo em si
    mod_mae = hab_2_mod_5[hab_mae]
    
    aux = puxa(mod_mae)
    
    aux_1 = aux['previous_modules']
    
    prev_mods_4 = []
    
    prev_mods_5 = []
    
    if len(aux_1) != 0:
        for dct in aux_1:
            if ((dct['year']['id'] == 5) and (dct['active'] == 1)):
                prev_mods_5.append(dct['id'])
            if ((dct['year']['id'] == 4) and (dct['active'] == 1)):
                prev_mods_4.append(dct['id'])
                
    else: 
        
        return {"response": "Este módulo não tem pré-requisito."},'',''
        
    prev_habs_5 = []
    
    prev_habs_4 = []
    
    if len(prev_mods_5) != 0:
        for mod in prev_mods_5:
            if (mod in aux_mod_id_5):
                prev_habs_5.append(mod_2_hab_5[mod])
                habs_para_csv.append(BNCC_5[mod_2_hab_5[mod]])
            
    if len(prev_mods_4) != 0:
        for mod in prev_mods_4:
            if (mod in aux_mod_id_4):
                print("FLUUUU", mod)
                prev_habs_4.append(mod_2_hab_4[mod])
                habs_para_csv.append(BNCC_4[mod_2_hab_4[mod]])
                
    print('prev_habs_5: ',prev_habs_5)
    print('prev_habs_4: ',prev_habs_4)

    ids_5 = []
    
    ids_4 = []
    
    if len(prev_habs_5) != 0:
        for hab in prev_habs_5:
            print('hab:', hab)
            if hab in bncc_5:
                ids_5.append(BNCC_5_inv[hab])
            
    if len(prev_habs_4) != 0:
        for hab in prev_habs_4:
            print('habHABHABHAB:', hab)
            if hab in bncc_4:
                print('hab4:', hab)
                ids_4.append(BNCC_4_inv[hab]) 
                
    print('ids_5: ',ids_5)
    print('ids_4: ',ids_4)
            
    recommend = {}
    
    vds_5 = []
    vds_4 = []
    

    
    if hab_mae_id in aux_skill_id:
        vds_5.append(skillid_2_videoid(hab_mae_id))
    
    
    if len(ids_5) != 0:
        for ide in ids_5:
            #if ide in aux_skill_id:
            vds_5.append(skillid_2_videoid(ide)) 
    
    if len(ids_4) != 0:
        for ide in ids_4:
            #if ide in aux_skill_id:
            vds_4.append(skillid_2_videoid(ide)) 
            
    recommend["vídeos_quinto_ano"] = vds_5
    recommend["vídeos_quarto_ano"] = vds_4
    
    challs_5 = []
    
    challs_4 = []
    
    if len(ids_5) != 0:
        for hab in ids_5:
            challs_5.append(from_skill_2_chall[hab])
            
    if len(ids_4) != 0:
        for hab in ids_4:
            challs_4.append(from_skill_2_chall[hab]) 
    
    # Descobrindo todos os data_student com aquele student_id
    df_student_id1 = df_data_students.loc[df_data_students['student_id'] == std_id]
    listra = list(df_student_id1['id'].values)
    
    print("listra: ", listra)
    
    dta_stu_ids = []
    
    # Achando os data_students correspondentes aos students id's encontrados acima.
    if len(challs_5) != 0:
        print("UEPA5")
        for ch in challs_5:
            df_ch = df_challenges[(df_challenges['challenge_id'] == ch)]
            lst_aux = list(df_ch['data_student_id'].values)
            print("LISTOMUI5: ",lst_aux)
            for l in listra:
                if l in lst_aux:
                    if l != dta_student:
                        dta_stu_ids.append(l)
                
    if len(challs_4) != 0:
        for ch in challs_4:
            df_ch = df_challenges[(df_challenges['challenge_id'] == ch)]
            lst_aux = list(df_ch['data_student_id'].values)
            print("LISTOMUI4: ",lst_aux)
            for l in listra:
                print("L")
                if l in lst_aux:
                    print("UEPA4")
                    if l != dta_student:
                        dta_stu_ids.append(l)
                        
    print("dta_stu_ids: ", dta_stu_ids)
                    
    
    # Usando os data_student_id conseguido de student_id pega as ações ligadas ao aluno: todos os challenges e notas daquele aluno
    for ide in dta_stu_ids:
            todos_dta_challs_1, todos_challs_1, todas_notas_1 = from_student_get_challenges(ide)
            todos_dta_challs = list(set(todos_dta_challs + todos_dta_challs_1))
            todos_challs = list(set(todos_challs + todos_challs_1))
            todas_notas = list(set(todas_notas + todas_notas_1))
            
     
    dcto = dict_maker(todos_challs, todos_dta_challs)
    
    erros_hab_mae = {}
    
    erros_hab_mae = get_errors_completo_aws(dcto[chall]) 
    
    #print(erros_hab_mae)
    
    print('CHALLS_5: ',challs_5)
    print('CHALLS_4: ',challs_4)
    
    resp_erradas_5 = {}
    resp_erradas_4 = {}
    
    if len(challs_5) != 0:
        #print("foi1")
        for cha in challs_5:
            if cha in dcto.keys():
                #print("foi1")
                #print(dcto[cha])
                resp_erradas_5[cha] = get_errors_completo_aws(dcto[cha])                
                
    if len(challs_4) != 0:       
        for cha in challs_4:
            if cha in dcto.keys():
                #print("foi2: ",dcto[cha])
                resp_erradas_4[cha] = get_errors_completo_aws(dcto[cha])

    erros_hab_filha, acertos_hab_filha, tempo_resp_err, tempo_resp_acerto = get_errors_completo(dcto[chall])
    
    re_er_5 = []
    re_er_4 = []
    if len(resp_erradas_5) != 0: 
        for dcio in resp_erradas_5.values():
            re_er_5.append(list(dcio.values()))
            
    if len(resp_erradas_4) != 0: 
        for dcio in resp_erradas_4.values():
            re_er_4.append(list(dcio.values()))
        
    print("re_er_5: ",re_er_5)
    print("re_er_4: ",re_er_4)
    
    soh_as_resp_erradas_4 = []
    soh_as_resp_erradas_5 = []
    
    if len(re_er_5) != 0:
        for lst in re_er_5:
            soh_as_resp_erradas_5 = soh_as_resp_erradas_5 + lst
    print("RESPOS ERRADAS 5",soh_as_resp_erradas_5)
    
    if len(re_er_4) != 0:
        for lst in re_er_4:
            soh_as_resp_erradas_4 = soh_as_resp_erradas_4 + lst
            
    print('soh_as_resp_erradas_4: ', soh_as_resp_erradas_4)
            
    
    x = list(erros_hab_mae.values())
    y = soh_as_resp_erradas_5
    z = soh_as_resp_erradas_4
    
   
    
    todas_resps_erradas = x + y + z
    
    recommend["questoes"] = todas_resps_erradas
    
    # Considerando passas as habs pra csv em uma lista
    return recommend,csv_1,habs_para_csv    


# In[141]:


back_bone(513200)


# In[82]:


# Função que separa questões por habilidade
# Variável df_1: df_de_interess_4 ou df_de_interess_5
# Variável df_2: df_de_interess_alternativas_4 ou df_de_interess_alternativas_5
# VAriável lista_1: aux_interess_4_id ou aux_interess_5_id
# Variável lista_2: BNCC_4 ou BNCC_5

def sep_questoes(hab, ano, df_1, df_2, lista_1, lista_2):    
    questoes_hab = []
    for q_id in lista_1:

        questao = {}

        questao['question_id'] = q_id
        
        #df_de_interess = lista_1

        df = df_1[df_1['id'] == q_id]

        aux_hab = list(df['skill_id'].values)
        skill = lista_2[aux_hab[0]]
        
        if skill == hab:
        
            questao['hab_bncc'] = skill 

            aux_title = list(df['title'].values)
            
            aux_support = list(df['support'].values)
            
            aux_command = list(df['command'].values)
            
            if (aux_title[0] != None) and (aux_support[0] != None) and (aux_command[0] != None):
           
                questao['Enunciado'] = aux_title[0] + ' ' + aux_support[0] + ' ' + aux_command[0]
            
            if (aux_support[0] == None) and (aux_command[0] == None):
           
                questao['Enunciado'] = aux_title[0]
            
            if (aux_support[0] == None) and (aux_command[0] != None):
           
                questao['Enunciado'] = aux_title[0] + ' ' + aux_command[0]
            
            if (aux_support[0] != None) and (aux_command[0] == None):
           
                questao['Enunciado'] = aux_title[0] + ' ' + aux_support[0]

            #questao['ano'] = ano

            df_alt = df_2[df_2['question_id'] == q_id]
            

            alt_content = list(df_alt['content'].values)
            alt_id = list(df_alt['id'].values)
            
            questao['A'] = alt_content[0]
            questao['B'] = alt_content[1]
            questao['C'] = alt_content[2]
            questao['D'] = alt_content[3]
            
            questao['A_alt_id'] = alt_id[0]
            questao['B_alt_id'] = alt_id[1]
            questao['C_alt_id'] = alt_id[2]
            questao['D_alt_id'] = alt_id[3]

            
            alt_correct = list(df_alt['correct'].values)

            for i in range(4):

                    if alt_correct[0] == 1:

                        questao['resposta_certa'] = 'A'

                    elif alt_correct[1] == 1:

                        questao['resposta_certa'] = 'B'

                    elif alt_correct[2] == 1:

                        questao['resposta_certa'] = 'C'

                    elif alt_correct[3] == 1:

                        questao['resposta_certa'] = 'D'

            questoes_hab.append(questao)


    return questoes_hab


# In[168]:


# Função recebe o id do desafio em que o aluno apresentou resultados insatisfatório junto com.../
# /... as tabelas de erros sistemáticos da habilidade atual e de seu pré-requisito direto.../
# /... identifica quais são os erros sistemáticos cometidos pelo aluno.../
# /... devolve uma recomendação de questões a serem feitas.
# Próximo passo imediato: adicionar um LLM que produza um relatório em português, baseado em um template.../
# /... que explique ao aluno, ou professor, o estatus atual das lacunas de conhecimento do aluno

def master_func(data_challenge):
    erros_cods = []
    
    erros = []
   
    tabelas_nomes = {'EF05MA01':['classificacao_erros_EF05MA01.csv', 'classificacao_erros_EF04MA01.csv'],
                    'EF05MA03':['classificacao_erros_EF05MA03.csv', 'classificacao_erros_EF04MA09_1.csv'], 
                    'EF05MA03A ':['classificacao_erros_EF05MA03.csv', 'classificacao_erros_EF04MA09_1.csv'],
                    'EF05MA04':['classificacao_erros_EF05MA04.csv', 'classificacao_erros_EF04MA09_2.csv'],
                    'EF05MA05':['classificacao_erros_EF05MA05.csv', 'classificacao_erros_EF05MA04.csv'],
                    'EF05MA06':['classificacao_erros_EF05MA06.csv', 'classificacao_erros_EF04MA0910.csv'],
                    'EF05MA07':['classificacao_erros_EF05MA07.csv', 'classificacao_erros_EF04MA10_1.csv'],
                    'EF05MA08':['classificacao_erros_EF05MA08.csv', 'classificacao_erros_EF04MA060710_1.csv'],
                    'EF05MA09':['classificacao_erros_EF05MA09.csv', 'classificacao_erros_EF04MA08.csv'],
                    'EF05MA10':['classificacao_erros_EF05MA10.csv', 'classificacao_erros_EF04MA14.csv'],
                    'EF05MA22':['classificacao_erros_EF05MA22.csv', 'classificacao_erros_EF04MA26_2.csv'],
                    'EF05MA23':['classificacao_erros_EF05MA23.csv', 'classificacao_erros_EF04MA26.csv']
                    } 
    
    # a função back_bone Puxa todas as alternativas que o aluno errou ao responder questões dos desafios da habilidade atual e de seu.../
    # /... pré-requisito
    erros_1,nome_tab_mae,habs_pre = back_bone(data_challenge)
    
    print("ChiiiiiiCagoo", erros_1)
    print('HABS_PRE: ',habs_pre)
    
    # É um dicionário que contém as tabelas de erros
    csv_1 = nome_tab_mae
    
    if csv_1 not in tabelas_nomes.keys():
        return {"KeyError": "Este sistema de recomendações ainda não pode ser usado para este capítulo"}
    
    df_tabela_de_erros_hab_atual = pd.read_csv(tabelas_nomes[csv_1][0])
    
    '''
    for nm in habs_pre:
        print('DFDFDFDF')
        tabs_erros = tabelas_nomes[nm]
        DFs[tabs_erros] = pd.read_csv(tabs_erros)
    print("UBA: ", DFs)
    '''
    
    # Tabelas de erros das pré-req
    tabs_erros = pd.read_csv(tabelas_nomes[csv_1][1])
    
    print(df_tabela_de_erros_hab_atual.tail(20))
    print(tabs_erros.tail(20))
    
    for x in erros_1["questoes"]:
        x = int(x)
        erros.append(x)
    
    aux_list_atual = list(df_tabela_de_erros_hab_atual['id_da_alternativa'].values)
    aux_list_pre = list(tabs_erros['id_da_alternativa'].values)
    
    
    print("ChiiiiiiCagoo", erros_1)
    print("Manhatã", erros)
    
    for er in erros:
        if er in aux_list_atual:
            aux_1 = list(df_tabela_de_erros_hab_atual[df_tabela_de_erros_hab_atual['id_da_alternativa'] == er]['nome_erro'].values)
            print("aux_11111\n\n", len(aux_1), er)
            if len(aux_1) != 0:
                erros_cods.append(aux_1[0])
                
    for er in erros:
        if er in aux_list_pre:
            aux_2 = list(tabs_erros[tabs_erros['id_da_alternativa'] == er]['nome_erro'])
            print("aux_22222\n\n", len(aux_2), er)
            if len(aux_2) != 0:
                for e in aux_2:
                    erros_cods.append(e)
    
    
    erros_cont = Counter(erros_cods)
    
    print("Casa dos Contos", erros_cont)
    
    # Função que faz o teste de frequência
    erros_sist = classifica_erros_sistematicos(erros_cont)
    
    #print("Erros Sistemáticos: ",erros_sist)
    
    recommendations_1 = []
    recommendations_2 = []
    
    for err in erros_sist:
    
        df_reco_1 = df_tabela_de_erros_hab_atual[df_tabela_de_erros_hab_atual['nome_erro'] == err]
        recommendations_1 = recommendations_1 + list(set(list(df_reco_1['id_da_questao'].values)))
        
    for err in erros_sist:
        
        df_reco_2 = tabs_erros[tabs_erros['nome_erro'] == err]
        recommendations_2 = recommendations_2 + list(set(list(df_reco_2['id_da_questao'].values)))
                
    recommendations = list(recommendations_1) + list(recommendations_2)               
    print("RECOCO RÓCOCÓ",recommendations)
    erros_1["questoes"] = recommendations
    
    vid = False
    anim = False
    
    if (len(erros_1['vídeos_quinto_ano']) != 0) or (len(erros_1['vídeos_quarto_ano']) != 0):
        vid = True
    
        
    reports, pdf_bytes = gena_relatorios_template_grafico(erros_cont, erros_sist, vid, anim)
    
    erros_1['relatório aluno'] = reports['student']
    erros_1['relatório professor'] = reports['teacher']
    erros_1['relatório pais'] = reports['parent']
    
    erros_1['gráfico'] = pdf_bytes
    
    return erros_1
    

# instancia o Flask, impoetações estão ao cabeçalho
app = Flask(__name__)

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    # Captura o payload (pode vir em form ou JSON)
    data = request.get_json(silent=True) or request.form
    # Extrai o ID do data desafio gerado pela tentativa do aluno
    try:
        data_challenge_id = int(data.get('data_challenge_id'))
    except (TypeError, ValueError):
        return app.response_class(
            response=json.dumps({"error": "data_challenge_id inválido"}),
            status=400,
            mimetype='application/json'
        )
    # Gera o dict de recomendações
    recs = master_func(data_challenge_id)
    # Envia o resultado como JSON
    return app.response_class(
        response=json.dumps(recs),
        status=200,
        mimetype='application/json'
    )

if __name__ == '__main__':
    # executa o servidor na porta 5000 (ou outra de sua preferência)
    app.run(host='0.0.0.0', port=5000)
    
    
    # Teste típico EF05MA01(513200) EF05MA05(476522) EF05MA03A(458626) Hab de português(447508) EF05MA15(494637) EF05MA23(480097) EF05MA03(458691)
    #recommendations = master_func(458691) 



# Teste típico EF05MA01(513200) EF05MA05(476522) EF05MA03A(458626) Hab de português(447508) EF05MA15(494637) EF05MA23(480097) EF05MA03(458691)
#recommendations = master_func(480097) 




