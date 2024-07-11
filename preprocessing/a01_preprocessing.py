# Librerías -----------------------------------------------------------------------------------

import os, sys
sys.path.append(os.getcwd())
from functions import text_changer as text_changer
import pandas as pd

# Carga los datos -----------------------------------------------------------------------------------

folder_path = "files/datasets/input/final_provider/"
df_contract = pd.read_csv(f'{folder_path}contract.csv')
df_internet = pd.read_csv(f'{folder_path}internet.csv')
df_personal = pd.read_csv(f'{folder_path}personal.csv')
df_phone = pd.read_csv(f'{folder_path}phone.csv')

# Arregla los datos -----------------------------------------------------------------------------------

# Contract -----------------------------------------------------------------------------------

#   - Estandariza los nombres de las columnas
#   - Cambia las columnas 'type' y 'payment_method' a tipo category
#   - Cambia el tipo de datos de las columnas "begin_date" y "end_date" a datetime
#   - Reemplaza valores en la columna "paperless_billing" para hacerla tipo int64
#   - Reemplaza los valores "espacio en blanco" de la columna "total_charges" a 0
#   - Cambia el tipo de datos de la columna "total_charges" a tipo float 

# Cambia los nombres de las columnas de camel_case a snake_case
df_contract.columns = pd.Series(df_contract.columns).apply(text_changer.split_camel_to_snake)
# Reemplaza la columna "id" por "customer_id"
df_contract.rename(columns={'id':'customer_id'}, inplace=True)
# Cambia las columnas "type" y "payment_method" a tipo category
df_contract[['type', 'payment_method']] = df_contract[['type', 'payment_method']].astype('category')
# Cambia las columnas "begin_date", "end_date" a tipo datetime
df_contract['begin_date'] = df_contract['begin_date'].apply(pd.to_datetime)
df_contract['end_date'] = df_contract['end_date'].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S', errors='coerce')
# Cambia la columna "paperless_billing" a tipo booleano
df_contract['paperless_billing'] = df_contract['paperless_billing'].replace({'Yes':1, 'No':0}).astype('int64')
# # Reemplaza los registros de la columna "total_charges" que tienen un espacio en blanco con 0
df_contract['total_charges'] = df_contract['total_charges'].replace(' ',0)
# Cambia las columnas "monthly_charges" y "total_charges" a float
df_contract[['total_charges']] = df_contract[['total_charges']].astype('float')

# Internet -----------------------------------------------------------------------------------

#   - Estandariza los nombres de las columnas
#   - Cambia la columna "internet_service" a tipo category
#   - Reemplaza valores de las columnas 2a a la última con 1 y 0

# Cambia los nombres de las columnas de camell a snake
df_internet.columns = pd.Series(df_internet.columns).apply(text_changer.split_camel_to_snake)
# Reemplaza la columna "id" por "customer_id"
df_internet.rename(columns={'id':'customer_id'}, inplace=True)
# Cambia la columna "internet_service" a tipo category
df_internet['internet_service'] = df_internet['internet_service'].astype('category')
# Cambia las columnas a tipo booleano excepto "customer_id" e "internet_service"
df_internet.iloc[:,2:] = df_internet.iloc[:,2:].replace({'Yes':1, 'No':0})

# Personal -----------------------------------------------------------------------------------

#   - Estandariza los nombres de las columnas
#   - Reemplaza los valores de la columna "gender" con 1 y 0
#   - Reemplaza los valores de las columnas 3 y 4 con 1 y 0
#   - Cambia los datos de las columnas "gender", "partner" y "dependents" a tipo int64


# Cambia los nombres de las columnas de camell a snake
df_personal.columns = pd.Series(df_personal.columns).apply(text_changer.split_camel_to_snake)
# Reemplaza la columna "id" por "customer_id"
df_personal.rename(columns={'id':'customer_id'}, inplace=True)
# Reemplaza Female a 1 y Male a 0
df_personal['gender'] = df_personal['gender'].replace({'Female':1, 'Male':0}).astype('int64')
# Cambia las columnas a tipo booleano excepto "customer_id" y "gender"
df_personal.iloc[:,3:] = df_personal.iloc[:,3:].replace({'Yes':1, 'No':0})
for per_col in df_personal.iloc[:,3:].columns:
    df_personal[per_col] = df_personal[per_col].astype('int64')

# Phone -----------------------------------------------------------------------------------

#   - Estandariza los nombres de las columnas
#   - Reemplaza los valores de la columna "multiple_lines" a 1 y 0
#   - Cambia el tipo de datos a int64


# Cambia los nombres de las columnas de camell a snake
df_phone.columns = pd.Series(df_phone.columns).apply(text_changer.split_camel_to_snake)
# Reemplaza la columna "id" por "customer_id"
df_phone.rename(columns={'id':'customer_id'}, inplace=True)
# Cambia la columna "multiple_lines" a tipo booleano
df_phone['multiple_lines'] = df_phone['multiple_lines'].replace({'Yes':1, 'No':0})
df_phone['multiple_lines'] = df_phone['multiple_lines'].astype('int64')

# Enriquece los datos -----------------------------------------------------------------------------------

# Crea una columna con las clases (si se fugó o no) en df_contract
df_contract['left'] = ~df_contract['end_date'].isna()
# Cambia el tipo de datos de la columna "left" a int64
df_contract['left'] = df_contract['left'].astype('int64')
# Crea columnas con el año, mes, día y día de la semana en que el cliente se unió en df_contract
df_contract['begin_year'] = df_contract['begin_date'].dt.year
df_contract['begin_month'] = df_contract['begin_date'].dt.month
df_contract['begin_day'] = df_contract['begin_date'].dt.day
df_contract['begin_dayofweek'] = df_contract['begin_date'].dt.dayofweek

# Une los conjuntos de datos en uno -----------------------------------------------------------------------------------

# Crea un df con todos los datos
df_all = df_contract.merge(df_personal, how='outer', on='customer_id')
# Incluye df_internet
df_all = df_all.merge(df_internet, how='outer', on='customer_id')
# Incluye df_phone
df_all = df_all.merge(df_phone, how='outer', on='customer_id')
# Convierte la columna "customer_id" a indice
df_all = df_all.set_index('customer_id')
# Elimina las columnas que no ocuparemos
df_all = df_all.drop(columns=['begin_date', 'end_date'])
# Elimina las observaciones con registros nulos
df_all = df_all.dropna()
# Cambia las columnas de las columnas de la 15 a la última a tipo int64
for col_all in df_all.iloc[:,15:].columns:
    df_all[col_all] = df_all[col_all].astype('int64')

# Save dataframes -----------------------------------------------------------------------------------

df_all.to_parquet('files/datasets/intermediate/a01_df_all.parquet')