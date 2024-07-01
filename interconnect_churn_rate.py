# %% [markdown]------------------------------
# # Interconnect
# %% [markdown]Librerias----------------------------------------------------------------------------------------------------------------------------------
# # Librerías. 
# %%
import pandas as pd
import os
import re

# %% [markdown]Funciones----------------------------------------------------------------------------------------------------------------------------------
# ## Funciones
# %%
# Función para cambiar de camel_case a snake_case
def split_camel_to_snake(string, case='camel'):
    """
    Función que aplica un split a una cadena en camel o dromedary case y la retorna como snake case.
    """
    if case == 'camel':
        if string.islower():
            return string
        else:
            # Split camel case
            return '_'.join(re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', string)).lower()
        
    elif case == 'dromedary':
        if string.islower():
            return string
        else:
            # Split dromedary case
            return '_'.join(re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', string)).lower()

# %% [markdown]--------------------------------------------------------------------------------------------------------
# ## Carga de los datos.
# %%
if os.path.exists("files/datasets/final_provider/"):
    folder_path = "files/datasets/final_provider/"
else:
    folder_path = "/datasets/final_provider/"

df_contract = pd.read_csv(f'{folder_path}contract.csv')
df_internet = pd.read_csv(f'{folder_path}internet.csv')
df_personal = pd.read_csv(f'{folder_path}personal.csv')
df_phone = pd.read_csv(f'{folder_path}phone.csv')
# %% [markdown]----------------------------------------------------------------------------------------------------------
# ## Muestra de los datos
# %% [markdown]------------------------------------------------------
# ### Contract
# %%
# Imprime la información del dataframe
df_contract.info()
# %%
# Imprime una muestra de los datos
df_contract.sample(5)
# %% [markdown]
# #### Comentario.
# - Cambiemos los nombres de las columnas para un mejor manejo de los datos.
# - No tenemos datos nulos.
# - Podemos probar creando columnas como:
#   - Tipo de clase (Se fugó o no el cliente)
#   - Días de contrato
# - Cambiar los tipos de datos de las columnas "BeginDate", "EndDate"  a tipo datetime
# - Investiga el motivo de que la columna "TotalCharges" sea de tipo object
# - Cambiar el tipo de datos de la columna "TotalCharges" a tipo float
# %% [markdown]------------------------------------------------------------
# ### Internet
# %%
# Muestra la información del dataframe
df_internet.info()
# %%
# Imprime una muestra de los datos.
df_internet.sample(5)
# %% [markdown]
# #### Comentario
# - No tenemos datos nulos
# - Confirmar que tenemos columnas booleanas y cambiar el tipo de dato al mismo.
# - Confirmar los datos únicos de la columna "InternetService" 
# - Podríamos agregar los clientes faltantes con registros con "unknown"
# %% [markdown]------------------------------------------------------------
# ### Personal
# %%
# Muestra la información del dataframe
df_personal.info()
# %% [markdown]
# Imprime una muestra de los datos
df_personal.sample(5)
# %% [markdown]
# - No tenemos datos nulos
# - Confirmar que tenemos columnas booleanas y cambiar el tipo de dato al mismo
# - Confirmar los datos únicos de la columna "gender" 
# %% [markdown]---------------------------------------------------------
# ### Phone
# %%
# Muestra la información del dataframe
df_phone.info()
# %%
# Imprime una muestra de los datos
df_phone.sample(5)
# %% [markdown]
# #### Comentario
# - No tenemos datos nulos
# - Confirmar que tenemos datos booleanos y cambiar el tipo de dato al mismo
# - Podríamos agregar los clientes faltantes con registros con "unknown"
# %% [markdown]----------------------------------------------------------------------------------------------------------------------------------
# ## Arregla los datos (Parte 1)
# %% [markdown]------------------------------------------------------------
# ### Contract
# %%
# Cambia los nombres de las columnas de camell a snake
df_contract.columns = pd.Series(df_contract.columns).apply(split_camel_to_snake)
# Reemplaza la columna "id" por "customer_id"
df_contract.rename(columns={'id':'customer_id'}, inplace=True)
# formato de fecha
date_format = "%Y-%m-%d"
# Cambia las columnas "begin_date", "end_date" a tipo datetime
df_contract[['begin_date', 'end_date']] = df_contract[['begin_date', 'end_date']].apply(pd.to_datetime, format=date_format, errors='coerce')
# Cambia la columna "paperless_billing" a tipo booleano
df_contract['paperless_billing'] = df_contract['paperless_billing'].replace({'Yes':True, 'No':False})
# Cambia el tipo de datos a booleano
df_contract['paperless_billing'] = df_contract['paperless_billing'].astype(bool)
# # Reemplaza los registros de la columna "total_charges" que tienen un espacio en blanco con 0
df_contract['total_charges'] = df_contract['total_charges'].replace(' ',0)
# Cambia las columnas "monthly_charges" y "total_charges" a float
df_contract[['monthly_charges', 'total_charges']] = df_contract[['monthly_charges',  'total_charges']].astype('float')
# %% [markdown]-----------------------------------------------------------
# ### Internet
# %%
# Cambia los nombres de las columnas de camell a snake
df_internet.columns = pd.Series(df_internet.columns).apply(split_camel_to_snake)
# Reemplaza la columna "id" por "customer_id"
df_internet.rename(columns={'id':'customer_id'}, inplace=True)
# Cambia las columnas a tipo booleano excepto "customer_id" e "internet_service"
df_internet.iloc[:,2:] = df_internet.iloc[:,2:].replace({'Yes':True, 'No':False}).astype(bool)
# %% [markdown]-----------------------------------------------------------
# ### Personal
# %%
# Cambia los nombres de las columnas de camell a snake
df_personal.columns = pd.Series(df_personal.columns).apply(split_camel_to_snake)
# Reemplaza la columna "id" por "customer_id"
df_personal.rename(columns={'id':'customer_id'}, inplace=True)
# Cambia la columna "senior_citizen" a tipo booleano
df_personal['senior_citizen'] = df_personal['senior_citizen'].astype(bool)
# Cambia las columnas a tipo booleano excepto "customer_id" y "gender"
df_personal.iloc[:,3:] = df_personal.iloc[:,3:].replace({'Yes':True, 'No':False}).astype(bool)
# %% [markdown]----------------------------------------------------------
# ### Phone
# %%
# Cambia los nombres de las columnas de camell a snake
df_phone.columns = pd.Series(df_phone.columns).apply(split_camel_to_snake)
# Reemplaza la columna "id" por "customer_id"
df_phone.rename(columns={'id':'customer_id'}, inplace=True)
# Cambia la columna "multiple_lines" a tipo booleano
df_phone['multiple_lines'] = df_phone['multiple_lines'].replace({'Yes':True, 'No':False}).astype(bool)