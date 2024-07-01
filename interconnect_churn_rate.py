# %% [markdown]------------------------------
# # Interconnect
# %% [markdown]Librerias----------------------------------------------------------------------------------------------------------------------------------
# # Librerías. 
# %%
import pandas as pd
import os

# %% [markdown]Funciones----------------------------------------------------------------------------------------------------------------------------------
# ## Funciones

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