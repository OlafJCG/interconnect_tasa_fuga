# %% [markdown]------------------------------
# # Interconnect
# %% [markdown] 
# # Librerías. 
# %%
import pandas as pd
import os




# %% [markdown]----------------------------------------------------------------------------------------------------------------------------------
# ## Funciones

# %% [markdown]------------------------------
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
# %% [markdown]------------------------------
# ## Muestra de los datos
# %% [markdown]-----
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
#   - ¿Cuántos meses permaneció?
# - Cambiar los tipos de datos de las columnas "BeginDate", "EndDate"  a tipo datetime
# - Cambiar los tipos de datos de las columnas "MonthlyCharges", "TotalCharges" a tipo float
# %% [markdown]-----
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
# %% [markdown]-----
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
# %% [markdown]-----
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
# %% [markdown]-------------------------------------------------------------------------------------------------------
# ## Plan de Trabajo.
# - Preprocesamiento, revisar si surgen errores al arreglar los datos.
#   - Cambiar tipo de datos
#   - Unir conjuntos de datos 
# - Revisar los datos que perderíamos o que están ausentes por la inconsistencia de registros entre los diferentes conjuntos de datos.
# - Análisis exploratorio de datos.
#   - Buscar correlacion de datos para reducir la dimensionalidad
#   - Identificar el desbalance de clases
# - Enriquecer los datos con nuevas columnas
# - Busqueda del mejor modelo