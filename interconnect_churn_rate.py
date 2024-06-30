# %% [markdown]
# Hola &#x1F600;
# 
# Soy **Hesus Garcia**  como "Jesús" pero con H. Sé que puede ser confuso al principio, pero una vez que lo recuerdes, ¡nunca lo olvidarás! &#x1F31D;	. Como revisor de código de Practicum, estoy emocionado de examinar tus proyectos y ayudarte a mejorar tus habilidades en programación. si has cometido algún error, no te preocupes, pues ¡estoy aquí para ayudarte a corregirlo y hacer que tu código brille! &#x1F31F;. Si encuentro algún detalle en tu código, te lo señalaré para que lo corrijas, ya que mi objetivo es ayudarte a prepararte para un ambiente de trabajo real, donde el líder de tu equipo actuaría de la misma manera. Si no puedes solucionar el problema, te proporcionaré más información en la próxima oportunidad. Cuando encuentres un comentario,  **por favor, no los muevas, no los modifiques ni los borres**. 
# 
# Revisaré cuidadosamente todas las implementaciones que has realizado para cumplir con los requisitos y te proporcionaré mis comentarios de la siguiente manera:
# 
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si todo está perfecto.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si tu código está bien pero se puede mejorar o hay algún detalle que le hace falta.
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si de pronto hace falta algo o existe algún problema con tu código o conclusiones.
# </div>
# 
# Puedes responderme de esta forma:
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class=“tocSkip”></a>
# </div>
# 
# </br>
# 
# **¡Empecemos!**  &#x1F680;

# %% [markdown]------------------------------
# # Interconnect
# %% [markdown] 
# # Librerías. 
# %%
import pandas as pd
import os

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
# %% [markdown]-------------------------------------------------------------------------------------------------------
# ## Preguntas aclaratorias
# - ¿Para qué porcentaje de clientes que planean fugarse hay presupuesto para ofrecerle promociones?
# - ¿Qué tipo de clientes son prioridad para identificar su posible fuga?

# %% [markdown]
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Hola Olaf,
# 
# Estoy entusiasmado con el progreso que has hecho en tu proyecto. Aquí te dejo algunas sugerencias para que puedas estructurar aún mejor tu entrega y aprovechar al máximo el análisis:
# 
# 1. **Análisis Exploratorio de Datos**:
#    - Busca correlaciones para reducir la dimensionalidad y identifica el desbalance de clases, lo cual es fundamental para preparar los datos para modelado.
#    - Enriquece los datos con nuevas columnas que aporten más información para los modelos predictivos.
# 
# 2. **Selección del Modelo**:
#    - Dedica tiempo a la búsqueda del mejor modelo que se ajuste a las características y necesidades de tu análisis. Considera diferentes algoritmos y compara su rendimiento basado en métricas adecuadas.
# 
# 3. **Preguntas Aclaratorias**:
#    - En relación a tus preguntas sobre el porcentaje de clientes que planean fugarse y las promociones, o el tipo de clientes a priorizar, te recomiendo definir estas respuestas a partir del análisis de datos. Aunque no contamos con todo el conocimiento del negocio en un entorno académico, puedes incluir tus suposiciones y considerar la tolerancia al riesgo en tus modelos.
# 
# Espero que estas indicaciones te ayuden a estructurar tu entrega de manera más efectiva y a profundizar en el análisis. 😊 ¡Estoy ansioso por ver los resultados detallados y bien fundamentados que vas a lograr!
# 
# Saludos cordiales,
# </div>