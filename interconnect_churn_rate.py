# %% [markdown]------------------------------
# # Interconnect
# %% [markdown]Librerias----------------------------------------------------------------------------------------------------------------------------------
# ## Librerías. 
# %%
import pandas as pd
import os
import re

from matplotlib import pyplot as plt

from sklearn import metrics
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder

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

## Funcion para evaluar los modelos
def evaluate_model(model, train_features, train_target, test_features, test_target):
    
    eval_stats = {}
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6)) 
    
    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):
        
        eval_stats[type] = {}
    
        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]

        eval_stats[type]['Accuracy'] = metrics.accuracy_score(target, pred_target)
        eval_stats[type]['F1'] = metrics.f1_score(target, pred_target)

        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]
        
        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)    
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps

        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # Valor F1
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'Valor F1') 

        # ROC
        ax = axs[1]    
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'            
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')        
        ax.set_title(f'Curva ROC')
        
        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC') 

    df_eval_stats = pd.DataFrame(eval_stats)
    # df_eval_stats = df_eval_stats.reindex(index=    ('Exactitud', 'F1', 'ROC AUC', 'APS'))
    df_eval_stats = df_eval_stats.rename(index={'Accuracy':'Exactitud'})
    df_eval_stats = df_eval_stats.round(2)
    
    print(df_eval_stats)
    
    return

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
df_contract['paperless_billing'] = df_contract['paperless_billing'].replace({'Yes':1, 'No':0})
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
df_internet.iloc[:,2:] = df_internet.iloc[:,2:].replace({'Yes':1, 'No':0})
# %% [markdown]-----------------------------------------------------------
# ### Personal
# %%
# Cambia los nombres de las columnas de camell a snake
df_personal.columns = pd.Series(df_personal.columns).apply(split_camel_to_snake)
# Reemplaza la columna "id" por "customer_id"
df_personal.rename(columns={'id':'customer_id'}, inplace=True)
# Reemplaza Female a 1 y Male a 0
df_personal['gender'] = df_personal['gender'].replace({'Female':1, 'Male':0})
# Cambia las columnas a tipo booleano excepto "customer_id" y "gender"
df_personal.iloc[:,3:] = df_personal.iloc[:,3:].replace({'Yes':1, 'No':0})
# %% [markdown]----------------------------------------------------------
# ### Phone
# %%
# Cambia los nombres de las columnas de camell a snake
df_phone.columns = pd.Series(df_phone.columns).apply(split_camel_to_snake)
# Reemplaza la columna "id" por "customer_id"
df_phone.rename(columns={'id':'customer_id'}, inplace=True)
# Cambia la columna "multiple_lines" a tipo booleano
df_phone['multiple_lines'] = df_phone['multiple_lines'].replace({'Yes':1, 'No':0})
# %% [markdown]----------------------------------------------------------------------------------------------------------------------------------
# ## EDA
# %% [markdown]-----------------------------------------------------------
# ### Contract

# %%
# Muestra un boxplot de las columnas "monthly_charges" y "total_charges"
for col in ['monthly_charges', 'total_charges']:
    plt.boxplot(df_contract[col])
    plt.title(f"Boxplot de la columna {col} para visualizar la distribución de los datos")
    plt.xlabel(f"{col}")
    plt.ylabel("Cantidad")
    plt.show()
# %%
# Muestra un gráfico con las fechas y la cantidad de clientes que se fugaron
plt.barh(list(df_contract[~df_contract['end_date'].isna()]['end_date'].value_counts().index), df_contract[~df_contract['end_date'].isna()]['end_date'].value_counts().values)
plt.title("Fechas y cantidad de clientes que se fugaron.")
plt.ylabel("Fechas")
plt.xlabel("Cantidad")
plt.show()
# %% [markdown]
# #### Comentario
# - A pesar de que nuestro boxplot no muestra datos atipicos, si vemos un sesgo en la distribución de "total_charges" debido a que son pocos los clientes que ya llevan más tiempo en la empresa.
# - Escalemos nuestras columnas numericas
# - Observamos que los clientes que se fugaron, lo hicieron en los meses 10, 11, 12 y 1, de los cuales los 3 primeros corresponden al año 2019 y el último a 2020, sería recomendable identificar la posible causa de esto.
# %% [markdown]--------------------------------------------------------------------------
# ## Enriquece los datos
# %%
# Crea una columna con las clases (si se fugó o no)
df_contract['left'] = ~df_contract['end_date'].isna()
# Crea columnas con el año, mes, día y día de la semana en que el cliente se unió
df_contract['begin_year'] = df_contract['begin_date'].dt.year
df_contract['begin_month'] = df_contract['begin_date'].dt.month
df_contract['begin_day'] = df_contract['begin_date'].dt.day
df_contract['begin_dayofweek'] = df_contract['begin_date'].dt.dayofweek

# %% [markdown]--------------------------------------------------------------------------
# ## Unión de los conjuntos de datos
# %%
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
# %%
# Elimina las observaciones con registros nulos
df_all = df_all.dropna()
# %%
# Cambia columnas a tipo bool
df_all.iloc[:,10:] = df_all.iloc[:,10:].astype('bool')
# %% [markdown]----------------------------------------------------------------------------------------------------------------------------------
# ## Busqueda de modelo
# %% [markdown]----------------------------------------------------------------------------------------------------------------------------------
# ## Escala las características numericas
# %%
# Escala las características númericas
df_all[['monthly_charges', 'total_charges']] = RobustScaler().fit_transform(df_all[['monthly_charges', 'total_charges']])
# %% [markdown]----------------------------------------------------------------------------------------------------------------------------------
# ### Preprocesamiento para Regresión Logística
# %%
# Crea una copia del df
df_lr = df_all.copy()
# %%
## Codifica las columnas "type", "payment_method", "gender" e "internet_service" con OHE
data_ohe = pd.get_dummies(df_lr[['type', 'payment_method', 'internet_service']], drop_first=True)
# Elimina las columnas que se codificaron
df_lr = df_lr.drop(columns=['type', 'payment_method', 'internet_service'])
# Agrega data_ohe al df
df_lr = df_lr.join(data_ohe)
# %% [markdown]
# ### Separa los datos
# %%
# Separa en conjuntos
X_train_lr, X_lr, y_train_lr, y_lr = train_test_split(df_lr.drop('left',axis=1), df_lr['left'], test_size=0.6, random_state=12345)
X_valid_lr, X_test_lr, y_valid_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.5, random_state=12345)

# %% [markdown]---------
# ### Regresión Logística
# %%
# Crea una instancia del modelo
clf_lr = LogisticRegression(class_weight='balanced', random_state=12345).fit(X_train_lr, y_train_lr)

# %%
# Evalúa el modelo
evaluate_model(clf_lr, X_train_lr, y_train_lr, X_valid_lr, y_valid_lr)
# %% [markdown]----------------------------------------------------------------------------------------------------------------------------------
# ### LighGBM con el conjunto para Regresión Logística

# %%
# Pasa el modelo por gridsearch
lgb_param_grid = {
    'num_leaves': ['2^max_depth']+list(range(6,32,5)),
    'max_depth':['-1']+list(range(5, 31, 5)),
    'class_weight' : ['balanced'],
    'learning_rate' : [0.05, 0.08, 0.1, 0.3, 0.5]
}
clf_lgb = lgb.LGBMClassifier(metric = 'auc', random_state=12345)
lgb_GSCV = GridSearchCV(clf_lgb, param_grid=lgb_param_grid, verbose=50).fit(X_train_lr, y_train_lr)
lgb_GSCV.best_params_

# %%
# Entrena el modelo con los mejores parámetros
clf_lgb = lgb.LGBMClassifier(max_depth=10, num_leaves=21, learning_rate=0.3, class_weight='balanced', metric='auc', random_state=12345).fit(X_train_lr, y_train_lr)
# %%
# Evalúa el modelo
evaluate_model(clf_lgb, X_train_lr, y_train_lr, X_valid_lr, y_valid_lr)
# %% [markdown]----------------------------------------------------------------------------------------------------------------------------------
# ### Bosque Aleatorio
# %%
# Crea una copia del df para bosque
df_rf = df_all.copy()
# %% [markdown]----------------------------------------------------------
# #### Preprocesamiento para Bosque
# %%
# Codifica las características
rf_encoder = OrdinalEncoder()
df_rf[['type', 'payment_method', 'internet_service']] = rf_encoder.fit_transform(df_rf[['type', 'payment_method', 'internet_service']])

# %%
# Separa el df en conjuntos de entrenamiento, validacion y prueba
X_train_rf, X_rf, y_train_rf, y_rf = train_test_split(df_rf.drop('left',axis=1), df_rf['left'], test_size=0.6, random_state=12345)
X_valid_rf, X_test_rf, y_valid_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.5,random_state=12345)

# %%
# Crea una instancia del modelo
clf_rf = RandomForestClassifier(random_state=12345)

# %%
# Pasa el modelo por gridsearchcv
rf_param_grid = {
    'n_estimators':list(range(30,101, 10)),
    'criterion':['gini', 'entropy', 'log_loss'],
    'max_depth':['None']+list(range(5, 51, 5)),
    'class_weight':['None','balanced', 'balanced_subsample']
}
# Busqueda del mejor grid
clf_rf_GSCV = GridSearchCV(clf_rf, param_grid=rf_param_grid, scoring='roc_auc', verbose=50).fit(X_train_rf, y_train_rf)
clf_rf_GSCV.best_params_

# %%
# Entrena el modelo con los mejores parámetros
clf_rf = RandomForestClassifier(class_weight='balanced_subsample', n_estimators=80, criterion='entropy', max_depth=10, random_state=12345).fit(X_train_rf, y_train_rf)
# %%
# Evalúa el modelo
evaluate_model(clf_rf, X_train_rf, y_train_rf, X_valid_rf, y_valid_rf)
# %% [markdown]
# #### Comentario
# - El mejor modelo fue con LightGBM
# - El descenso de gradiente hizo la diferencia