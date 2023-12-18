"""
Created on Fri Dec 15 13:41:58 2023

@author: Bren Guzmán, Brenda García, María José Merino
"""
#%% LIBRERÍAS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# FASE II
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.manifold import MDS

# FASE III
from scipy.stats import yeojohnson
from scipy.stats import boxcox



#%% FASE II. COMPRENSIÓN DE LOS DATOS

#%%% Fuente de datos
# Recopilación de los datos
df = pd.read_csv("loan_data.csv")

#%%% Exploración de los datos
#%%%% Información de las variables
class_distribution = df['not.fully.paid'].value_counts() #verificar si está balanceado
print(class_distribution)

print("\n\nDescribe: \n",df.describe()) #estadísticos básicos
print("\n\n NaN Values: \n",df.isna().sum()) #Valores nulos
print("\n\nInfo:\n",df.info) #Información de dataframe
print("\n\nTipos:\n",df.dtypes) #Tipos de datos
print("\n\nValores únicos:\n",df.nunique()) #valores únicos

#%%%% Histogramas

# Se seleccionan las columnas numéricas.
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    plt.figure(figsize=(8, 6))
    plt.hist(df[col], bins=10)  
    plt.title(f'Histograma de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.show()

#%%%% Countplots
# Se seleccionan las columnas categóricas
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=col)
    plt.title(f'Gráfico de Conteo para {col}')
    plt.xlabel(col)
    plt.ylabel('Conteo')
    plt.xticks(rotation=90)  
    plt.show()

#%%%% Gráficas de caja
for col in num_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[col])
    plt.title(f'Diagrama de Caja de {col}')
    plt.show()

    
#%%% Relaciones entre variables (gráfica de dispersión)

#%%%% todas
# Especificar los colores para la variable objetivo
colors = {0: 'green', 1: 'red'}

# Añadir una columna de colores al DataFrame
df['color'] = df['not.fully.paid'].map(colors)

# Seleccionar solo las variables numéricas (excluyendo la variable objetivo y de colores)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Crear un pairplot con colores diferenciados por la variable objetivo
sns.pairplot(df, hue='not.fully.paid', palette=colors, vars=numeric_columns)
plt.suptitle('Pairplot de Variables con Colores por Variable Objetivo', y=1.02)

# Mostrar el gráfico
plt.show()

#%%%% seleccionadas

# Variables seleccionadas
selected_vars = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths']

# Crear un DataFrame con las variables seleccionadas y la variable objetivo
df_selected = df[selected_vars + ['not.fully.paid']]

# Asignar colores a la variable objetivo
colors = {0: 'green', 1: 'red'}
df_selected['colors'] = df_selected['not.fully.paid'].map(colors)

# Visualización con pairplot
sns.pairplot(df_selected, hue='not.fully.paid', palette=colors)
plt.show()

#%%% Relaciones entre variables (mapa de calor)

# Calcular las matrices de correlación
corr_pearson = df.corr(method='pearson')
corr_spearman = df.corr(method='spearman')
corr_kendall = df.corr(method='kendall')

# Configurar el tamaño de la figura
plt.figure(figsize=(15, 5))

# Mapa de calor para la correlación de Pearson
plt.subplot(1, 3, 1)
sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlación de Pearson')

# Mapa de calor para la correlación de Spearman
plt.subplot(1, 3, 2)
sns.heatmap(corr_spearman, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlación de Spearman')

# Mapa de calor para la correlación de Kendall
plt.subplot(1, 3, 3)
sns.heatmap(corr_kendall, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlación de Kendall')

# Ajustar el diseño
plt.tight_layout()

# Mostrar el gráfico
plt.show()

#%%% Visualización con reducción de dimensionalidad (PCA a 3D)

# Separar las características (X) y la variable objetivo (y)
X = df.drop(['not.fully.paid','color'], axis=1)
y = df['not.fully.paid']

# Utilizar Binary Encoder para manejar variables categóricas
encoder = ce.BinaryEncoder(cols=['purpose'])
X_encoded = encoder.fit_transform(X)

# Estandarizar las características para PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Aplicar PCA a 3 dimensiones
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Crear un DataFrame con las componentes principales y la variable objetivo
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['not.fully.paid'] = y

# Visualizar en 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot para ambas clases con colores personalizados
colors = {0: 'green', 1: 'red'}
scatter = ax.scatter(
    xs=df_pca['PC1'],
    ys=df_pca['PC2'],
    zs=df_pca['PC3'],
    c=df_pca['not.fully.paid'].map(colors),
    marker='o'
)

# Configuraciones adicionales para mejorar la visualización
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA 3 Dimensiones')

# Mostrar la leyenda
legend_labels = {0: 'Fully Paid (0)', 1: 'Not Fully Paid (1)'}
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=10, label=legend_labels[label]) for label in legend_labels]
ax.legend(handles=handles, title='Variable Objetivo', loc='upper right')

plt.show()


#%%% Visualización con reducción de dimensionalidad (PCA a 2D)

# Separar las características (X) y la variable objetivo (y)
X = df.drop(['not.fully.paid','color'], axis=1)
y = df['not.fully.paid']

# Utilizar Binary Encoder para manejar variables categóricas
encoder = ce.BinaryEncoder(cols=['purpose'])
X_encoded = encoder.fit_transform(X)

# Estandarizar las características para PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Aplicar PCA a 2 dimensiones
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Crear un DataFrame con las componentes principales y la variable objetivo
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['not.fully.paid'] = y

# Visualizar en 2D
plt.figure(figsize=(10, 8))

# Scatter plot para ambas clases con colores personalizados
colors = {0: 'green', 1: 'red'}
scatter = plt.scatter(
    x=df_pca['PC1'],
    y=df_pca['PC2'],
    c=df_pca['not.fully.paid'].map(colors),
    marker='o'
)

# Configuraciones adicionales para mejorar la visualización
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA 2 Dimensiones')

# Mostrar la leyenda
legend_labels = {0: 'Fully Paid (0)', 1: 'Not Fully Paid (1)'}
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=10, label=legend_labels[label]) for label in legend_labels]
plt.legend(handles=handles, title='Variable Objetivo', loc='upper right')

plt.show()


#%%% Visualización con reducción de dimensionalidad (MDS a 3D)

# Separar las características (X) y la variable objetivo (y)
X = df.drop(['not.fully.paid', 'color'], axis=1)
y = df['not.fully.paid']

# Utilizar Binary Encoder para manejar variables categóricas
encoder = ce.BinaryEncoder(cols=['purpose'])
X_encoded = encoder.fit_transform(X)

# Estandarizar las características para MDS
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Aplicar MDS a 3 dimensiones
mds = MDS(n_components=3)
X_mds = mds.fit_transform(X_scaled)

# Crear un DataFrame con las dimensiones MDS y la variable objetivo
df_mds = pd.DataFrame(X_mds, columns=['Dimension 1', 'Dimension 2', 'Dimension 3'])
df_mds['not.fully.paid'] = y

# Visualizar en 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot para ambas clases con colores personalizados
colors = {0: 'green', 1: 'red'}
scatter = ax.scatter(
    xs=df_mds['Dimension 1'],
    ys=df_mds['Dimension 2'],
    zs=df_mds['Dimension 3'],
    c=df_mds['not.fully.paid'].map(colors),
    marker='o'
)

# Configuraciones adicionales para mejorar la visualización
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('MDS 3 Dimensiones')

# Mostrar la leyenda
legend_labels = {0: 'Fully Paid (0)', 1: 'Not Fully Paid (1)'}
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=10, label=legend_labels[label]) for label in legend_labels]
ax.legend(handles=handles, title='Variable Objetivo', loc='upper right')

plt.show()

#%%% Visualización con reducción de dimensionalidad (MDS a 2D)

# Separar las características (X) y la variable objetivo (y)
X = df.drop(['not.fully.paid', 'color'], axis=1)
y = df['not.fully.paid']

# Utilizar Binary Encoder para manejar variables categóricas
encoder = ce.BinaryEncoder(cols=['purpose'])
X_encoded = encoder.fit_transform(X)

# Estandarizar las características para MDS
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Aplicar MDS a 2 dimensiones
mds = MDS(n_components=2)
X_mds = mds.fit_transform(X_scaled)

# Crear un DataFrame con las dimensiones MDS y la variable objetivo
df_mds = pd.DataFrame(X_mds, columns=['Dimension 1', 'Dimension 2'])
df_mds['not.fully.paid'] = y

# Visualizar en 2D
plt.figure(figsize=(10, 8))

# Scatter plot para ambas clases con colores personalizados
colors = {0: 'green', 1: 'red'}
scatter = plt.scatter(
    x=df_mds['Dimension 1'],
    y=df_mds['Dimension 2'],
    c=df_mds['not.fully.paid'].map(colors),
    marker='o'
)

# Configuraciones adicionales para mejorar la visualización
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('MDS 2 Dimensiones')

# Mostrar la leyenda
legend_labels = {0: 'Fully Paid (0)', 1: 'Not Fully Paid (1)'}
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=10, label=legend_labels[label]) for label in legend_labels]
plt.legend(handles=handles, title='Variable Objetivo', loc='upper right')

plt.show()


#%% FASE III. TRANSFORMACIÓN DE LOS DATOS

#%%% Codificación de purpose (Binary Encoder)

# Inicializar el codificador BinaryEncoder
encoder = ce.BinaryEncoder(cols=['purpose'])

# Aplicar la codificación al DataFrame
df_encoded = encoder.fit_transform(df)

# Ver el DataFrame resultante
print(df_encoded.head())

#%%% Manejo de valores atípicos

columns_of_interest = ['installment', 'log.annual.inc', 'fico', 'days.with.cr.line', 'revol.bal', 'inq.last.6mths']

# Calcula el Z-Score para cada columna de interés
for column in columns_of_interest:
    z_score_column = column + '_Z_Score'
    df_encoded[z_score_column] = (df_encoded[column] - df_encoded[column].mean()) / df_encoded[column].std()

# Define el umbral para identificar valores atípicos
umbral = 2

# Identifica registros con valores atípicos en al menos una de las columnas
outliers_df = df_encoded[(df_encoded.filter(regex='_Z_Score').abs() > umbral).any(axis=1)]

# Muestra el DataFrame con valores atípicos
print(outliers_df)

# Elimina los registros en df_encoded que están en outliers_df
indices_to_drop = outliers_df.index
df_encoded = df_encoded.drop(indices_to_drop)


# Lista de las columnas a eliminar
columns_to_drop = [column + '_Z_Score' for column in columns_of_interest]

# Eliminar las columnas
df_encoded = df_encoded.drop(columns=columns_to_drop)

#df_encoded = df_encoded.drop(['color'], axis=1)

#%%% Normalización de los datos.

# Selecciona todas las columnas numéricas para aplicar la normalización
#numeric_columns = df_encoded.select_dtypes(include=['float64', 'int64']).columns
numeric_columns = [ 'int.rate', 'installment', 'log.annual.inc', 'dti',
                   'fico', 'days.with.cr.line', 'revol.bal', 'revol.util']

# Inicializa el objeto StandardScaler
scaler = StandardScaler()

# Aplica la normalización Z-Score a todas las columnas seleccionadas
df_encoded[numeric_columns] = scaler.fit_transform(df_encoded[numeric_columns])

# Muestra el DataFrame resultante con las columnas normalizadas
print(df_encoded)

#df_encoded.to_csv("data.csv", index=False)

#%%% Transformaciones para aproximar a distribución normal (Yeo-Johnson, Box-Cox)

#df_encoded = pd.read_csv("data.csv")

columns_to_transform = ['days.with.cr.line', 'revol.bal']

for col in columns_to_transform:
    plt.figure(figsize=(8, 6))
    plt.hist(df_encoded[col], bins=10)  
    plt.title(f'Histograma de {col} (antes)')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.show()
 

df_encoded['days.with.cr.line'], _ = yeojohnson(df_encoded['days.with.cr.line'] + 1)  # Agregamos 1 para manejar valores no positivos
df_encoded['revol.bal'], _ = boxcox(df_encoded['revol.bal'] + 1)  # Agregamos 1 para manejar valores no positivos

for col in columns_to_transform:
    plt.figure(figsize=(8, 6))
    plt.hist(df_encoded[col], bins=10)  
    plt.title(f'Histograma de {col} (después)')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.show()
    
#df_encoded.to_csv("data.csv", index=False)

