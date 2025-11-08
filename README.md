# README — MovieLens 25M (Proyecto de Concurrencia/Distribución)

## 0) Resumen del dataset
El dataset **MovieLens 25M** contiene **25,000,095 calificaciones** y **1,093,360 etiquetas** sobre **62,423 películas** evaluadas por **162,541 usuarios (1995–2019)**.  
Formato CSV con cabecera (UTF-8, separador `,`).  
Archivos principales: **ratings.csv** y **movies.csv**.  
Los demás archivos son opcionales y se utilizarán solo para la interfaz o un modelo híbrido posterior.

---

## 1) Archivos y propósito

| Archivo | ¿Para qué sirve en general? | ¿Para qué lo usaremos nosotros? |
|----------|-----------------------------|----------------------------------|
| ratings.csv | Calificaciones de usuarios (base del filtrado colaborativo). | **Core:** matriz usuario-ítem, similitudes, top-k vecinos y recomendaciones. |
| movies.csv | Metadatos de películas (título y géneros). | Mostrar títulos/géneros en la interfaz y justificar recomendaciones. |
| tags.csv | Etiquetas libres que los usuarios aplican a películas. | No se usará en PC3. Podría emplearse en la interfaz o modelo híbrido. |
| genome-tags.csv | Lista de tags curados del Tag Genome. | No se usará. Aplicable solo para un enfoque híbrido content-based. |
| genome-scores.csv | Relevancia (0–1) de cada tag del genome por película. | No se usará. Relevante para contenido semántico avanzado. |
| links.csv | IDs externos (IMDB/TMDB). | Solo para futura integración visual o con APIs externas. |

> Para PC3/PC4 bastará con **ratings.csv** y **movies.csv**.

---

## 2) Diccionario de datos

### 2.1 ratings.csv

| Campo | Tipo | Descripción |
|--------|------|--------------|
| userId | int | Identificador anónimo de usuario. |
| movieId | int | Identificador único de película. |
| rating | float (0.5–5.0) | Calificación con incrementos de 0.5. |
| timestamp | int64 (UNIX) | Momento de la calificación (UTC). |

Notas: Ordenado por `userId` y luego `movieId`. Es disperso (sparse): no todos los usuarios califican todas las películas.

### 2.2 movies.csv

| Campo | Tipo | Descripción |
|--------|------|--------------|
| movieId | int | Identificador de película. |
| title | string | Título y año entre paréntesis (UTF-8). |
| genres | string | Géneros separados por `|` (p. ej. Action\|Comedy). |

---

## 3) Relaciones conceptuales

- **ratings** (`userId`, `movieId`) conecta usuarios con películas.  
- **movies** (`movieId`) define el catálogo de películas.  
- El sistema forma un **grafo bipartito usuarios–películas**, donde las aristas están ponderadas por ratings.

> Otros archivos (`tags`, `genome`, `links`) aportan metadatos opcionales para futuras etapas de interfaz o filtrado híbrido.

---

## 4) Preprocesamiento

El modelo utilizará **Item-Based Collaborative Filtering** con similitud **Coseno** (y prueba comparativa con Pearson).  
Dataset base: `ratings.csv` (para construir la matriz usuario-ítem) y `movies.csv` (para enriquecer la interfaz).

### Pipeline de PC3

1. **Carga inicial** → leer `ratings.csv` y `movies.csv`.  
2. **Limpieza** → verificar rangos, nulos, duplicados y usuarios/películas con baja actividad.  
3. **Remapeo** → convertir `userId` y `movieId` en índices contiguos (`uIdx`, `iIdx`).  
4. **Normalización** → centrar ratings por usuario (`r' = r - media_usuario`).  
5. **Estructuras dispersas** → índices invertidos `user→(item, r')` e `item→(user, r')`.  
6. **Cálculo de similitud** → aplicar Coseno (baseline) y comparar con Pearson.  
7. **Selección top-k vecinos** → guardar los k ítems más similares.  
8. **Benchmarks** → medir rendimiento (5%, 10%, 25%, 50%, 100%) con versión secuencial y concurrente.

> Para PC3 no se usará base de datos. Para PC4 se guardarán similitudes y recomendaciones en MongoDB y se cachearán en Redis.
