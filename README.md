# README — PC3

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

# 4) Preprocesamiento — Etapa 1: Análisis de Datos 

Este documento describe **qué hicimos, por qué lo hicimos y cómo usar los artefactos** generados en el preprocesamiento del dataset **MovieLens 25M** para el proyecto de **Filtrado Colaborativo (CF)** concurrente.

> **Resumen del pipeline**
>
> 1. **Inspección (clean.go)** → diagnóstico sin modificar datos.  
> 2. **Filtrado por soporte + Remapeo (remap.go)** → quedarnos con ítems con suficiente señal y mapear a índices contiguos.  
> 3. **Normalización + CSR (normalize.go)** → centrar ratings por usuario y construir la matriz dispersa en formato eficiente.  
>
> Salidas principales: `artifacts/ratings_min5.csv`, `artifacts/ratings_ui.csv`, `artifacts/index/*`, `artifacts/user_means.csv`, `artifacts/matrix_user_csr/*`.

---

## 4.1 Inspección (clean.go)

**Objetivo.** Explorar la calidad de `ratings.csv` y justificar reglas de limpieza.  
**Acciones realizadas:**
- Verificación de **nulos**, **rango de ratings** \([0.5, 5.0] en pasos de 0.5\) y **duplicados consecutivos** `(userId, movieId)`.
- Cálculo de **distribuciones de ratings**: histograma de 0.5 en 0.5 y por estrellas enteras (1–5).
- **Actividad por usuario** y **por ítem** (min, max, avg y buckets: `0-4, 5-9, 10-19, 20-49, 50-99, 100+`).

**Hallazgos clave (resumen de `artifacts/clean_report.txt`):**
- Filas totales: **25,000,095** (todas válidas; sin nulos, fuera de rango ni pasos inválidos).
- Usuarios distintos: **162,541**.
- Películas con al menos un rating: **59,047** (en `movies.csv` hay 62,423).  
- **Actividad por película** muestra un **gran número de ítems con muy pocos ratings**:  
  - `<5 ratings`: **26,327** películas (44.59%)  
  - `<10 ratings`: **34,717** películas (58.80%)

**Regla de limpieza decidida (soporte mínimo por ítem):**
- Conservar **películas con ≥5 ratings**.  
  **Motivación**: Coseno y Pearson **son inestables** con soporte muy pequeño (pocas co-ocurrencias) y generan ruido; además, el cómputo de similitud se reduce drásticamente.

**Resultado del filtrado** (ver `artifacts/clean_filter_report.txt`):
- **Filas originales**: 25,000,095  
- **Filas retenidas**: **24,945,870**  
- **Filas eliminadas**: 54,225 (**0.22%**)
- **Películas retenidas**: **32,720** (se eliminaron 26,327 con <5 ratings)  
- **Usuarios en limpio**: 162,541 (todos mantienen ≥1 rating tras el corte)

> Nota: El corte es **para cómputo de similitudes**. 

---

## 4.2 Filtrado + Remapeo (remap.go)

Tras aplicar el soporte mínimo, trabajamos con `artifacts/ratings_min5.csv` (cabecera: `userId,movieId,rating,timestamp`).

### ¿Por qué remapear?
Los IDs originales (MovieLens) **no son contiguos** (p. ej., movieId 1, 296, 307…). Para construir estructuras matriciales eficientes necesitamos índices **0..U-1** y **0..I-1**.

### Qué hace el remapeo
Construimos dos diccionarios:
- `userId → uIdx` (índice de fila 0..U-1)
- `movieId → iIdx` (índice de columna 0..I-1)

Y generamos el fichero **ordenado por `uIdx`**:
- `artifacts/ratings_ui.csv` con columnas: `uIdx,iIdx,rating`

Además, persistimos los mapas para consumo por la API/UI:
- `artifacts/index/user_map.csv`  → `userId,uIdx`
- `artifacts/index/item_map.csv`  → `movieId,iIdx`

**Tamaños alcanzados** (ver `artifacts/remap_report.txt`):
- Usuarios (U): **162,541**
- Ítems (I): **32,720**
- Ratings (NNZ): **24,945,870**

> **Propiedad importante**: el archivo `ratings_ui.csv` queda **ordenado por `uIdx` y después por `iIdx`**, lo que facilita su conversión directa a CSR.

---

### 4.3 Normalización + Matriz Dispersa CSR (normalize.go)

### 4.3.1 ¿Qué hace este proceso?

El script `normalize.go` realiza la **normalización de los ratings** y la **construcción de matrices dispersas (CSR)** para representar de forma eficiente los datos de usuarios e ítems.  
En esta versión, se incorporó la posibilidad de generar **centrado por usuario, centrado por ítem o ambos** mediante el parámetro `--axis`.

#### Entrada principal
- `artifacts/ratings_ui.csv` → columnas: `uIdx, iIdx, rating`

#### Salidas principales (según el eje seleccionado)

| Eje | Archivos generados | Descripción |
|-----|--------------------|--------------|
| **user** | `user_means.csv` y `matrix_user_csr/*` | Centrado por usuario: \( r'_{u,i} = r_{u,i} - \mu_u \) |
| **item** | `item_means.csv` y `matrix_item_csr/*` | Centrado por ítem: \( r'_{u,i} = r_{u,i} - \mu_i \) |
| **both** | Genera ambas versiones | Recomendado para usar tanto Pearson user-based como item-based |

### 4.3.2 ¿Qué es “centrar por usuario” y “centrar por ítem”?

Cada usuario o ítem tiene un sesgo propio (algunos puntúan alto, otros bajo). Para evitar que ese sesgo **distorsione la similitud**, se aplica un centrado que resta la media correspondiente:

#### a) Centrado por usuario
\[
\mu_u = \frac{1}{n_u}\sum_{i \in I(u)} r_{u,i}
\qquad
r'_{u,i} = r_{u,i} - \mu_u
\]

#### b) Centrado por ítem
\[
\mu_i = \frac{1}{n_i}\sum_{u \in U(i)} r_{u,i}
\qquad
r'_{u,i} = r_{u,i} - \mu_i
\]

Donde:
- \( \mu_u, \mu_i \): medias de usuario o ítem  
- \( n_u, n_i \): número de valoraciones por usuario o ítem  
- \( r'_{u,i} \): rating centrado (desviación respecto a la media)

Este centrado es fundamental para **Pearson**, ya que la correlación mide la relación entre **desviaciones respecto a la media**.  
En cambio, **Coseno** y **Jaccard** no requieren normalización, ya que se basan en la forma del vector o la co-ocurrencia binaria.

### 4.3.3 ¿Por qué usar formato CSR (Compressed Sparse Row)?

La matriz usuario–ítem contiene millones de celdas vacías. Para optimizar el almacenamiento y acceso, se usa el formato **CSR**, que guarda solo las entradas no nulas.  
Este formato permite iterar rápidamente por las filas (usuarios o ítems).

| Archivo | Tipo | Contenido | Descripción |
|----------|------|------------|--------------|
| `indptr.bin` | `int64` (len = filas+1) | Punteros de inicio/fin de cada fila | Define los límites de cada usuario o ítem |
| `indices.bin` | `int32` (len = NNZ) | Índices de columna | Identifica a qué ítem o usuario pertenece cada valor |
| `data.bin` | `float32` (len = NNZ) | Ratings centrados \( r'_{u,i} \) | Valores normalizados |
| `meta.json` | JSON | `{users, items, nnz, dtypes}` | Metadatos del CSR |
| `*_means.csv` | CSV | Medias por usuario o ítem | Necesario para reconstruir predicciones |
| `normalize_report.txt` | TXT | Resumen general | Incluye conteos y rutas de salida |

**Ejemplo de dimensiones finales** (según `normalize_report.txt`):
- Usuarios (U): 162,541  
- Ítems (I): 32,720  
- Ratings (NNZ): 24,945,870

### 4.3.4 Cómo se usarán las matrices

#### Acceso por usuario (CSR)
```go
for k := indptr[u]; k < indptr[u+1]; k++ {
    i := indices[k]  // índice del ítem
    r := data[k]     // r' = r - mu[u]
}
```

#### Acceso por ítem (CSR invertido)
```go
for k := indptr[i]; k < indptr[i+1]; k++ {
    u := indices[k]  // índice del usuario
    r := data[k]     // r' = r - mu[i]
}
```

#### Ejemplo de reconstrucción de predicción (Pearson)
\[
\widehat{r}_{u,i} = \mu_u +
\frac{\sum_{v \in N_k(u)} s(u,v) \cdot r'_{v,i}}
{\sum_{v \in N_k(u)} |s(u,v)|}
\]

donde:
- \( s(u,v) \): similitud entre usuarios (o entre ítems)
- \( N_k(u) \): los k vecinos más similares

#### Ejecución recomendada
```bash
go run -tags normalize ./cmd/preprocess/normalize.go --axis=both
```

Esto generará las dos estructuras necesarias para:
- **User-Pearson** → centrado por usuario  
- **Item-Pearson** → centrado por ítem  
- **Item-Cosine / Item-Jaccard** → sin centrado (usan ratings_ui.csv)

---

## 4.4 Artefactos y rutas

```
artifacts/
├─ clean_report.txt                 # diagnóstico completo
├─ clean_filter_report.txt          # detalle del filtro ≥5 ratings
├─ ratings_min5.csv                 # ratings tras el soporte mínimo
├─ index/
│  ├─ user_map.csv                  # userId,uIdx
│  └─ item_map.csv                  # movieId,iIdx
├─ ratings_ui.csv                   # (uIdx,iIdx,rating) ordenado por uIdx
├─ remap_report.txt                 # U, I, NNZ
├─ user_means.csv                   # media por usuario
└─ matrix_user_csr/
   ├─ indptr.bin                    # int64, len=U+1
   ├─ indices.bin                   # int32, len=NNZ
   ├─ data.bin                      # float32, len=NNZ
   ├─ meta.json                     # {U,I,NNZ,dtypes}
   └─ normalize_report.txt          # resumen normalización
```

---

## 4.5 Cómo se consumirá desde la API/UI

- La UI siempre trabaja con **IDs originales** (`userId`, `movieId`, `title`, `genres`).  
- El motor trabaja con **índices internos** (`uIdx`, `iIdx`) usando CSR y medias.
- Endpoints típicos:
  - `GET /api/recommendations?userId=X`  
    1) `userId → uIdx` (con `user_map.csv`)  
    2) Motor usa CSR + similitudes → `[]iIdx`  
    3) `iIdx → movieId` (con `item_map.csv`) + `movies.csv` para metadatos  
    4) Respuesta `{movieId, title, genres, score}`

---

## 4.6 Decisiones y justificación rápida

- **Filtro por soporte ≥ 5**: estabiliza similitudes y reduce coste; elimina 0.22% de filas y 44.59% de ítems con señal insuficiente.  
- **Remapeo a índices contiguos**: necesario para construir matrices y hacer acceso O(1) por fila/columna.  
- **Centrado por usuario + CSR**: estandariza escalas individuales y permite cómputo eficiente para Coseno/Pearson con memoria acotada.

---

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
