# Procesamiento de Lenguaje Natural I - Desafios

<img src="https://github.com/hernancontigiani/ceia_memorias_especializacion/raw/master/Figures/logoFIUBA.jpg" width="300" align="center">

## Universidad de Buenos Aires (UBA) - Facultad de Ingeniería (FIUBA)
### Laboratorio de Sistemas Embebidos
### Carrera de Especialización en Ingeniería Artificial (CEIA)
### Materia: Procesamiento del Lenguaje Natural

**Estudiante:** Mg. Ing. David Canal  

---

## 📋 Descripción General

Este repositorio contiene la implementación de cuatro desafIos prácticos realizados en el marco de la materia Procesamiento de Lenguaje Natural I de la CEIA de la FIUBA. Cada desafio aborda diferentes aspectos fundamentales del procesamiento de lenguaje natural, desde técnicas básicas de vectorización hasta modelos avanzados de redes neuronales recurrentes.

Los desafios fueron desarrollados utilizando Python y diversas librerías especializadas en NLP, incluyendo scikit-learn, TensorFlow/Keras, PyTorch, Gensim y otras herramientas modernas del ecosistema de machine learning.

---

## 🎯 Desafios Implementados

### 📊 Desafio 1: Vectorización y Clasificación de Documentos
**Archivo:** `Desafio 1/Canal_Desafio_1.ipynb`

#### Objetivos
- Implementar técnicas de vectorización de documentos usando TF-IDF
- Analizar similitudes entre documentos mediante similaridad coseno
- Desarrollar modelos de clasificación por prototipos (zero-shot)
- Optimizar modelos Naïve Bayes para clasificación de texto
- Explorar similitudes entre palabras mediante matrices transpuestas

#### Técnicas Implementadas
- **Vectorización TF-IDF** con scikit-learn
- **Similaridad coseno** para análisis de documentos
- **Clasificación por prototipos** (modelo zero-shot)
- **Naïve Bayes** (MultinomialNB y ComplementNB) con optimización de hiperparámetros
- **Análisis de similitud entre palabras** usando matrices término-documento

#### Dataset
- **20newsgroups**: 11,314 documentos de entrenamiento y 7,532 de test
- **20 categorías** temáticas diferentes

#### Resultados Destacados
- **Coherencia promedio**: 68% en análisis de similitud de documentos
- **Mejor modelo**: ComplementNB con bigramas (F1-macro: 0.7015)
- **Análisis semántico**: Identificación exitosa de relaciones entre palabras

---

### 🔤 Desafio 2: Word Embeddings con Word2Vec
**Archivo:** `Desafio 2/Desafio_2_Canal_David.ipynb`

#### Objetivos
- Crear embeddings personalizados usando Word2Vec (Skip-gram)
- Analizar similitudes semánticas entre palabras
- Implementar visualizaciones 2D y 3D con t-SNE
- Realizar clustering con K-means para identificar grupos semánticos

#### Técnicas Implementadas
- **Word2Vec Skip-gram** con Gensim
- **Preprocesamiento** con Keras text_to_word_sequence
- **Visualización** con t-SNE (2D y 3D)
- **Clustering K-means** con análisis de Silhouette Score
- **Análisis semántico** de palabras relacionadas

#### Dataset
- **Canciones de Britney Spears**: 3,848 líneas de letras
- **Vocabulario**: 620 palabras únicas
- **Vectores**: 300 dimensiones

#### Resultados Destacados
- **Vocabulario entrenado**: 620 palabras
- **Convergencia**: Pérdida reducida de 197,458 a 79,613
- **Clustering óptimo**: k=2 clusters con Silhouette Score de 0.0865
- **Temas identificados**: Amor, empoderamiento femenino, música, baile

---

### 🧠 Desafio 3: Modelo de Lenguaje con RNN
**Archivo:** `Desafio 3/Desafio_3_Canal_David.ipynb`

#### Objetivos
- Implementar modelos de lenguaje usando redes neuronales recurrentes
- Comparar arquitecturas: SimpleRNN, LSTM y GRU
- Desarrollar estrategias de generación de texto (Greedy, Beam Search)
- Crear interfaz interactiva con Gradio

#### Técnicas Implementadas
- **Tokenización por caracteres** (vocabulario de 42 caracteres)
- **Arquitecturas RNN**: SimpleRNN, LSTM, GRU
- **Estructura Many-to-Many** para aprendizaje denso
- **Callback personalizado** con métrica de perplejidad
- **Estrategias de generación**:
  - Greedy Search
  - Beam Search Determinístico
  - Beam Search Estocástico con temperatura
- **Interfaz interactiva** con Gradio

#### Dataset
- **El Quijote** de Miguel de Cervantes
- **100,000 caracteres** procesados
- **Vocabulario**: 42 caracteres únicos

#### Resultados Destacados
- **Mejor modelo**: GRU (perplejidad: 4.7408)
- **Eficiencia**: GRU más eficiente que LSTM
- **Convergencia**: Mejora del 42.58% en perplejidad
- **Generación**: Texto coherente con estilo cervantino

---

### 🤖 Desafio 4: QA Bot con LSTM
**Archivo:** `Desafio 4/Desafio_4_Canal_David.ipynb`

#### Objetivos
- Construir un chatbot de preguntas y respuestas usando arquitectura Seq2Seq
- Implementar encoder-decoder con LSTM
- Utilizar embeddings FastText españoles
- Evaluar el desempeño con preguntas específicas

#### Técnicas Implementadas
- **Arquitectura Seq2Seq** con encoder-decoder
- **LSTM** con dropout (0.2) y 128 unidades
- **Embeddings FastText** españoles (300 dimensiones)
- **Tokenización** con tokens especiales (<sos>, <eos>)
- **Entrenamiento** con early stopping
- **Función de generación** simplificada pero funcional

#### Configuración Técnica
- **MAX_VOCAB_SIZE**: 8,000
- **MAX_INPUT_LEN**: 16 caracteres
- **MAX_OUT_LEN**: 18 caracteres
- **EMBEDDING_DIM**: 300 (FastText español)
- **LSTM_UNITS**: 128
- **DROPOUT**: 0.2
- **EPOCHS**: 40 (dentro del rango 30-50)

#### Dataset
- **8,000 pares QA** expandidos con variaciones
- **Preguntas sugeridas** incluidas:
  - "Do you read?"
  - "Do you have any pet?"
  - "Where are you from?"

#### Resultados Destacados
- **Accuracy final**: 100% en entrenamiento y validación
- **Loss final**: 0.019 (entrenamiento), 0.018 (validación)
- **Respuestas coherentes** para todas las preguntas de prueba
- **Modelo funcional** con 571,540 parámetros

---

## 🛠️ Tecnologías Utilizadas

### Librerías Principales
- **scikit-learn**: Vectorización, clasificación, métricas
- **TensorFlow/Keras**: Redes neuronales recurrentes
- **PyTorch**: Implementación del QA Bot
- **Gensim**: Word2Vec y embeddings
- **NumPy/Pandas**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualizaciones
- **Gradio**: Interfaz interactiva

### Herramientas de Procesamiento
- **NLTK/spaCy**: Preprocesamiento de texto
- **BeautifulSoup**: Parsing de HTML
- **scikit-learn**: Tokenización y padding
- **t-SNE**: Reducción dimensional
- **K-means**: Clustering

---

## 📈 Métricas y Resultados

### Desafio 1: Clasificación
- **F1-Score macro**: 0.7015 (ComplementNB con bigramas)
- **Coherencia similitud**: 68% promedio
- **Accuracy**: 48.6% (modelo prototipos)

### Desafio 2: Word Embeddings
- **Vocabulario**: 620 palabras
- **Convergencia**: 20 épocas
- **Silhouette Score**: 0.0865 (k=2)

### Desafio 3: Modelo de Lenguaje
- **Perplejidad**: 4.7408 (GRU)
- **Accuracy**: 51.01% (GRU)
- **Tiempo entrenamiento**: 1,141 segundos

### Desafio 4: QA Bot
- **Accuracy**: 100% (entrenamiento y validación)
- **Loss**: 0.018 (validación)
- **Parámetros**: 571,540

---

## 🎓 Aprendizajes y Conclusiones

### Técnicas Fundamentales
1. **Vectorización**: TF-IDF como base sólida para representación de documentos
2. **Embeddings**: Word2Vec para capturar relaciones semánticas
3. **RNN**: LSTM y GRU para modelado de secuencias
4. **Seq2Seq**: Arquitectura encoder-decoder para tareas conversacionales

### Insights Técnicos
- **ComplementNB** supera a MultinomialNB en datasets desbalanceados
- **GRU** ofrece mejor balance eficiencia-rendimiento que LSTM
- **Tokenización por caracteres** permite mayor flexibilidad
- **Beam Search** mejora la calidad de generación vs Greedy Search

### Aplicaciones Prácticas
- **Sistemas de recomendación** basados en contenido
- **Análisis de sentimientos** y clasificación de texto
- **Generación automática** de contenido
- **Chatbots conversacionales** para atención al cliente

---

## 📁 Estructura del Repositorio

```
Desafios_Canal/
├── README.md                           # Este archivo
├── Desafio 1/
│   └── Canal_Desafio_1.ipynb          # Vectorización y clasificación
├── Desafio 2/
│   └── Desafio_2_Canal_David.ipynb    # Word embeddings
├── Desafio 3/
│   └── Desafio_3_Canal_David.ipynb    # Modelo de lenguaje RNN
└── Desafio 4/
    ├── Desafio_4_Canal_David.ipynb    # QA Bot
    ├── qa_bot_canal_david.csv          # Evaluación del bot
    └── torch_helpers.py                # Utilidades PyTorch
```

---

## 🚀 Cómo Ejecutar los desafios

### Requisitos Previos
```bash
pip install numpy pandas matplotlib seaborn
pip install scikit-learn tensorflow torch
pip install gensim nltk gradio
pip install beautifulsoup4 requests
```

### Ejecución
1. **desafio 1**: Abrir `Canal_Desafio_1.ipynb` y ejecutar todas las celdas
2. **desafio 2**: Abrir `Desafio_2_Canal_David.ipynb` y ejecutar secuencialmente
3. **desafio 3**: Abrir `Desafio_3_Canal_David.ipynb` y ejecutar (incluye interfaz Gradio)
4. **desafio 4**: Abrir `Desafio_4_Canal_David.ipynb` y ejecutar para entrenar el QA Bot

---

## 📚 Referencias y Material de Clase

Este trabajo se realizó utilizando el material de clases y ejercicios proporcionados en el repositorio oficial de la materia:

**Contenido del Curso:**
- Clase 1: Introducción a NLP y vectorización de documentos
- Clase 2: Preprocesamiento de texto e information-retrieval bots
- Clase 3: Word embeddings, CBOW y SkipGRAM
- Clase 4: Redes recurrentes (RNN) y problemas de secuencia
- Clase 5: Redes LSTM y análisis de sentimiento
- Clase 6: Modelos Seq2Seq, bots conversacionales y traductores
- Clase 7: Celdas con Attention, Transformers, BERT y ELMo
- Clase 8: Deployment de servicios NLP, Flask, APIs, Docker y Tensorflow Serving
