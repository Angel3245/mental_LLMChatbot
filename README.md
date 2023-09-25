Aplicación de técnicas de aprendizaje profundo al desarrollo de un bot conversacional para contribuír a la mejora de la salud mental
=======

Proyecto desarrollado durante el Trabajo de Fin de Grado (TFG) en el [Grado en Ingeniería Informática](https://esei.uvigo.es/es/estudos/grao-en-enxenaria-informatica/) de la Universidad de Vigo por el alumno Jose Ángel Pérez Garrido y tutorizado por Dra. Anália Maria Garcia Lourenço en el curso académico 2022-2023. 

Esta aplicación proporciona las herramientas necesarias para la creación y evaluación de un bot conversacional mediante modelos LLM.


# Requisitos
1. **S.O.** Windows 10/Ubuntu 20.04 o superiores
2. **Intérprete Python 3.** Probado en la versión [3.9](https://www.python.org/downloads/release/python-390/).
3. **Base de Datos MySQL.** Probado en la versión [10.4.25-MariaDB](https://mariadb.com/kb/en/mariadb-10425-release-notes/).


# Instalación del entorno
Deben seguirse los siguientes pasos para instalar las dependencias necesarias para la ejecución de la aplicación.

# Instalación en el sistema local
**Solo recomendable en sistemas Ubuntu.** 
1. Desplazarse hasta el directorio del proyecto
```
cd (`ruta_del_proyecto`)
```

2. Crear un entorno virtual (de nombre venv)
```
python3 –m venv venv
```

3. Activar el entorno virtual

```
source venv/bin/activate
```

4. Instalar las dependencias incluídas en el archivo requirements.txt
```
pip install –r requirements.txt
```

# Instalación en Docker
**Se necesita tener [Docker](https://www.docker.com/) instalado en el sistema y en ejecución.**

1. Desplazarse hasta el directorio del proyecto
```
cd <ruta_do_proxecto>
```

2. Crear la imagen Docker a partir del archivo Dockerfile (de nombre ```mentalbot```)
```
docker build -t mentalbot .
```

3. Crear un contenedor Docker a partir de la imagen ```mentalbot``` y proporcionar acceso a GPU.  (Opcional) -v: Agrega un directorio compartido entre la máquina anfitriona y el contenedor Docker
```
docker run -it -p 8000:8000 --gpus=all \
[-v <directorio_en_local>:/<directorio_en_docker>/] mentalbot
```

4. Acceder a un terminal del contenedor Docker
```
docker exec -it <container_id> /bin/bash
```

# Uso de la aplicación
Cuando finalice la instalación ya se pueden utilizar los distintos scripts que se encuentran dentro del directorio app y se ejecutan desde la consola de comandos situada na raíz da carpeta do proxecto. El formato del comando sería el siguiente:

```
python3 app/(`nombre_script`).py –o option (`–a args`)
```

Cada script posée un manual propio que se puede consultar con la opción _--help_. Por ejemplo, para _chatbot.py_ el formato del comando sería:

```
python3 app/chatbot.py --help
```


# Scripts
Los scripts disponibles según el tipo de usuario son los seguintes:

## Scripts de usuario

* ```chatbot.py```: contiene la lógica necesaria para la ejecución del bot conversacional. Carga un modelo de _/file/chatbot_model_.

## Scripts de desarrollador

* ```evaluation.py```: contiene la lógica necesaria para evaluar la eficiencia de los modelos.
* ```prepare_dataset.py```: contiene la lógica necesaria para realizar la limpieza, transformación, construcción y adaptación de los datos para el caso de uso.
* ```reddit_scripts.py```: contiene la lógica necesaria para extraer datos de Reddit.
* ```test_cuda.py```: contiene la lógica para la comprobación de que CUDA está disponible y configurado correctamente en el equipo.
* ```training.py```: contiene la lógica para entrenar los modelos puntuadores.

# Pasos seguidos para la creación y evaluación de un bot conversacional
1. Obtener datos de fuentes externas (Reddit) mediante ```reddit_scripts.py``` (ya creado en _/file/datasets_).

**Para poder ejecutar el script se necesitan realizar unos pasos previos de configuración descritos a continuación:**

    - Crear una base de datos MySQL de nombre ```reddit``` e importar el archivo de base de datos disponible en ```/file/reddit/reddit.sql```

    - Crear un usuario de nombre redditUser y con contraseña redditPass con acceso a 
    esta base de datos
    ```
    GRANT ALL PRIVILEGES ON reddit.* TO redditUser@localhost IDENTIFIED BY redditPass;
    ```

    - Crear y cubrir los datos del archivo ```/file/reddit/client_secrets.json``` siguiendo el 
    modelo de ```/file/reddit/client_secrets_example.json```. ([Más información](https://praw.readthedocs.io/en/stable/getting_started/authentication.html#passwor))

    - Obtener el valor de refresh_token con el siguiente script y añadirlo a ```/file/reddit/client_secrets.json```:
    ```
    python app/reddit_scripts.py -o refresh_token
    ```

Una vez esté configurado el modulo ya es posible descargar información de Reddit. Por ejemplo, para descargar publicaciones del subreddit ```r/Anxiety``` que contengan alguno de los siguientes flairs ```["Advice Needed","Needs a Hug/Support"]``` el comando sería el siguiente:
```
python app\reddit_scripts.py -o extraction_search_by_flair -s Anxiety -d reddit -f "Advice Needed","Needs a Hug/Support"
```


2. Crear el corpus (MentalKnowledge) parseando los conjuntos de datos obtenidos de fuentes externas mediante ```prepare_dataset.py``` (ya creado en _/file/data_)

```
python app/prepare_dataset.py -o parsing_text_generation -d MentalKnowledge
```


3. Entrenar el modelo seleccionado con los mejores hiperparámetros mediante ```training.py```

```
# Obtener los mejores hiperparametros para el entrenamiento del modelo seleccionado
python app/training.py -o hyperparameter_search -m gpt2

# Entrenar el modelo
python app/training.py -o finetune_model -m gpt2
```

4. Generar los resultados con los casos de prueba, calcular las métricas y guardar la información en un archivo CSV mediante ```evaluate.py```

```
python app/evaluation.py -o evaluate -m gpt2
```

# Estructura del proyecto
El proyecto se organiza de la siguinte manera:

*	_/app_: directorio que almacena los scripts y todo el código de la aplicación organizado en diferentes subdirectorios:
    *   _/view_: contiene las funciones necesarias para devolver una respuesta al usuario a partir de una entrada empleando el sistema de búsqueda semántica.
    *	_/clean_data_: contiene las funciones necesarias para realizar unha limpeza de los datos.
    *	_/database_: contiene las funciones necesarias para realizar unha conexión con la Base de Datos.
    *	_/extraction_: contiene las funciones necesarias para conectarse a la API de Reddit y descargar los datos.
    *	_/model_: contiene las entidades para la comunicación con la Base de Datos mediante SQLAlchemy o ElasticSearch.
    *	_/parsers_: contiene las clases para analizar, limpiar y extraer datos de diferentes fuentes de información.
    *	_/shared_: contiene las funciones comunes empleadas por diferentes módulos, como la lectura o escritura de ficheros.
    *	_/text_generation_: contiene las funciones necesarias para entrenar, validar y probar los diferentes modelos de generación de texto.
    *	_/transformation_: contiene las funciones necesarias para preparar los conjuntos de datos.
    *	_/view_: contiene las funciones relacionadas con la vista que se presenta al usuario de la aplicación.
*	_/file_: directorio que almacena información requerida para la ejecución de la aplicación organizada en diferentes subdirectorios:
    *	_/chatbot_model_: directorio que almacena los pesos del modelo puntuador (SBERT) seleccionado para el bot conversacional.
    *	_/data_: contiene los conjuntos de datos preparados mediante la aplicación.
    *	_/datasets_: contiene los conjuntos de datos extraídos de diferentes fuentes de datos.
    *	_/evaluation_: contiene los resultados de la evaluación de los diferentes modelos entrenados.
    *	_/reddit_: contiene la información requerida para configurar y emplear el módulo de extracción da información de Reddit.
    *	_/templates_: contiene los modelos de instrucción empleados para la creación de las instancias de instrución.
    *	_/test_: contiene los casos de prueba para la evaluación de los modelos.
*	_/output_: directorio donde se almacenan los modelos puntuadores entrenados mediante la aplicación junto con los resultados de su evaluación.