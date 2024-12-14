import os
import json
import hashlib
from pdfminer.high_level import extract_text

class PDFIngestionPipeline:

    """
    Clase PDFIngestionPipeline para cargar y convertir de archivos PDF en texto,
      detectando si un archivo es nuevo o ha cambiaod desde la ultima ejecucion del codigo
      por medio de un hash MD5.
    """

    def __init__(self, directory_path):
        """
        Inicializa el pipeline de ingesta de PDF.
            Args:
                directory_path (str): Ruta a la carpeta donde se encuentran los archivos PDF
                processed_files     : Diccionario que guarda los nombres de los archivos y sus hash
                documents           : Lista que almacena los docuemntos cargados
        """

        self.directory_path = directory_path
        self.processed_files = {}
        self.documents = []



    def hash_file(self, filepath):
        """
        Esta función genera un hash MD5 de un archivo pdf.
            Args:
                filepath (str): Ruta del archivo para generar el hash.
            Returns:
                str: Cadena hash HD5 que representa el hash del archivo.
        """

        with open(filepath, 'rb') as file:
            return hashlib.md5(file.read()).hexdigest()
        


    def load_pdfs(self):
        """
        La funcion load_pdf carga y procesa los pdf, extrayendo la informacion si son nuevos o si
        se ha realizado un cambio.

        Flujo: 
            1.- Itera sobre los archivos de la carpeta donde estan contenidos los pdf's.
            2.- Filtra solo los archivos que terminan en .pdf (para el proyecto solo se pide 
                    carga de pdf, es lo minimo. Se trabajará sobre un modificacion de este loader para que 
                    pueda procesar otro tipo de documento, .txt, .docx, .pdf).
            3.- Genera un hash_file para cada archivo.
            4.- Compara el hash actual con el almacenado en processed_files. Si el archivo es nuevo o 
            o ha sufrido modificaciones se procesa. Si no ha sufrido modificaciones se omite.
            5.- Se utiliza extract_documents para extraer el texto del archivo pdf.
            6.- Se guardan los datos extraidos en la lista self.documents como un dict con el sigueinte detalle:
                    {nombre del archivo, pais, texto del pdf, hash}


            Returns:
                list: Lista de documentos procesados con metadatos.
        """

        for filename in os.listdir(self.directory_path):
            if filename.endswith('.pdf'):
                filepath = os.path.join(self.directory_path, filename)
                file_hash = self.hash_file(filepath)

                # Procesar si es nuevo o ha cambiado
                if self.processed_files.get(filename) != file_hash:
                    try:
                        text = extract_text(filepath)
                        self.documents.append({
                            'file_name': filename,
                            'country': filename.split('.')[0],
                            'text': text,
                            'hash': file_hash
                        })
                        self.processed_files[filename] = file_hash
                    except Exception as e:
                        print(f"Error al procesar {filename}: {e}")

        return self.documents



    def save_to_json(self, output_file="documents_text.json"):

        """
        Funcion que almacena los documentos procesados en un archivo formato json con codificacion
            UTF-8.

        Args:
            output_file (str): Nombre del archivo de salida que contiene la informacion de los pdf's
        """

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=4)

        print(f"Documentos guardados en {output_file}")



    def load_pipeline(self):
        """
        Funcion que ejecuta todo el pipeline de carga, conversión y guardado de PDF.
        """
        self.load_pdfs()
        self.save_to_json()