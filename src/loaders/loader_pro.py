import os
import hashlib
from pdfminer.high_level import extract_text
from docx import Document  
import json
class DocumentIngestionPipeline:
    """
    Clase que puede procesar distintos tipos de archivos (pdf, docx y txt) y conviertiendolos a texto. 
    
    Utiliza un hash HD5 para detectar cambios en los archivos y reprocesare solo los archivos nuevos o que
    hayan sufrido modificaciones.

    La informacion extraida se guarda en un archivo json.
    """
    def __init__(self, directory_path):
        """
        Inicializa el pipeline de ingesta de documentos.
        Args:
            directory_path (str): Ruta a la carpeta donde se encuentran los archivos.
        Atributos:
            self.directory_path (str)      : guarda la ruta donde se encuentran los archivos.
            self.processed_files (dict)    : diccionario que almacena el hash de los archivos.
            self.documents (list)          : lista que guarda los documentos procesados con sus metadatos. 
        """
        self.directory_path = directory_path
        self.processed_files = {}
        self.documents = []

    def hash_file(self, filepath):
        """
        La funcion genera un hash HD5 para el archivo para detectar cambios en  los documentos.
        Args:
            filepath (str): Ruta del archivo para generar el hash.
        Returns:
            str: Hash MD5 del archivo.
        """
        with open(filepath, 'rb') as file:
            return hashlib.md5(file.read()).hexdigest()
        
    def extract_text_from_pdf(self, filepath):
        """
        Extrae texto de un archivo PDF.
        Args:
            filepath (str): Ruta del archivo PDF.
        Returns:
            str: Texto extraído del PDF.
        """

        return extract_text(filepath)

    def extract_text_from_docx(self, filepath):
        """
        Extrae texto de un archivo DOCX.
        Args:
            filepath (str): Ruta del archivo DOCX.
        Returns:
            str: Texto extraído del documento DOCX.
        """
        doc = Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs])

    def extract_text_from_txt(self, filepath):
        """
        Extrae texto de un archivo TXT.
        Args:
            filepath (str): Ruta del archivo TXT.
        Returns:
            str: Texto extraído del archivo TXT.
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
        
    def load_documents(self):
        """
        La funcion procesa todos los archivos en la ruta y extrae y almacen la informacion
        en una lista.

        Flujo:
            1.- Itera sobre los archivos especificados en la ruta
            2.- Genera el hash para cada archivo con hash_file()
            3.- Comprueba si han habido cambios en el archivo con el hash actual con processed_files
                Si el archivo ha cambiado:
                    3.1 - Identifica el tipo de extension del archivo
                    3.2 - Se utiliza el metodo según extension del archivo
                    3.3 - Se guarda la informacion contenida en self.documents como un dict{}
            4.- Manejo de excepciones y reporta error.
            5.- Retorna una lista con la informacion contenida en los archivos de la ruta.


        Returns:
            list: Lista de documentos procesados con metadatos.
        """
        for filename in os.listdir(self.directory_path):
            filepath = os.path.join(self.directory_path, filename)
            file_hash = self.hash_file(filepath)

            # Procesar si es nuevo o ha cambiado
            if self.processed_files.get(filename) != file_hash:
                try:
                    # Detectar el tipo de archivo por extensión
                    if filename.endswith('.pdf'):
                        text = self.extract_text_from_pdf(filepath)
                    elif filename.endswith('.docx'):
                        text = self.extract_text_from_docx(filepath)
                    elif filename.endswith('.txt'):
                        text = self.extract_text_from_txt(filepath)
                    else:
                        print(f"Tipo de archivo no soportado: {filename}")
                        continue

                    # Guardar el documento procesado
                    self.documents.append({
                        'file_name': filename,
                        'file_type': filename.split('.')[-1],
                        'text': text,
                        'hash': file_hash
                    })
                    self.processed_files[filename] = file_hash
                except Exception as e:
                    print(f"Error al procesar {filename}: {e}")

        return self.documents
    
    def save_to_json(self, output_file="documents_text.json"):
        """
        La funcion guarda los documentos procesados en un archivo json.

        Args:
            output_file (str): Nombre del archivo de salida.
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=4)

        print(f"Documentos guardados en {output_file}")

    def load_pipeline(self):
        """
        Ejecuta todo el pipeline de carga, conversión y guardado de documentos.
        """
        self.load_documents()
        self.save_to_json()
