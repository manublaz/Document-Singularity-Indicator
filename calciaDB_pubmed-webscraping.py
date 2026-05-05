#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PUBMED CENTRAL CORPUS BUILDER - Versión para PMC
Extrae artículos de PubMed Central y los almacena en base de datos CALCIA
Prof. Manuel Blázquez Ochando - Universidad Complutense de Madrid
"""

import os
import time
import requests
import sqlite3
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode, urljoin, quote_plus
import re
import random
from datetime import datetime
import logging

# Configuración
DB_PATH = "calcia.db"
MAX_RETRIES = 3
REQUEST_DELAY = (2, 5)  # Rango de segundos entre requests
PMC_BASE_URL = "https://pmc.ncbi.nlm.nih.gov"
PUBMED_BASE_URL = "https://www.ncbi.nlm.nih.gov/pubmed"

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pubmed_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PubMedCorpusBuilder:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        })
        self.processed_urls = set()
        self.successful_extractions = 0
        self.failed_extractions = 0
        
        # Verificar que existe la base de datos
        self.verify_database()
    
    def verify_database(self):
        """Verifica que existe la base de datos y tabla corpus"""
        if not os.path.exists(self.db_path):
            logger.error(f"Base de datos {self.db_path} no existe. Ejecute createCalciaDB.py primero.")
            raise FileNotFoundError(f"Base de datos {self.db_path} no encontrada")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='corpus';")
            if not cursor.fetchone():
                logger.error("Tabla 'corpus' no existe en la base de datos")
                raise Exception("Tabla corpus no encontrada")
            conn.close()
            logger.info("Base de datos verificada correctamente")
        except Exception as e:
            logger.error(f"Error verificando base de datos: {e}")
            raise
    
    def human_delay(self):
        """Pausa humana entre requests"""
        delay = random.uniform(*REQUEST_DELAY)
        time.sleep(delay)
    
    def safe_request(self, url, max_retries=MAX_RETRIES):
        """Request con reintentos y manejo de errores"""
        for attempt in range(max_retries):
            try:
                self.human_delay()
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    logger.warning(f"Rate limit (429) en intento {attempt + 1}. Esperando...")
                    time.sleep(30)
                    continue
                else:
                    logger.warning(f"HTTP {response.status_code} en {url}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error en request (intento {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)
                    continue
                
        return None
    
    def extract_snippet_data(self, snippet_div):
        """
        Extrae datos básicos de un snippet de resultado
        
        Args:
            snippet_div: Elemento BeautifulSoup del snippet
            
        Returns:
            dict con url, title, author, datepub o None si falta información
        """
        try:
            # Extraer URL y título
            title_link = snippet_div.select_one("a.docsum-title")
            if not title_link:
                logger.warning("No se encontró enlace de título en snippet")
                return None
            
            url = title_link.get("href")
            if not url.startswith("http"):
                url = PMC_BASE_URL + url
            
            # Limpiar título (remover etiquetas HTML como <b>)
            title_soup = BeautifulSoup(title_link.get_text(), 'html.parser')
            title = title_soup.get_text(strip=True)
            
            # Extraer autores
            author_spans = snippet_div.select("ul.usa-list span")
            authors = []
            for span in author_spans:
                text = span.get_text(strip=True)
                # Si contiene punto y coma o comas, probablemente son autores
                if ',' in text and not any(x in text.lower() for x in ['pmcid', 'doi', 'med']):
                    authors.append(text)
                    break  # Tomar solo la primera línea de autores
            
            author = authors[0] if authors else ""
            
            # Extraer año de publicación
            datepub = ""
            citation_parts = snippet_div.select(".citation-part")
            for part in citation_parts:
                text = part.get_text(strip=True)
                # Buscar patrón de año (4 dígitos)
                year_match = re.search(r'\b(19|20)\d{2}\b', text)
                if year_match:
                    datepub = year_match.group()
                    break
            
            # Verificar que tenemos toda la información necesaria
            if not all([url, title, author, datepub]):
                logger.warning(f"Snippet incompleto: url={bool(url)}, title={bool(title)}, author={bool(author)}, datepub={bool(datepub)}")
                return None
            
            return {
                'url': url,
                'title': title,
                'author': author,
                'datepub': datepub
            }
            
        except Exception as e:
            logger.error(f"Error extrayendo datos de snippet: {e}")
            return None
    
    def extract_full_text(self, article_url):
        """
        Extrae el texto completo del artículo desde PMC
        
        Args:
            article_url: URL del artículo en PMC
            
        Returns:
            str con texto limpio o None si hay error
        """
        logger.info(f"Extrayendo texto completo de: {article_url}")
        
        response = self.safe_request(article_url)
        if not response:
            logger.error(f"No se pudo acceder a {article_url}")
            return None
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Buscar la sección principal del artículo
        main_body = soup.select_one("section.body.main-article-body")
        if not main_body:
            logger.warning(f"No se encontró sección 'body main-article-body' en {article_url}")
            return None
        
        # Extraer texto limpio
        text = main_body.get_text(separator=' ', strip=True)
        
        # Limpiar texto
        text = re.sub(r'\s+', ' ', text)  # Múltiples espacios
        text = re.sub(r'\n+', '\n', text)  # Múltiples saltos de línea
        text = text.strip()
        
        if len(text) < 100:  # Filtro de calidad mínimo
            logger.warning(f"Texto muy corto ({len(text)} caracteres) en {article_url}")
            return None
        
        logger.info(f"Texto extraído: {len(text)} caracteres")
        return text
    
    def extract_pmid_from_article(self, article_url):
        """
        Extrae el PMID del artículo para buscar citaciones
        
        Args:
            article_url: URL del artículo en PMC
            
        Returns:
            str con PMID o None si no se encuentra
        """
        response = self.safe_request(article_url)
        if not response:
            return None
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Buscar el PMID en la sección principal
        main_body = soup.select_one("section.body.main-article-body")
        if not main_body:
            logger.warning(f"No se encontró sección principal para extraer PMID de {article_url}")
            return None
        
        # Buscar enlace con PMID
        pmid_links = main_body.select("a.usa-link[href*='pubmed.ncbi.nlm.nih.gov']")
        for link in pmid_links:
            href = link.get("href", "")
            # Extraer número del enlace
            pmid_match = re.search(r'/(\d+)/?$', href)
            if pmid_match:
                pmid = pmid_match.group(1)
                logger.info(f"PMID encontrado: {pmid}")
                return pmid
        
        # Búsqueda alternativa en el texto
        pmid_text_match = re.search(r'PMID:\s*(\d+)', main_body.get_text())
        if pmid_text_match:
            pmid = pmid_text_match.group(1)
            logger.info(f"PMID encontrado en texto: {pmid}")
            return pmid
        
        logger.warning(f"No se encontró PMID en {article_url}")
        return None
    
    def get_citation_count(self, pmid):
        """
        Obtiene el número de citaciones para un PMID
        
        Args:
            pmid: ID de PubMed
            
        Returns:
            int con número de citaciones o 0 si hay error
        """
        citations_url = f"{PUBMED_BASE_URL}/?linkname=pubmed_pubmed_citedin&from_uid={pmid}"
        logger.info(f"Buscando citaciones en: {citations_url}")
        
        response = self.safe_request(citations_url)
        if not response:
            logger.error(f"No se pudo acceder a página de citaciones para PMID {pmid}")
            return 0
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Buscar el div con los resultados
        results_div = soup.select_one("div.results-amount h3 span.value")
        if results_div:
            try:
                # Remover comas del texto antes de convertir a entero
                citations_text = results_div.get_text(strip=True).replace(',', '')
                citations = int(citations_text)
                logger.info(f"Citaciones encontradas: {citations}")
                return citations
            except ValueError:
                logger.warning(f"No se pudo convertir citaciones a número: {results_div.get_text()}")
                return 0
        
        logger.warning(f"No se encontró información de citaciones para PMID {pmid}")
        return 0
    
    def is_duplicate_in_database(self, url):
        """
        Verifica si la URL ya existe en la base de datos
        
        Args:
            url: URL a verificar
            
        Returns:
            bool: True si existe, False si no
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM corpus WHERE url = ?", (url,))
            exists = cursor.fetchone() is not None
            conn.close()
            return exists
        except Exception as e:
            logger.error(f"Error verificando duplicado: {e}")
            return False
    
    def save_to_database(self, metadata, full_text, citations):
        """
        Guarda los datos completos en la base de datos CALCIA
        
        Args:
            metadata: dict con url, title, author, datepub
            full_text: str con texto completo
            citations: int con número de citaciones
            
        Returns:
            bool: True si se guardó exitosamente
        """
        try:
            # Verificar duplicados
            if self.is_duplicate_in_database(metadata['url']):
                logger.info(f"Artículo ya existe en BD: {metadata['title'][:50]}...")
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insertar nuevo registro
            insert_sql = """
            INSERT INTO corpus (url, title, author, datepub, text, textoriginal, citations)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(insert_sql, (
                metadata['url'],
                metadata['title'],
                metadata['author'],
                metadata['datepub'],
                full_text,
                full_text,
                citations
            ))
            
            conn.commit()
            doc_id = cursor.lastrowid
            conn.close()
            
            logger.info(f"Artículo guardado en BD con ID {doc_id}: {metadata['title'][:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando en BD: {e}")
            return False
    
    def process_single_article(self, snippet_data):
        """
        Procesa un artículo completo: metadata + texto + citaciones + BD
        
        Args:
            snippet_data: dict con url, title, author, datepub
            
        Returns:
            bool: True si se procesó exitosamente
        """
        url = snippet_data['url']
        
        if url in self.processed_urls:
            logger.info(f"Ya procesado: {url}")
            return False
        
        self.processed_urls.add(url)
        
        # 1. Extraer texto completo
        full_text = self.extract_full_text(url)
        if not full_text:
            logger.error(f"No se pudo extraer texto completo de {url}")
            self.failed_extractions += 1
            return False
        
        # 2. Extraer PMID
        pmid = self.extract_pmid_from_article(url)
        if not pmid:
            logger.error(f"No se pudo extraer PMID de {url}")
            self.failed_extractions += 1
            return False
        
        # 3. Obtener citaciones
        citations = self.get_citation_count(pmid)
        
        # 4. Guardar en base de datos
        success = self.save_to_database(snippet_data, full_text, citations)
        
        if success:
            self.successful_extractions += 1
            logger.info(f"[OK] Procesado exitosamente: {snippet_data['title'][:50]}... ({len(full_text)} chars, {citations} citas)")
        else:
            self.failed_extractions += 1
        
        return success
    
    def get_search_results_from_page(self, encoded_term, page_num):
        """
        Obtiene los snippets de una página específica
        
        Args:
            encoded_term: término de búsqueda codificado
            page_num: número de página (1, 2, 3, etc.)
            
        Returns:
            list de elementos div.docsum-wrap o lista vacía si hay error
        """
        if page_num == 1:
            page_url = f"{PMC_BASE_URL}/search/?term={encoded_term}"
        else:
            page_url = f"{PMC_BASE_URL}/search/?term={encoded_term}&page={page_num}"
        
        logger.info(f"Cargando página {page_num}: {page_url}")
        
        response = self.safe_request(page_url)
        if not response:
            logger.warning(f"No se pudo cargar página {page_num}")
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        snippets = soup.select("div.docsum-wrap")
        
        logger.info(f"Página {page_num}: {len(snippets)} resultados")
        return snippets
    
    def extract_search_results(self, search_term, max_pages=1):
        """
        Extrae resultados de búsqueda de PubMed Central
        
        Args:
            search_term: término de búsqueda
            max_pages: número máximo de páginas a procesar
            
        Returns:
            list de diccionarios con datos de artículos
        """
        # Codificar el término de búsqueda para URL
        encoded_term = quote_plus(search_term)
        logger.info(f"Iniciando búsqueda para: '{search_term}' en {max_pages} páginas")
        
        all_snippets = []
        
        # Obtener snippets de cada página
        for page_num in range(1, max_pages + 1):
            page_snippets = self.get_search_results_from_page(encoded_term, page_num)
            
            if not page_snippets:
                logger.info(f"Página {page_num} sin resultados. Terminando búsqueda.")
                break
            
            all_snippets.extend(page_snippets)
            
            # Pequeña pausa entre páginas
            if page_num < max_pages:
                time.sleep(1)
        
        logger.info(f"Total de snippets obtenidos: {len(all_snippets)} de {max_pages} páginas")
        
        # Procesar cada snippet para extraer datos
        articles_data = []
        for i, snippet in enumerate(all_snippets, 1):
            logger.info(f"Procesando snippet {i}/{len(all_snippets)}")
            
            snippet_data = self.extract_snippet_data(snippet)
            if snippet_data:
                articles_data.append(snippet_data)
                logger.info(f"Snippet válido: {snippet_data['title'][:50]}...")
            else:
                logger.warning(f"Snippet {i} inválido o incompleto")
        
        logger.info(f"Extracción completa: {len(articles_data)} artículos válidos de {len(all_snippets)} snippets")
        return articles_data
    
    def build_corpus_from_search(self, search_term, max_pages=1):
        """
        Construye corpus completo desde búsqueda en PubMed Central
        
        Args:
            search_term: término de búsqueda
            max_pages: número de páginas a procesar
        """
        logger.info("="*70)
        logger.info("PUBMED CENTRAL CORPUS BUILDER - Iniciando extracción")
        logger.info(f"Término: '{search_term}' | Páginas: {max_pages}")
        logger.info("="*70)
        
        # 1. Extraer datos de búsqueda
        articles_data = self.extract_search_results(search_term, max_pages)
        if not articles_data:
            logger.error("No se encontraron artículos válidos")
            return
        
        logger.info(f"Procesando {len(articles_data)} artículos...")
        
        # 2. Procesar cada artículo
        for i, article_data in enumerate(articles_data, 1):
            logger.info(f"\n[{i}/{len(articles_data)}] Procesando artículo completo...")
            logger.info(f"Título: {article_data['title'][:60]}...")
            
            self.process_single_article(article_data)
            
            # Progreso cada 10 artículos
            if i % 10 == 0:
                logger.info(f"Progreso: {i}/{len(articles_data)} - Exitosos: {self.successful_extractions}, Fallidos: {self.failed_extractions}")
        
        # 3. Resumen final
        total_processed = len(articles_data)
        success_rate = (self.successful_extractions / total_processed * 100) if total_processed > 0 else 0
        
        logger.info("="*70)
        logger.info("RESUMEN FINAL")
        logger.info(f"Término de búsqueda: '{search_term}'")
        logger.info(f"Páginas procesadas: {max_pages}")
        logger.info(f"Total artículos encontrados: {total_processed}")
        logger.info(f"Guardados exitosamente: {self.successful_extractions}")
        logger.info(f"Fallidos: {self.failed_extractions}")
        logger.info(f"Tasa de éxito: {success_rate:.1f}%")
        logger.info("="*70)

def main():
    """Función principal"""
    print("PUBMED CENTRAL CORPUS BUILDER")
    print("Prof. Manuel Blázquez Ochando - UCM")
    print("="*60)
    
    # Solicitar término de búsqueda
    while True:
        search_term = input("\nTérmino de búsqueda: ").strip()
        if search_term:
            break
        print("(!) Debe introducir un término de búsqueda válido")
    
    # Solicitar número de páginas
    while True:
        try:
            max_pages = int(input("Número de páginas a procesar [1]: ") or "1")
            if max_pages > 0:
                break
            print("(!) Debe ser un número positivo")
        except ValueError:
            print("(!) Debe introducir un número válido")
    
    # Confirmar configuración
    print(f"\n[CONFIGURACION]:")
    print(f"   Término: '{search_term}'")
    print(f"   Páginas: {max_pages}")
    print(f"   Base de datos: {DB_PATH}")
    
    confirm = input("\n¿Continuar? (s/n): ").lower()
    if confirm != 's':
        print("Operación cancelada.")
        return
    
    # Inicializar y ejecutar
    try:
        builder = PubMedCorpusBuilder()
        builder.build_corpus_from_search(search_term, max_pages)
        
        print(f"\n[OK] Proceso completado. Revise el archivo 'pubmed_extraction.log' para detalles.")
        print(f"[INFO] Los datos están guardados en: {DB_PATH}")
        
    except Exception as e:
        print(f"[ERROR] Error durante la ejecución: {e}")
        logger.error(f"Error fatal: {e}")

if __name__ == "__main__":
    main()
