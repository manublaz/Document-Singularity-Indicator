import sqlite3
import os
from datetime import datetime

def create_calcia_database(db_path: str = "calcia.db"):
    """
    Crea la base de datos CALCIA con la tabla corpus
    
    Tabla corpus con campos:
    - id: Identificador único (INTEGER PRIMARY KEY)
    - url: URL del documento (TEXT)
    - title: Título del documento (TEXT)
    - author: Autor del documento (TEXT) 
    - datepub: Fecha de publicación (TEXT)
    - text: Texto completo del documento (TEXT)
    - citations: Número de citas (INTEGER)
    """
    
    # Verificar si la base de datos ya existe
    if os.path.exists(db_path):
        response = input(f"La base de datos '{db_path}' ya existe. ¿Desea recrearla? (s/n): ")
        if response.lower() != 's':
            print("Operación cancelada.")
            return False
        else:
            os.remove(db_path)
            print(f"Base de datos anterior '{db_path}' eliminada.")
    
    try:
        # Crear conexión a la base de datos
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Crear tabla corpus
        create_table_sql = """
        CREATE TABLE corpus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            title TEXT NOT NULL,
            author TEXT,
            datepub TEXT,
            text TEXT NOT NULL,
            textoriginal TEXT,
            citations INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(create_table_sql)
        
        # Crear índices para optimizar consultas
        cursor.execute("CREATE INDEX idx_title ON corpus(title);")
        cursor.execute("CREATE INDEX idx_author ON corpus(author);")
        cursor.execute("CREATE INDEX idx_citations ON corpus(citations);")
        cursor.execute("CREATE INDEX idx_datepub ON corpus(datepub);")
        
        # Confirmar cambios
        conn.commit()
        
        print(f"✅ Base de datos '{db_path}' creada exitosamente.")
        print("✅ Tabla 'corpus' creada con los siguientes campos:")
        print("   - id (INTEGER PRIMARY KEY AUTOINCREMENT)")
        print("   - url (TEXT)")
        print("   - title (TEXT NOT NULL)")
        print("   - author (TEXT)")
        print("   - datepub (TEXT)")
        print("   - text (TEXT NOT NULL)")
        print("   - textoriginal (TEXT)")
        print("   - citations (INTEGER DEFAULT 0)")
        print("   - created_at (TIMESTAMP)")
        print("   - updated_at (TIMESTAMP)")
        
        # Mostrar información de la tabla
        cursor.execute("PRAGMA table_info(corpus);")
        columns = cursor.fetchall()
        print("\n📋 Estructura detallada de la tabla:")
        for col in columns:
            print(f"   {col[1]} - {col[2]} {'(NOT NULL)' if col[3] else ''}")
        
        return True
        
    except sqlite3.Error as e:
        print(f"❌ Error al crear la base de datos: {e}")
        return False
        
    finally:
        if conn:
            conn.close()

def insert_sample_document(db_path: str = "calcia.db"):
    """
    Función de utilidad para insertar un documento de prueba
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        sample_doc = {
            'url': 'https://ejemplo.com/doc1',
            'title': 'Documento de Prueba para CALCIA',
            'author': 'Investigador de Prueba',
            'datepub': '2024-01-01',
            'text': '''Este es un documento de prueba para validar el funcionamiento 
                      de la base de datos CALCIA. Contiene texto suficiente para 
                      generar n-gramas y probar los algoritmos de singularidad 
                      desarrollados por el Prof. Manuel Blázquez Ochando.''',
            'citations': 5
        }
        
        insert_sql = """
        INSERT INTO corpus (url, title, author, datepub, text, textoriginal, citations)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor.execute(insert_sql, (
            sample_doc['url'],
            sample_doc['title'], 
            sample_doc['author'],
            sample_doc['datepub'],
            sample_doc['text'],
            sample_doc['text'],  
            sample_doc['citations']
        ))
        
        conn.commit()
        doc_id = cursor.lastrowid
        
        print(f"✅ Documento de prueba insertado con ID: {doc_id}")
        return doc_id
        
    except sqlite3.Error as e:
        print(f"❌ Error al insertar documento: {e}")
        return None
        
    finally:
        if conn:
            conn.close()

def show_database_info(db_path: str = "calcia.db"):
    """
    Muestra información sobre el contenido de la base de datos
    """
    if not os.path.exists(db_path):
        print(f"❌ La base de datos '{db_path}' no existe.")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Contar documentos
        cursor.execute("SELECT COUNT(*) FROM corpus;")
        doc_count = cursor.fetchone()[0]
        
        print(f"\n📊 INFORMACIÓN DE LA BASE DE DATOS '{db_path}':")
        print(f"   Total de documentos: {doc_count}")
        
        if doc_count > 0:
            # Estadísticas básicas
            cursor.execute("SELECT MIN(citations), MAX(citations), AVG(citations) FROM corpus;")
            min_cit, max_cit, avg_cit = cursor.fetchone()
            
            cursor.execute("SELECT author, COUNT(*) FROM corpus GROUP BY author ORDER BY COUNT(*) DESC LIMIT 5;")
            top_authors = cursor.fetchall()
            
            print(f"   Citas mínimas: {min_cit}")
            print(f"   Citas máximas: {max_cit}")
            print(f"   Citas promedio: {avg_cit:.2f}")
            
            print(f"\n   Top 5 autores:")
            for author, count in top_authors:
                print(f"     {author or 'Sin autor'}: {count} documentos")
            
            # Mostrar algunos títulos
            cursor.execute("SELECT id, title, citations FROM corpus ORDER BY citations DESC LIMIT 3;")
            top_docs = cursor.fetchall()
            
            print(f"\n   Documentos más citados:")
            for doc_id, title, citations in top_docs:
                title_short = title[:50] + "..." if len(title) > 50 else title
                print(f"     [{doc_id}] {title_short} ({citations} citas)")
        
    except sqlite3.Error as e:
        print(f"❌ Error al consultar la base de datos: {e}")
        
    finally:
        if conn:
            conn.close()

def main():
    """
    Función principal del script
    """
    print("="*60)
    print("🗄️  CREADOR DE BASE DE DATOS CALCIA")
    print("   Sistema de Cálculo de Singularidad para IA")
    print("   Prof. Manuel Blázquez Ochando - UCM")
    print("="*60)
    
    while True:
        print("\n📋 OPCIONES DISPONIBLES:")
        print("1. Crear nueva base de datos CALCIA")
        print("2. Mostrar información de base de datos existente")
        print("3. Insertar documento de prueba")
        print("4. Salir")
        
        option = input("\nSeleccione una opción (1-4): ").strip()
        
        if option == "1":
            db_name = input("Nombre de la base de datos [calcia.db]: ").strip()
            if not db_name:
                db_name = "calcia.db"
            
            if create_calcia_database(db_name):
                print(f"\n🎉 Base de datos '{db_name}' lista para usar con el programa principal.")
            
        elif option == "2":
            db_name = input("Nombre de la base de datos [calcia.db]: ").strip()
            if not db_name:
                db_name = "calcia.db"
            show_database_info(db_name)
            
        elif option == "3":
            db_name = input("Nombre de la base de datos [calcia.db]: ").strip()
            if not db_name:
                db_name = "calcia.db"
                
            if os.path.exists(db_name):
                insert_sample_document(db_name)
                show_database_info(db_name)
            else:
                print(f"❌ La base de datos '{db_name}' no existe. Créela primero.")
                
        elif option == "4":
            print("👋 ¡Hasta pronto!")
            break
            
        else:
            print("❌ Opción no válida. Intente nuevamente.")

if __name__ == "__main__":
    main()
