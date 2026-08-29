import sqlite3
import json
import numpy as np

DB_NAME = "rag_db.db"

def init_db():
    """SQLite veritabanını ve tabloyu oluşturur."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Text chunk'larını, embedding vektörlerini ve kaynak dosyayı saklayacak tablo
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_content TEXT NOT NULL,
            embedding TEXT NOT NULL,
            source TEXT
        )
    ''')

    conn.commit()
    conn.close()

def save_chunk(text: str, embedding: list, source: str = None):
    """Metin parçasını ve embedding vektörünü JSON formatında SQLite'a kaydeder."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Vektörü JSON string olarak saklıyoruz
    embedding_json = json.dumps(embedding)

    cursor.execute('''
        INSERT INTO document_chunks (text_content, embedding, source)
        VALUES (?, ?, ?)
    ''', (text, embedding_json, source))

    conn.commit()
    conn.close()

def clear_db():
    """Veritabanındaki eski verileri temizler."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM document_chunks")
    conn.commit()
    conn.close()

def cosine_similarity(vec_a, vec_b):
    """İki vektör arasındaki Cosine Similarity (Açısal Benzerlik) değerini hesaplar."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_similar_chunks(query_embedding: list, top_k: int = 3):
    """Soru vektörüne en yakın top_k adet metin parçasını SQLite'tan bulur.

    (text_content, source, score) tuple listesi döndürür.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT id, text_content, embedding, source FROM document_chunks")
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row_id, text, emb_json, source in rows:
        db_vec = json.loads(emb_json)
        score = cosine_similarity(query_embedding, db_vec)
        results.append((score, text, source))

    # Benzerlik skoruna göre büyükten küçüğe sırala
    results.sort(key=lambda x: x[0], reverse=True)

    # En yüksek skora sahip top_k kadar metni döndür
    return [(text, source, score) for score, text, source in results[:top_k]]

def count_chunks():
    """Veritabanındaki toplam chunk sayısını döndürür."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM document_chunks")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def list_sources():
    """Veritabanındaki kaynak dosyaları ve her birinin chunk sayısını döndürür."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT source, COUNT(*) FROM document_chunks GROUP BY source ORDER BY source")
    rows = cursor.fetchall()
    conn.close()
    return rows
