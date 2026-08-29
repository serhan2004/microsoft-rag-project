import os
import db_helper
from sentence_transformers import SentenceTransformer

_embedder = None

def get_embedder():
    """Embedding modelini tek seferde yükler (singleton)."""
    global _embedder
    if _embedder is None:
        print("Yerel Embedding modeli yükleniyor (all-MiniLM-L6-v2)...")
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder

def chunk_text(content: str):
    """Ham metni satır satır / paragraf paragraf ayırır (Chunking)."""
    return [c.strip() for c in content.split("\n") if c.strip()]

def ingest_text(filename: str, content: str):
    """Tek bir metni chunk'layıp embedding'ler ve SQLite'a kaydeder.

    Kaydedilen chunk sayısını döndürür.
    """
    embedder = get_embedder()
    chunks = chunk_text(content)
    count = 0

    for chunk in chunks:
        embedding_vector = embedder.encode(chunk).tolist()
        db_helper.save_chunk(chunk, embedding_vector, source=filename)
        count += 1

    return count

def ingest_file(file_path: str):
    """Dosyayı okuyup SQLite'a işler. Kaydedilen chunk sayısını döndürür."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return ingest_text(os.path.basename(file_path), content)

def load_and_ingest():
    print("Veritabanı ilklendiriliyor...")
    db_helper.init_db()
    db_helper.clear_db()  # Her çalıştırmada temiz veri yüklemek için

    docs_dir = "docs"
    if not os.path.exists(docs_dir):
        print(f"'{docs_dir}' klasörü bulunamadı!")
        return

    print("Dokümanlar işleniyor ve vektörler SQLite'a kaydediliyor...")
    total = 0

    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt") or filename.endswith(".md"):
            file_path = os.path.join(docs_dir, filename)
            count = ingest_file(file_path)
            total += count
            print(f" {filename}: {count} chunk kaydedildi.")

    print(f"\n Doküman yükleme işlemi başarıyla tamamlandı! Toplam chunk: {total}")

if __name__ == "__main__":
    load_and_ingest()
