# 🧠 Yerel RAG Asistanı (Local RAG Assistant)

Bu proje; internet bağlantısı olmadan, tamamen cihaz üzerinde yerel olarak çalışan ve doküman tabanlı soru-cevap yapabilen bir **Retrieval-Augmented Generation (RAG)** uygulamasıdır.

Harici API'lere bağımlı kalmadan, veri gizliliğini maksimum seviyede tutarak; dokümanlarınızı vektör benzerlik araması (Cosine Similarity) ile tarar ve yanıtı yerel bir dil modeli ile üretir.

---

## 🛠️ Mimari ve Kullanılan Teknolojiler

- **Python 3.9+**: Ana programlama dili.
- **Ollama (`phi4-mini`)**: Microsoft'un yerel olarak çalışan Phi-4 Mini modeli — yanıt üretimi.
- **Sentence-Transformers (`all-MiniLM-L6-v2`)**: Metinleri 384 boyutlu vektörlere dönüştüren offline embedding modeli.
- **SQLite3**: Sunucusuz, hafif ve yerel vektör depolama çözümü.
- **Streamlit**: İnteraktif ve kullanıcı dostu web arayüzü (sohbet + dosya yükleme).
- **NumPy**: Vektörler arası Cosine Similarity (Açısal Benzerlik) hesaplaması.

---

## 📂 Proje Yapısı

```
microsoft-rag-project/
├── docs/
│   └── sample_kb.txt       # Vektörleştirilecek ham metin dokümanları
├── app.py                  # Streamlit web arayüzü: sohbet, dosya yükleme ve RAG pipeline
├── db_helper.py            # SQLite veritabanı ve Cosine Similarity arama modülü
├── ingest.py               # Doküman okuma, chunking ve embedding işleme betiği
├── requirements.txt        # Proje bağımlılıkları
└── .gitignore              # Git dışı bırakılacak dosyalar (db, venv vb.)
```

---

## 🚀 Kurulum ve Çalıştırma Adımları

### 1. Ollama'yı kurun ve modeli indirin

```bash
brew install ollama          # macOS
brew services start ollama
ollama pull phi4-mini        # ~2.5 GB, ilk kullanımda iner
```

> Diğer işletim sistemleri için: https://ollama.com

### 2. Python ortamını hazırlayın

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Örnek dokümanları vektörleştirip SQLite'a yükleyin

```bash
python ingest.py
```

Bu komut `docs/` klasöründeki `.txt`/`.md` dosyalarını okur, satır satır chunk'lara böler, her parçayı `all-MiniLM-L6-v2` embedding modeli ile vektöre çevirir ve `rag_db.db` dosyasına kaydeder.

### 4. Web arayüzünü başlatın

```bash
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` adresine gidin.

---

## ✨ Özellikler

- **💭 Sohbet ekranı**: Sorunuz aynı embedding modeli ile vektöre çevrilir; SQLite'tan Cosine Similarity ile en alakalı 3 parça çekilir; bu parçalar bağlam (context) olarak `phi4-mini` modeline prompt verilir ve yanıt yerel olarak üretilir.
- **🔍 Kaynak gösterimi**: Her yanıtın altında kullanılan parçalar, kaynak dosya adı ve benzerlik skoru ile listelenir (nereden alıntı yapıldığı görülür).
- **📂 Dosya yükleme ekranı**: `.txt` / `.md` dosyaları anında chunk'lanıp vektöre çevrilerek veritabanına eklenir; dizinlenen kaynaklar listelenir.
- **💬 Sohbet geçmişi**: Aynı oturumda konuşma akışı korunur.

---

## 🔍 RAG Pipeline Nasıl Çalışır?

1. **İndeksleme (Indexing)**: `ingest.py` dokümanları okur, satır bazlı chunk'lara ayırır ve her parçanın vektörünü SQLite'a kaydeder.
2. **Alma (Retrieval)**: Sohbetteki soru aynı embedding modeli ile vektöre çevrilir.
3. **Skorlama**: `db_helper.search_similar_chunks()` veritabanındaki tüm chunk'ların Cosine Similarity skorunu hesaplar.
4. **Bağlam Oluşturma**: En yüksek skorlu `top_k=3` parça bağlam olarak `phi4-mini` modelinin prompt'una eklenir.
5. **Üretim (Generation)**: Model yalnızca bağlama dayanarak yanıt üretir (halüsinasyonu önlemek için sistem prompt'u bunu zorunlu kılar).

---

> **Not:** Bu proje bir Proof of Concept (PoC)'tir. Tüm işlemler yerelde gerçekleşir; verileriniz cihazdan çıkmaz ve hiçbir bulut servisine gönderilmez.
