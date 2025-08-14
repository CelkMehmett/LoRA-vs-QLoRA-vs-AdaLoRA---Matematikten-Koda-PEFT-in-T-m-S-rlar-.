# LoRA vs QLoRA vs AdaLoRA — Matematikten Koda, PEFT'in Tüm Sırları

Bu repo, **Parameter-Efficient Fine-Tuning (PEFT)** tekniklerini — **LoRA**, **QLoRA** ve **AdaLoRA** — matematiksel temellerinden başlayarak, pratik Python kod örnekleri ile anlatan kapsamlı bir çalışmayı içerir.  
Notebook, hem teorik açıklamaları hem de Hugging Face PEFT kütüphanesi ile çalışan eğitim kodlarını bir arada sunar.

## 📚 İçerik
- **Giriş: Neden PEFT?** — Tam fine-tuning’in maliyeti, PEFT avantajları, adapter kavramı.
- **LoRA** — Düşük-rank adaptör mantığı, matematik formülleri, parametre tasarruf hesapları, kod örnekleri.
- **QLoRA** — 4-bit quantization (NF4 + double quant), bellek optimizasyonu, Hugging Face + bitsandbytes ayarları.
- **AdaLoRA** — Dinamik rank tahsisi, önem skorları ile kapasite yönetimi, hiperparametreler.
- **Hangisini Ne Zaman Kullanmalı?** — Karar tabloları ve senaryo örnekleri.
- **İleri İpuçları** — VRAM’e göre ayar tarifleri, gradient checkpointing, flash-attention, adapter fusion, merge vs deployment.

## 🛠 Kurulum
Bu notebook’u çalıştırmak için Python 3.10+ ve aşağıdaki kütüphaneler gereklidir:

```bash
pip install -U transformers datasets peft accelerate bitsandbytes
```

Eğer GPU kullanıyorsanız, CUDA sürümünüze uygun **bitsandbytes** versiyonunu yüklemeyi unutmayın.

## ▶️ Kullanım
1. Bu repo’yu klonlayın:
   ```bash
   git clone https://github.com/kullanici_adi/LoRA-QLoRA-AdaLoRA-PEFT.git
   cd LoRA-QLoRA-AdaLoRA-PEFT
   ```

2. Notebook’u açın:
   ```bash
   jupyter notebook "LoRA_vs_QLoRA_vs_AdaLoRA _ Matematikten_Koda,_PEFT'in_Tüm Sırları.ipynb"
   ```

3. **Hugging Face Token** ile giriş yapmanız gerekiyorsa:
   ```python
   from huggingface_hub import login
   login(token="HF_TOKENINIZ")
   ```

4. İlgili hücreleri çalıştırarak LoRA, QLoRA veya AdaLoRA eğitim örneklerini deneyin.

## 💡 Örnek Çıktılar
- LoRA ile **orta boy model** üzerinde hızlı fine-tuning
- QLoRA ile **7B+ modelleri** tek GPU’da eğitme
- AdaLoRA ile **aynı parametre bütçesinde** daha yüksek kalite

## 📄 Lisans
Bu proje MIT lisansı ile lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.
