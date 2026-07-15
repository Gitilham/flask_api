# Audit migrasi Flask ke FastAPI

`app.py` lama adalah Flask monolitik yang memuat classifier, scaler, YOLO, dan Xception saat import. Route-nya `/`, `/health`, `/model-info`, `/predict-video`; field upload adalah `video`. Upload lama tidak dibersihkan.

Pipeline mengambil frame dengan `numpy.linspace`, crop YOLO dengan fallback center crop, embedding Xception, lalu menyusun fitur: agregasi embedding, agregasi fitur visual yang di-scaler per frame, dan agregasi artefak. Panjang final 21.250. V21 memakai probabilitas class 1, optional flip, L2, weighted local reference, dan blend borderline. Kontrak modelnya tetap threshold 0.5, label `0=REAL/1=FAKE`, config `k=3,temp=.06,power=1,alpha=.55,mode=borderline`. Aturan tampilan keputusan diperbarui: kurang dari 50% REAL, lebih dari 50% DEEPFAKE, dan tepat 50% MENCURIGAKAN.

Source Flask dibackup utuh di `legacy/app_flask_v21.py`. Fungsi algoritma dipertahankan; `torch.inference_mode()` hanya menonaktifkan gradient YOLO. Registry lifespan memuat model satu kali. Inferensi memakai thread+semaphore. Upload di-stream ke file acak dan dibersihkan dalam `finally`. `/health` menjadi liveness dan `/ready` menjadi readiness.

Belum terbukti tanpa build/sample: regresi video penuh, kompatibilitas binary pickle, durasi load, RAM, dan inferensi nyata. Tidak ada sample video dalam repository.
