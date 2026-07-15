# Aturan keputusan V21

Label model tetap `0=REAL` dan `1=FAKE`. Probabilitas kelas FAKE diambil dari classifier, lalu `base_use_flip` diterapkan tepat satu kali oleh `apply_flip_score`; local similarity menghasilkan `fake_score` final dan `real_score=1-fake_score`.

Keputusan terpusat pada `determine_final_decision`: probabilitas non-finite ditolak, nilai di luar rentang di-clamp, dan `math.isclose(..., abs_tol=1e-9, rel_tol=0)` digunakan hanya untuk kesetaraan 0.5. `<0.5=REAL`, `>0.5=DEEPFAKE`, tepat 0.5=`MENCURIGAKAN`. Keputusan memakai nilai penuh; pembulatan enam digit hanya untuk response.

Contoh `fake_score=0.444902`, sehingga `real_score=0.555098` dan hasilnya REAL.

