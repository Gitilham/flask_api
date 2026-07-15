# Profiling performa

Baseline pengguna: 32 frame, 32 wajah, sekitar 18,36 detik atau 0,57 detik/frame. Belum ada benchmark sesudah perubahan.

`feature_debug.timings` mencatat video probe, ekstraksi frame, deteksi+crop wajah, Xception, feature vector, classifier+local similarity, dan total pipeline. Log request tetap memiliki request_id dan total processing time. Detail timing publik dapat disaring di deployment production bila tidak diperlukan.

Benchmark manual membutuhkan sample video asli dan dijalankan pengguna. Bandingkan label, frame, wajah, feature shape, semua score, total waktu, RAM, dan CPU; jangan menerima perbedaan probabilitas di atas toleransi regression yang terdokumentasi.

