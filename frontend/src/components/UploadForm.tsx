import { useState } from 'react';
import ResultScreen from './ResultScreen';

interface AnalyzeResult {
  session_id: number | null;
  jump_type: string;
  confidence: number;
  rotations: number;
  all_probs: Record<string, number>;
}

export default function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null); // Сбрасываем старую ошибку
    const selected = e.target.files?.[0] || null;
    
    if (!selected) {
      setFile(null);
      setVideoUrl(null);
      return;
    }

    // 1. Проверка типа файла (видео или нет)
    if (!selected.type.startsWith("video/")) {
      setError("Ошибка: выбранный файл не является видео.");
      setFile(null);
      setVideoUrl(null);
      return;
    }

    // 2. Проверка размера (макс 200 МБ)
    const MAX_SIZE_MB = 200;
    if (selected.size > MAX_SIZE_MB * 1024 * 1024) {
      setError(`Ошибка: файл слишком большой (макс. ${MAX_SIZE_MB} МБ).`);
      setFile(null);
      setVideoUrl(null);
      return;
    }

    // Если всё ок сохраняем файл
    setFile(selected);
    setVideoUrl(URL.createObjectURL(selected));
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Ошибка');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      padding: 20,
      maxWidth: 800,
      margin: '0 auto',
      textAlign: 'center'
    }}>
      <h2>Загрузка видео</h2>

      <input
        type="file"
        accept="video/*"
        onChange={handleFileChange}
      />

      <button
        onClick={handleUpload}
        disabled={!file || loading}
        style={{ marginLeft: 10 }}
      >
        {loading ? 'Анализ...' : 'Анализировать'}
      </button>

      {error && <p style={{ color: 'red', fontWeight: 'bold' }}>Ошибка: {error}</p>}

      {videoUrl && (
        <div style={{ marginTop: 20 }}>
          <h4>Выбранное видео:</h4>
          <video
            src={videoUrl}
            controls
            style={{
              width: '100%',
              maxWidth: 600,
              borderRadius: 8,
              border: '1px solid #ccc',
              display: 'block',
              margin: '0 auto'
            }}
          >
            Ваш браузер не поддерживает видео.
          </video>
        </div>
      )}

      {result && <ResultScreen result={result} />}
    </div>
  );
}