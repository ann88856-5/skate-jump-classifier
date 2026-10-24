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
    const selected = e.target.files?.[0] || null;
    setFile(selected);
    if (selected) {
      setVideoUrl(URL.createObjectURL(selected));
    } else {
      setVideoUrl(null);
    }
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
      margin: '0 auto',        // ← центрирование по горизонтали
      textAlign: 'center'      // ← центрирование содержимого
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

      {error && <p style={{ color: 'red' }}>Ошибка: {error}</p>}

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
              margin: '0 auto'     // ← центрирование видео
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