interface AnalyzeResult {
  session_id: number | null;
  jump_type: string;
  confidence: number;
  rotations: number;
  all_probs: Record<string, number>;
  skeleton_url?: string;
}

interface Props {
  result: AnalyzeResult;
}

export default function ResultScreen({ result }: Props) {
  const skeletonUrl = result.skeleton_url
    ? `http://localhost:8000${result.skeleton_url}`
    : null;

  const reportUrl = result.session_id
    ? `http://localhost:8000/report/${result.session_id}`
    : null;

  return (
    <div style={{ marginTop: 20, padding: 20, border: '2px solid #4CAF50', borderRadius: 8, background: '#f9fff9' }}>
      <h3 style={{ color: '#4CAF50' }}>Результат анализа</h3>
      <p><b>ID сессии:</b> {result.session_id}</p>
      <p style={{ fontSize: 20 }}><b>Тип прыжка:</b> <span style={{ color: '#2196F3' }}>{result.jump_type}</span></p>
      <p><b>Уверенность:</b> {(result.confidence * 100).toFixed(1)}%</p>
      <p><b>Обороты:</b> {result.rotations.toFixed(2)}</p>

      {reportUrl && (
        <div style={{ margin: '20px 0' }}>
          <a href={reportUrl} target="_blank" rel="noopener noreferrer" style={{ display: 'inline-block', padding: '10px 20px', backgroundColor: '#f44336', color: 'white', textDecoration: 'none', borderRadius: '5px', fontWeight: 'bold' }}>
            📄 Скачать PDF-отчёт
          </a>
        </div>
      )}

      {skeletonUrl && (
        <>
          <h4>Видео с наложенным скелетом:</h4>
          <video src={skeletonUrl} controls autoPlay loop style={{ width: '100%', maxWidth: 600, borderRadius: 8, border: '1px solid #ccc', display: 'block', margin: '0 auto' }}>
            Ваш браузер не поддерживает видео.
          </video>
        </>
      )}

      <h4>Вероятности по классам:</h4>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {Object.entries(result.all_probs).sort(([, a], [, b]) => b - a).map(([cls, prob]) => (
          <li key={cls} style={{ marginBottom: 5 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ width: 80 }}>{cls}:</span>
              <div style={{ flex: 1, height: 12, background: '#eee', borderRadius: 6, overflow: 'hidden' }}>
                <div style={{ width: `${prob * 100}%`, height: '100%', background: '#4CAF50' }} />
              </div>
              <span style={{ width: 60, textAlign: 'right' }}>{(prob * 100).toFixed(1)}%</span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}