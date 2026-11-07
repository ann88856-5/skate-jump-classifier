import { useRef, useState } from "react";

const skaterImage =
  "https://images.unsplash.com/photo-1509362803586-12d1a67df239?crop=entropy&cs=tinysrgb&fit=crop&fm=jpg&q=88&w=1600";

type IconProps = { className?: string };

function Arrow({ className = "size-5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M5 12h14M13 6l6 6-6 6" />
    </svg>
  );
}

function Play({ className = "size-5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <path d="M8 5.5v13l11-6.5-11-6.5Z" />
    </svg>
  );
}

function SparkIcon() {
  return (
    <svg className="size-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M12 2l1.25 5.3L18 9l-4.75 1.7L12 16l-1.25-5.3L6 9l4.75-1.7L12 2Z" />
      <path d="M18.5 15l.7 2.8 2.8.7-2.8.7-.7 2.8-.7-2.8-2.8-.7 2.8-.7.7-2.8Z" />
    </svg>
  );
}

const features = [
  {
    n: "01",
    title: "Распознаёт прыжок",
    text: "Нейросеть точно определяет тип элемента: аксель, лутц, флип, риттбергер, сальхов или тулуп.",
  },
  {
    n: "02",
    title: "Считает обороты",
    text: "Алгоритм на основе 3D-скелета вычисляет количество вращений с высокой точностью.",
  },
  {
    n: "03",
    title: "Анализирует технику",
    text: "Система визуализирует скелет на видео, позволяя разобрать каждый элемент.",
  },
];

const faqs = [
  [
    "Какие видео подходят для анализа?",
    "Подойдёт запись в MP4 или MOV до 200 МБ. Лучше всего — статичная камера, полный рост спортсмена и хорошо видимые коньки.",
  ],
  [
    "Насколько точен анализ?",
    "Наша нейросеть показывает точность 93.6% на тестовых данных. Система помогает разбирать технику, но не заменяет официальное судейство.",
  ],
];

export default function App() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState("");
  const [openFaq, setOpenFaq] = useState(0);
  const [showDemo, setShowDemo] = useState(false);

  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  const chooseVideo = () => inputRef.current?.click();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setFileName(selected.name);
      setVideoUrl(URL.createObjectURL(selected));
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setShowDemo(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Ошибка:", error);
      alert("Ошибка при загрузке. Проверьте, запущен ли сервер Python.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen overflow-hidden bg-[#f3f7f8] text-[#0c2730]">
      <header className="absolute inset-x-0 top-0 z-30">
        <div className="mx-auto flex max-w-[1400px] items-center justify-between px-5 py-6 md:px-10 lg:px-14">
          <a href="#" className="flex items-center gap-2.5 text-white">
            <span className="grid size-9 place-items-center rounded-full border border-white/30 bg-white/10 backdrop-blur">
              <SparkIcon />
            </span>
            <span className="text-xl font-semibold tracking-[-0.04em]">FigureSkate AI</span>
          </a>
          <nav className="hidden items-center gap-8 text-sm text-white/75 md:flex">
            <a className="transition hover:text-white" href="#how">
              Как работает
            </a>
            <a className="transition hover:text-white" href="#analysis">
              Возможности
            </a>
            <a className="transition hover:text-white" href="#faq">
              Вопросы
            </a>
          </nav>
        </div>
      </header>

      <section className="relative min-h-[790px] bg-[#0b252c] pt-32 text-white md:min-h-[850px]">
        <img
          src={skaterImage}
          alt="Фигуристка выполняет элемент на льду"
          className="absolute inset-0 h-full w-full object-cover object-[60%_center] opacity-65"
        />
        <div className="absolute inset-0 bg-[linear-gradient(90deg,rgba(5,25,31,.98)_0%,rgba(5,25,31,.88)_36%,rgba(5,25,31,.22)_72%,rgba(5,25,31,.5)_100%)]" />
        <div className="absolute inset-0 bg-[linear-gradient(0deg,rgba(5,25,31,.82)_0%,transparent_38%)]" />
        <div className="ice-lines absolute inset-0 opacity-30" />

        <div className="relative z-10 mx-auto grid max-w-[1400px] px-5 pt-14 md:px-10 md:pt-24 lg:grid-cols-[1fr_420px] lg:px-14">
          <div className="max-w-[760px]">
            <h1 className="text-[clamp(2.5rem,5.4vw,4.5rem)] font-medium leading-[1.06] tracking-[-0.05em]">
              Классификатор прыжков
              <br />
              <span className="font-serif font-normal italic text-[#a8dfdb]">в фигурном катании</span>
            </h1>
            <p className="mt-8 max-w-[560px] text-lg leading-8 text-white/65 md:text-xl">
              Нейросеть распознает тип прыжка, оценивает технику и объясняет результат — кадр за
              кадром.
            </p>
            <div className="mt-10 flex flex-col gap-3 sm:flex-row">
              <button
                onClick={file ? handleAnalyze : chooseVideo}
                disabled={loading}
                className="group flex items-center justify-center gap-3 rounded-full bg-[#b9f0eb] px-7 py-4 font-semibold text-[#0b252c] transition hover:scale-[1.02] hover:bg-white disabled:opacity-50"
              >
                {loading ? "Анализируем..." : file ? "Начать анализ" : "Загрузить видео"}
                {!loading && <Arrow className="size-5 transition group-hover:translate-x-1" />}
              </button>
              <button
                onClick={() => setShowDemo(true)}
                className="flex items-center justify-center gap-3 rounded-full border border-white/25 bg-white/5 px-7 py-4 font-medium text-white backdrop-blur transition hover:bg-white/15"
              >
                <span className="grid size-7 place-items-center rounded-full bg-white text-[#0b252c]">
                  <Play className="ml-0.5 size-3" />
                </span>
                Смотреть демо
              </button>
            </div>
            {fileName && <p className="mt-4 text-sm text-[#b9f0eb]">Выбрано: {fileName}</p>}
          </div>

          <div className="mt-16 self-end lg:mt-52">
            <div className="score-card ml-auto max-w-[390px] rounded-[28px] border border-white/20 bg-[#09232a]/75 p-5 shadow-2xl backdrop-blur-xl">
              <div className="mb-5 flex items-start justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.16em] text-white/45">
                    Результат анализа
                  </p>
                  <p className="mt-1 text-2xl font-medium">
                    {result ? result.jump_type : "Ожидание видео..."}
                  </p>
                </div>
                <span className="rounded-full bg-[#b9f0eb] px-3 py-1.5 text-xs font-semibold text-[#0c2730]">
                  {result ? `${(result.confidence * 100).toFixed(0)}%` : "--%"}
                </span>
              </div>
              <div className="mb-4 h-20 overflow-hidden rounded-2xl bg-white/7">
                <svg className="h-full w-full" viewBox="0 0 350 80" preserveAspectRatio="none">
                  <path
                    d="M0 63 C35 62,47 58,72 58 S112 57,127 47 S154 9,182 14 S214 54,242 56 S300 57,350 62"
                    fill="none"
                    stroke="#89e3db"
                    strokeWidth="2"
                  />
                  <path
                    d="M0 63 C35 62,47 58,72 58 S112 57,127 47 S154 9,182 14 S214 54,242 56 S300 57,350 62 L350 80 L0 80Z"
                    fill="url(#wave)"
                    opacity=".18"
                  />
                  <defs>
                    <linearGradient id="wave" x1="0" y1="0" x2="0" y2="1">
                      <stop stopColor="#89e3db" />
                      <stop offset="1" stopColor="#89e3db" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                </svg>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div className="rounded-2xl p-3 bg-white/[.07]">
                  <p className="text-xl font-semibold text-[#b9f0eb]">
                    {result ? result.jump_type : "Тип"}
                  </p>
                  <p className="mt-1 text-xs text-white/40">Прыжок</p>
                </div>
                <div className="rounded-2xl p-3 bg-white/[.07]">
                  <p className="text-xl font-semibold">
                    {result ? result.rotations.toFixed(2) : "0.00"}
                  </p>
                  <p className="mt-1 text-xs text-white/40">Обороты</p>
                </div>
                <div className="rounded-2xl p-3 bg-[#b9f0eb] text-[#0c2730]">
                  <p className="text-xl font-semibold">
                    {result ? `${(result.confidence * 100).toFixed(0)}%` : "--%"}
                  </p>
                  <p className="mt-1 text-xs text-[#0c2730]/55">Точность</p>
                </div>
              </div>
              {result && result.session_id && (
                <a
                  href={`http://127.0.0.1:8000/report/${result.session_id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-4 flex w-full items-center justify-center gap-2 rounded-2xl bg-white/10 py-3 text-sm font-medium text-white transition hover:bg-white/20"
                >
                  <svg className="size-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                    <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Скачать PDF-отчёт
                </a>
              )}
            </div>
          </div>
        </div>
      </section>

      <section id="how" className="px-5 py-24 md:px-10 md:py-32 lg:px-14">
        <div className="mx-auto max-w-[1300px]">
          <div className="grid gap-8 lg:grid-cols-[.8fr_1.2fr]">
            <div>
              <p className="section-label">Как это работает</p>
              <h2 className="mt-5 max-w-md text-4xl font-medium leading-[1.06] tracking-[-0.05em] md:text-6xl">
                От видео до анализа —{" "}
                <span className="font-serif italic text-[#397c7b]">три шага</span>
              </h2>
            </div>
            <div className="border-t border-[#0c2730]/15">
              {features.map((item) => (
                <div
                  key={item.n}
                  className="group grid gap-4 border-b border-[#0c2730]/15 py-7 sm:grid-cols-[50px_220px_1fr] sm:items-start"
                >
                  <span className="text-sm text-[#397c7b]">{item.n}</span>
                  <h3 className="text-xl font-semibold tracking-[-0.03em]">{item.title}</h3>
                  <p className="leading-7 text-[#0c2730]/55">{item.text}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section id="analysis" className="bg-[#dce9e9] px-5 py-24 md:px-10 md:py-32 lg:px-14">
        <div className="mx-auto max-w-[1300px]">
          <div className="mb-14 flex flex-col justify-between gap-5 md:flex-row md:items-end">
            <div>
              <p className="section-label">Внутри анализа</p>
              <h2 className="mt-5 text-4xl font-medium tracking-[-0.05em] md:text-6xl">
                AI использует <br />
                <span className="font-serif italic text-[#397c7b]">следующие ключевые метрики</span>
              </h2>
            </div>
            <p className="max-w-sm leading-7 text-[#0c2730]/55">
              FigureSkate AI превращает сложную биомеханику в понятные метрики.
            </p>
          </div>
          <div className="grid gap-4 md:grid-cols-3">
            {[
              [
                result ? result.jump_type : "Loop", 
                "тип прыжка", 
                "Нейросеть определяет один из шести типов прыжков с точностью 93.6%."
              ],
              [
                result ? result.rotations.toFixed(2) : "3.00", 
                "оборота", 
                "Система вычисляет количество полных вращений фигуриста в воздухе."
              ],
              [
                result ? "93.6%" : "93.6%", 
                "точность", 
                "Общая точность модели BiLSTM, обученной на 3D-скелетах фигуристов."
              ],
            ].map(([value, label, text], i) => (
              <article
                key={label}
                className={`metric-card min-h-[320px] rounded-[28px] p-7 ${i === 1 ? "bg-[#0c2b33] text-white" : "bg-[#f5f8f8]"}`}
              >
                <p
                  className={`text-xs uppercase tracking-[.18em] ${i === 1 ? "text-white/40" : "text-[#0c2730]/40"}`}
                >
                  0{i + 1} / Метрика
                </p>
                <div className="mt-20">
                  <p className="text-5xl font-medium tracking-[-0.06em]">{value}</p>
                  <p className={`mt-1 text-xl ${i === 1 ? "text-[#a8dfdb]" : "text-[#397c7b]"}`}>
                    {label}
                  </p>
                  <p
                    className={`mt-7 max-w-xs leading-6 text-center mx-auto ${i === 1 ? "text-white/50" : "text-[#0c2730]/50"}`}
                  >
                    {text}
                  </p>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section id="faq" className="px-5 pb-24 pt-24 md:px-10 md:pb-32 lg:px-14">
        <div className="mx-auto grid max-w-[1100px] gap-10 lg:grid-cols-[.7fr_1.3fr]">
          <div>
            <p className="section-label">FAQ</p>
            <h2 className="mt-4 text-4xl font-medium tracking-[-0.05em]">
              Важное
              <br />о сервисе
            </h2>
          </div>
          <div className="border-t border-[#0c2730]/15">
            {faqs.map(([q, a], i) => (
              <button
                key={q}
                onClick={() => setOpenFaq(openFaq === i ? -1 : i)}
                className="w-full border-b border-[#0c2730]/15 py-6 text-left"
              >
                <span className="flex items-center justify-between gap-5 text-lg font-semibold">
                  <span>{q}</span>
                  <span className="text-2xl font-light text-[#397c7b]">
                    {openFaq === i ? "−" : "+"}
                  </span>
                </span>
                <span
                  className={`grid overflow-hidden text-[#0c2730]/55 transition-all ${openFaq === i ? "mt-4 grid-rows-[1fr]" : "grid-rows-[0fr]"}`}
                >
                  <span className="min-h-0 leading-7">{a}</span>
                </span>
              </button>
            ))}
          </div>
        </div>
      </section>

      <footer className="border-t border-[#0c2730]/10 px-5 py-8 md:px-10 lg:px-14">
        <div className="mx-auto flex max-w-[1300px] flex-col gap-5 text-sm text-[#0c2730]/45 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-2 font-semibold text-[#0c2730]">
            <SparkIcon /> FigureSkate AI
          </div>
          <p>AI-инструмент для тренировок. Не заменяет официальное судейство.</p>
        </div>
      </footer>

      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={handleFileChange}
      />

      {showDemo && (
        <div
          className="fixed inset-0 z-50 grid place-items-center bg-[#06181d]/80 p-5 backdrop-blur-sm"
          onClick={() => setShowDemo(false)}
        >
          <div
            className="relative aspect-video w-full max-w-4xl overflow-hidden rounded-[28px] bg-[#102f36] shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {result && result.skeleton_url ? (
              <video
                src={`http://127.0.0.1:8000${result.skeleton_url}`}
                controls
                autoPlay
                loop
                className="h-full w-full object-contain" 
              />
            ) : videoUrl ? (
               <video
                src={videoUrl}
                controls
                autoPlay
                className="h-full w-full object-contain opacity-50"
              />
            ) : (
              <div className="grid h-full place-items-center text-white/50">
                Загрузите видео для анализа
              </div>
            )}
            
            {loading && (
              <div className="absolute inset-0 grid place-items-center bg-black/60 text-white">
                <div className="text-center">
                   <div className="mx-auto mb-4 h-10 w-10 animate-spin rounded-full border-4 border-white/20 border-t-[#b9f0eb]"></div>
                   <p>Нейросеть анализирует видео...</p>
                </div>
              </div>
            )}

            <button
              className="absolute right-5 top-5 grid size-10 place-items-center rounded-full bg-black/50 text-xl text-white hover:bg-black/80 transition"
              onClick={() => setShowDemo(false)}
              aria-label="Закрыть"
            >
              ×
            </button>
          </div>
        </div>
      )}
    </main>
  );
}