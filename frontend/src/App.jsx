import { useState, useRef } from "react";
import axios from "axios";

const API = "http://localhost:8000";

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [filename, setFilename] = useState("");
  const fileInputRef = useRef(null);

  const analyzeText = async () => {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const { data } = await axios.post(`${API}/analyze`, { text });
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);
    setFilename(file.name);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const { data } = await axios.post(`${API}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      if (data.extracted_text) setText(data.extracted_text);
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const wordCount = text.trim().split(/\s+/).filter(Boolean).length;
  const canAnalyze = wordCount >= 20 && !loading;

  return (
    <div className="min-h-screen bg-slate-50 py-10 px-4">
      <div className="max-w-3xl mx-auto">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-slate-900">AI Text Detector</h1>
          <p className="text-slate-600 mt-2">
            Paste text or upload a document to estimate whether it was AI-generated or human-written.
          </p>
        </header>

        <div className="mb-4 p-4 bg-white border border-slate-200 rounded-xl flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-slate-700">Upload a document</p>
            <p className="text-xs text-slate-500">Supports .txt, .docx, .pdf (max 5 MB)</p>
            {filename && (
              <p className="text-xs text-blue-600 mt-1">Loaded: {filename}</p>
            )}
          </div>
          <label className="px-4 py-2 bg-slate-800 text-white text-sm font-medium rounded-lg cursor-pointer hover:bg-slate-900">
            Choose file
            <input
              ref={fileInputRef}
              type="file"
              accept=".txt,.docx,.pdf"
              onChange={handleFile}
              className="hidden"
            />
          </label>
        </div>

        <div className="text-center text-xs text-slate-400 mb-4">— or paste text below —</div>

        <textarea
          className="w-full h-56 p-4 border border-slate-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
          placeholder="Paste at least 20 words of text here…"
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setFilename("");
          }}
        />

        <div className="flex items-center justify-between mt-3">
          <span className="text-sm text-slate-500">
            {wordCount} words {wordCount < 20 && wordCount > 0 && "(need at least 20)"}
          </span>
          <button
            onClick={analyzeText}
            disabled={!canAnalyze}
            className="px-6 py-2 bg-blue-600 text-white font-medium rounded-lg shadow-sm hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Analyzing…" : "Analyze"}
          </button>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        {result && <ResultCard result={result} />}
      </div>
    </div>
  );
}

function ResultCard({ result }) {
  const aiPct = (result.ai_probability * 100).toFixed(1);
  const humanPct = (result.human_probability * 100).toFixed(1);
  const confPct = (result.confidence * 100).toFixed(1);
  const leansAI = result.ai_probability > 0.5;

  return (
    <div className="mt-6 bg-white rounded-2xl shadow-md p-6 space-y-5">
      <div>
        <div className="flex justify-between text-sm font-medium text-slate-700 mb-1">
          <span>Human {humanPct}%</span>
          <span>AI {aiPct}%</span>
        </div>
        <div className="w-full h-4 bg-slate-200 rounded-full overflow-hidden">
          <div
            className={leansAI ? "h-full bg-red-500" : "h-full bg-green-500"}
            style={{ width: `${aiPct}%` }}
          />
        </div>
        <p className="text-sm text-slate-500 mt-1">Confidence: {confPct}%</p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Metric label="Perplexity" value={result.perplexity} hint="Lower = more AI-like" />
        <Metric
          label="Burstiness"
          value={result.features.burstiness.toFixed(3)}
          hint="Higher = more human-like"
        />
        <Metric label="Word count" value={result.features.word_count} />
        <Metric label="Sentences" value={result.features.sentence_count} />
      </div>

      <div>
        <h3 className="font-semibold text-slate-900 mb-2">Why this verdict</h3>
        <ul className="space-y-2">
          {result.explanation.map((reason, i) => (
            <li key={i} className="flex gap-2 text-sm text-slate-700">
              <span className="text-blue-500">•</span>
              <span>{reason}</span>
            </li>
          ))}
        </ul>
      </div>

      <details className="text-sm">
        <summary className="cursor-pointer font-semibold text-slate-700">
          Signal breakdown
        </summary>
        <div className="mt-2 space-y-1 text-slate-600">
          {Object.entries(result.components).map(([k, v]) => (
            <div key={k} className="flex justify-between border-b border-slate-100 py-1">
              <span>{k.replace(/_/g, " ")}</span>
              <span className="font-mono">{(v * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </details>

      <details className="text-sm">
        <summary className="cursor-pointer font-semibold text-slate-700">
          Raw features
        </summary>
        <pre className="bg-slate-50 rounded p-3 mt-2 overflow-x-auto text-xs">
{JSON.stringify(result.features, null, 2)}
        </pre>
      </details>
    </div>
  );
}

function Metric({ label, value, hint }) {
  return (
    <div className="bg-slate-50 rounded-lg p-3">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className="text-2xl font-bold text-slate-900 mt-1">{value}</div>
      {hint && <div className="text-xs text-slate-500 mt-1">{hint}</div>}
    </div>
  );
}