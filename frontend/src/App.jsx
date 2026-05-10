import React, { useState, useRef } from 'react';
import { Copy, Mic, Upload, FileText, Sparkles, Loader2, Github, Languages, ChevronLeft, PanelLeftClose, PanelLeftOpen, Plus, Download, Pencil, Check, X } from 'lucide-react';
import HomePage from './HomePage';

const API_BASE_URL = 'http://localhost:8000';

const SUPPORTED_LANGS = {
  'Albanian': 'sq', 'Arabic': 'ar', 'Armenian': 'hy', 'Assamese': 'as', 'Azerbaijani': 'az', 'Bengali': 'bn', 'Bhojpuri': 'bho', 'Bosnian': 'bs', 'Bulgarian': 'bg', 'Catalan': 'ca', 'Chinese (simplified)': 'zh-CN', 'Chinese (traditional)': 'zh-TW', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Dutch': 'nl', 'English': 'en', 'Estonian': 'et', 'Filipino': 'tl', 'Finnish': 'fi', 'French': 'fr', 'Georgian': 'ka', 'German': 'de', 'Greek': 'el', 'Gujarati': 'gu', 'Hindi': 'hi', 'Hungarian': 'hu', 'Icelandic': 'is', 'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it', 'Japanese': 'ja', 'Javanese': 'jw', 'Kannada': 'kn', 'Kazakh': 'kk', 'Khmer': 'km', 'Korean': 'ko', 'Lao': 'lo', 'Latvian': 'lv', 'Lithuanian': 'lt', 'Macedonian': 'mk', 'Malay': 'ms', 'Malayalam': 'ml', 'Marathi': 'mr', 'Mongolian': 'mn', 'Myanmar': 'my', 'Nepali': 'ne', 'Norwegian': 'no', 'Odia (oriya)': 'or', 'Pashto': 'ps', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt', 'Punjabi': 'pa', 'Romanian': 'ro', 'Russian': 'ru', 'Sanskrit': 'sa', 'Serbian': 'sr', 'Sindhi': 'sd', 'Sinhala': 'si', 'Slovak': 'sk', 'Slovenian': 'sl', 'Spanish': 'es', 'Swedish': 'sv', 'Tamil': 'ta', 'Telugu': 'te', 'Thai': 'th', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uzbek': 'uz', 'Vietnamese': 'vi', 'Welsh': 'cy'
};

// ─── Safe error messages by HTTP status ─────────────────────────────────────
const STATUS_MESSAGES = {
  400: 'Invalid request. Please check your input.',
  422: 'Invalid input format.',
  500: 'Server error. Please try again.',
  503: 'Service unavailable. Please try again later.',
};

const apiFetch = async (url, options = {}) => {
  const response = await fetch(url, options);
  if (!response.ok) {
    const message = STATUS_MESSAGES[response.status] || 'Something went wrong. Please try again.';
    throw new Error(message);
  }
  return response.json();
};

// ─── Session state factory ───────────────────────────────────────────────────
const createSession = (label = 'New Session') => ({
  id: Date.now(),
  label,
  mode: 'manual',
  liveAudio: null,
  uploadedFile: null,
  manualText: '',
  manualName: '',
  isRecording: false,
  transcriptionResult: null,
  editedTranscription: '',
  processedResults: [],
  toolResult: null,
  selectedTool: '',
  targetLang: 'es',
  emailData: { to: '', subject: '', user_message: '', sender: '' },
});

// ─── Sidebar Component ───────────────────────────────────────────────────────
const Sidebar = ({ sessions, activeId, onSelect, onNew, onRename, collapsed, onToggle }) => {
  const [editingId, setEditingId] = useState(null);
  const [editValue, setEditValue] = useState('');

  const startEdit = (e, s) => {
    e.stopPropagation();
    setEditingId(s.id);
    setEditValue(s.label);
  };

  const commitEdit = (id) => {
    if (editValue.trim()) onRename(id, editValue.trim());
    setEditingId(null);
  };

  const cancelEdit = () => setEditingId(null);

  return (
    <aside className={`flex flex-col bg-slate-950 border-r border-slate-800/70 transition-all duration-300 ${collapsed ? 'w-14' : 'w-64'} min-h-screen`}>
      <div className="flex items-center justify-between px-3 py-4 border-b border-slate-800/70">
        {!collapsed && <span className="text-slate-300 text-sm font-semibold tracking-wide">Sessions</span>}
        <button onClick={onToggle} className="text-slate-500 hover:text-slate-300 transition-colors ml-auto">
          {collapsed ? <PanelLeftOpen className="w-5 h-5" /> : <PanelLeftClose className="w-5 h-5" />}
        </button>
      </div>
      <div className="px-2 py-3 border-b border-slate-800/70">
        <button
          onClick={onNew}
          className={`flex items-center gap-2 w-full rounded-xl px-2 py-2 text-sm text-slate-400 hover:text-white hover:bg-slate-800/60 transition-all ${collapsed ? 'justify-center' : ''}`}
        >
          <Plus className="w-4 h-4 flex-shrink-0" />
          {!collapsed && <span>New Session</span>}
        </button>
      </div>
      <div className="flex-1 overflow-y-auto py-2 space-y-0.5 px-2">
        {sessions.map(s => (
          <div
            key={s.id}
            onClick={() => onSelect(s.id)}
            className={`group flex items-center gap-2 rounded-xl px-2 py-2.5 cursor-pointer transition-all ${s.id === activeId ? 'bg-blue-600/20 border border-blue-500/30 text-white' : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200 border border-transparent'}`}
          >
            {collapsed ? (
              <div className="w-6 h-6 rounded-lg bg-slate-700/60 flex items-center justify-center flex-shrink-0">
                <Mic className="w-3 h-3 text-slate-400" />
              </div>
            ) : editingId === s.id ? (
              <div className="flex items-center gap-1 flex-1 min-w-0" onClick={e => e.stopPropagation()}>
                <input
                  autoFocus
                  value={editValue}
                  onChange={e => setEditValue(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') commitEdit(s.id); if (e.key === 'Escape') cancelEdit(); }}
                  className="flex-1 bg-slate-800 border border-blue-500/50 rounded-lg px-2 py-0.5 text-xs text-white outline-none min-w-0"
                />
                <button onClick={() => commitEdit(s.id)} className="text-green-400 hover:text-green-300"><Check className="w-3 h-3" /></button>
                <button onClick={cancelEdit} className="text-red-400 hover:text-red-300"><X className="w-3 h-3" /></button>
              </div>
            ) : (
              <>
                <div className="w-5 h-5 rounded-md bg-slate-700/60 flex items-center justify-center flex-shrink-0">
                  <Mic className="w-2.5 h-2.5 text-slate-400" />
                </div>
                <span className="flex-1 text-xs truncate">{s.label}</span>
                <button onClick={e => startEdit(e, s)} className="opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0 text-slate-500 hover:text-slate-300">
                  <Pencil className="w-3 h-3" />
                </button>
              </>
            )}
          </div>
        ))}
      </div>
    </aside>
  );
};

// ─── Main Tool Component ─────────────────────────────────────────────────────
const ToolPage = ({ session, onUpdate, onGoHome }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isToolLoading, setIsToolLoading] = useState(false);
  const [notification, setNotification] = useState(null);

  const s = session;
  const update = (patch) => onUpdate(s.id, patch);

  const showNotification = (type, message) => {
    setNotification({ type, message });
    setTimeout(() => setNotification(null), 4000);
  };

  const deriveLabelFromText = (text) => {
    const words = text.trim().split(/\s+/).slice(0, 5).join(' ');
    return words || 'New Session';
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks = [];
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        update({ liveAudio: blob, uploadedFile: null, manualText: '' });
        stream.getTracks().forEach(t => t.stop());
      };
      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
    } catch { showNotification('error', 'Could not access microphone'); }
  };

  const stopRecording = () => { if (mediaRecorder && mediaRecorder.state !== 'inactive') { mediaRecorder.stop(); setIsRecording(false); } };
  const toggleRecording = () => { if (isRecording) stopRecording(); else startRecording(); };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) update({ uploadedFile: file, liveAudio: null, manualText: '', transcriptionResult: null, processedResults: [] });
  };

  const transcribeAudio = async () => {
    const audioToTranscribe = s.liveAudio || s.uploadedFile;
    if (!audioToTranscribe) { showNotification('error', 'Please provide audio file'); return; }
    setIsTranscribing(true);
    try {
      const formData = new FormData();
      formData.append('file', audioToTranscribe, s.liveAudio ? 'recording.webm' : s.uploadedFile.name);
      const result = await apiFetch(`${API_BASE_URL}/transcribe/audio`, { method: 'POST', body: formData });
      const label = deriveLabelFromText(result.data.transcription);
      update({ transcriptionResult: result.data, editedTranscription: result.data.transcription, processedResults: [], toolResult: null, selectedTool: '', label });
      showNotification('success', result.message);
    } catch (err) { showNotification('error', err.message); } finally { setIsTranscribing(false); }
  };

  const transcribeAndProcess = async () => {
    const audioToProcess = s.liveAudio || s.uploadedFile;
    if (!audioToProcess) { showNotification('error', 'Please provide audio file'); return; }
    setIsTranscribing(true);
    try {
      const formData = new FormData();
      formData.append('file', audioToProcess, s.liveAudio ? 'recording.webm' : s.uploadedFile.name);
      const result = await apiFetch(`${API_BASE_URL}/transcribe-and-process/audio`, { method: 'POST', body: formData });
      const label = deriveLabelFromText(result.transcription.transcription);
      const newEntry = {
        id: Date.now(),
        text: result.preprocessed.preprocessed_transcription,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        fullResult: result.preprocessed,
      };
      update({ transcriptionResult: result.transcription, editedTranscription: result.transcription.transcription, processedResults: [newEntry], toolResult: null, selectedTool: '', label });
      showNotification('success', 'Transcribed and processed successfully');
    } catch (err) { showNotification('error', err.message); } finally { setIsTranscribing(false); }
  };

  const submitDirectText = async () => {
    if (!s.manualName.trim() || !s.manualText.trim()) { showNotification('error', 'Please provide name and text'); return; }
    setIsTranscribing(true);
    try {
      const result = await apiFetch(`${API_BASE_URL}/transcribe/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: s.manualName, transcription: s.manualText }),
      });
      const label = deriveLabelFromText(result.data.transcription);
      update({ transcriptionResult: result.data, editedTranscription: result.data.transcription, manualName: '', manualText: '', processedResults: [], toolResult: null, selectedTool: '', label });
      showNotification('success', result.message);
    } catch (err) { showNotification('error', err.message); } finally { setIsTranscribing(false); }
  };

  const processTranscription = async () => {
    if (!s.transcriptionResult) return;
    setIsProcessing(true);
    try {
      const result = await apiFetch(`${API_BASE_URL}/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: s.transcriptionResult.id, name: s.transcriptionResult.name, transcription: s.editedTranscription }),
      });
      const newEntry = {
        id: Date.now(),
        text: result.data.preprocessed_transcription,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        fullResult: result.data,
      };
      update({ processedResults: [...s.processedResults, newEntry] });
    } catch (err) { showNotification('error', err.message); } finally { setIsProcessing(false); }
  };

  const latestProcessed = s.processedResults.length > 0 ? s.processedResults[s.processedResults.length - 1] : null;

  const downloadAllVersions = () => {
    if (!s.processedResults.length) return;
    const lines = [];
    lines.push(`# VoxFlow AI — Processed Transcription Versions`);
    lines.push(`**Session:** ${s.label}`);
    lines.push(`**Original transcription:** ${s.transcriptionResult?.name || 'N/A'}`);
    lines.push(`**Total versions:** ${s.processedResults.length}`);
    lines.push('');
    s.processedResults.forEach((v, i) => {
      lines.push(`---`);
      lines.push(`## Version ${i + 1} — ${v.timestamp}`);
      lines.push('');
      lines.push(v.text);
      lines.push('');
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `voxflow-${s.label.replace(/\s+/g, '-').toLowerCase()}-versions.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const triggerEmail = async () => {
    setIsToolLoading(true);
    try {
      await apiFetch(`${API_BASE_URL}/send-email`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...s.emailData, processed_data: latestProcessed.text }),
      });
      showNotification('success', 'Email sent!');
    } catch (err) { showNotification('error', err.message); } finally { setIsToolLoading(false); }
  };

  const triggerExtraction = async () => {
    setIsToolLoading(true);
    try {
      const result = await apiFetch(`${API_BASE_URL}/extract-text`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ processed_data: latestProcessed.text }),
      });
      update({ toolResult: { type: 'extraction', data: result.data } });
    } catch (err) { showNotification('error', err.message); } finally { setIsToolLoading(false); }
  };

  const triggerTranslation = async () => {
    setIsToolLoading(true);
    try {
      const result = await apiFetch(`${API_BASE_URL}/translate`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language: s.targetLang, processed_data: latestProcessed.text }),
      });
      update({ toolResult: { type: 'translation', data: result.translated_data } });
    } catch (err) { showNotification('error', err.message); } finally { setIsToolLoading(false); }
  };

  const Divider = () => (
    <div className="flex items-center gap-4 my-2">
      <div className="flex-1 h-px bg-slate-700/50" />
      <span className="text-slate-500 font-bold text-xs tracking-widest">OR</span>
      <div className="flex-1 h-px bg-slate-700/50" />
    </div>
  );

  const audioButtonLabel = s.mode === 'auto'
    ? (isTranscribing ? <><Loader2 className="w-4 h-4 animate-spin" /> Processing...</> : <><Sparkles className="w-4 h-4" /> Transcribe & Process</>)
    : (isTranscribing ? <><Loader2 className="w-4 h-4 animate-spin" /> Transcribing...</> : 'Transcribe');

  const handleAudioAction = s.mode === 'auto' ? transcribeAndProcess : transcribeAudio;

  return (
    <div className="flex-1 min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-4 md:p-8 overflow-y-auto relative">

      {notification && (
        <div className={`fixed top-8 right-8 z-50 px-6 py-4 rounded-xl shadow-2xl flex items-center gap-3 ${notification.type === 'success' ? 'bg-green-600' : 'bg-red-600'}`}>
          <span className="text-white font-medium">{notification.message}</span>
        </div>
      )}

      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-10">
          <button onClick={onGoHome} className="flex items-center gap-2 text-slate-500 hover:text-slate-300 transition-colors text-sm">
            <ChevronLeft className="w-4 h-4" /> Home
          </button>
          <div className="text-center">
            <h1 className="text-3xl md:text-4xl font-bold text-white tracking-tight">VoxFlow AI</h1>
            <p className="text-slate-400 text-sm mt-1">Transcribe and preprocess audio with AI precision</p>
          </div>
          <div className="w-16" />
        </div>

        {/* Input Methods */}
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-3xl shadow-2xl p-6 border border-slate-700/50 mb-8">

          {/* Mode Toggle */}
          <div className="flex items-center justify-end mb-5">
            <div className="flex items-center gap-1 bg-slate-950/60 border border-slate-700/60 rounded-xl p-1">
              <button
                onClick={() => update({ mode: 'manual' })}
                className={`px-4 py-1.5 rounded-lg text-xs font-semibold transition-all ${s.mode === 'manual' ? 'bg-blue-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
              >
                Manual
              </button>
              <button
                onClick={() => update({ mode: 'auto' })}
                className={`px-4 py-1.5 rounded-lg text-xs font-semibold transition-all ${s.mode === 'auto' ? 'bg-purple-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
              >
                Auto
              </button>
            </div>
          </div>

          <div className="flex flex-col gap-4">

            {/* Live Recording */}
            <div className="bg-slate-900/50 rounded-2xl p-5 border border-slate-700/50 text-center">
              <div className="flex items-center gap-3 mb-4 text-left">
                <Mic className="w-5 h-5 text-blue-400" />
                <h3 className="text-white font-semibold">Live Recording</h3>
              </div>
              <button onClick={toggleRecording} className={`w-full border-2 border-dashed rounded-xl p-6 transition-all ${isRecording ? 'border-red-500 bg-red-500/10' : 'border-slate-600 hover:border-blue-500'}`}>
                <Mic className={`w-8 h-8 mx-auto mb-2 ${isRecording ? 'text-red-500 animate-pulse' : 'text-slate-500'}`} />
                <p className={`text-sm font-medium ${isRecording ? 'text-red-400' : 'text-slate-400'}`}>{isRecording ? 'Recording... Click to Stop' : 'Click to Record'}</p>
              </button>
              {s.liveAudio && (
                <div className="flex justify-center mt-4">
                  <button onClick={handleAudioAction} disabled={isTranscribing} className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg text-sm font-semibold hover:bg-blue-500 transition-colors">
                    {audioButtonLabel}
                  </button>
                </div>
              )}
            </div>

            <Divider />

            {/* Upload Audio */}
            <div className="bg-slate-900/50 rounded-2xl p-5 border border-slate-700/50 text-center">
              <div className="flex items-center gap-3 mb-4 text-left"><Upload className="w-5 h-5 text-blue-400" /><h3 className="text-white font-semibold">Upload Audio</h3></div>
              <label className="block cursor-pointer">
                <input type="file" accept="audio/*" onChange={handleFileUpload} className="hidden" />
                <div className="border-2 border-dashed border-slate-600 rounded-xl p-6 hover:border-blue-500 transition-colors">
                  <Upload className="w-8 h-8 text-slate-500 mx-auto mb-2" />
                  <p className="text-slate-400 text-sm">{s.uploadedFile ? s.uploadedFile.name : 'Choose audio file'}</p>
                </div>
              </label>
              {s.uploadedFile && (
                <div className="flex justify-center mt-4">
                  <button onClick={handleAudioAction} disabled={isTranscribing} className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg text-sm font-semibold hover:bg-blue-500 transition-colors">
                    {audioButtonLabel}
                  </button>
                </div>
              )}
            </div>

            {/* Direct Text — manual only */}
            {s.mode === 'manual' && (
              <>
                <Divider />
                <div className="bg-slate-900/50 rounded-2xl p-5 border border-slate-700/50 text-center">
                  <div className="flex items-center gap-3 mb-4 text-left"><FileText className="w-5 h-5 text-blue-400" /><h3 className="text-white font-semibold">Direct Text</h3></div>
                  <input type="text" value={s.manualName} onChange={(e) => update({ manualName: e.target.value })} className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white text-sm mb-3" placeholder="Session Name" />
                  <textarea value={s.manualText} onChange={(e) => update({ manualText: e.target.value })} className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white text-sm" rows={4} placeholder="Paste your raw text here..." />
                  {s.manualText && s.manualName && (
                    <div className="flex justify-center mt-4">
                      <button onClick={submitDirectText} disabled={isTranscribing} className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg text-sm font-semibold hover:bg-blue-500 transition-colors">
                        {isTranscribing ? <><Loader2 className="w-4 h-4 animate-spin" /> Preparing...</> : 'Save & Prepare'}
                      </button>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>

        {/* Transcription Editor — manual only */}
        {s.mode === 'manual' && s.transcriptionResult && (
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-3xl p-6 border border-slate-700/50 mb-8">
            <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-3"><FileText className="w-6 h-6 text-blue-400" />Original Transcription</h2>
            <textarea value={s.editedTranscription} onChange={(e) => update({ editedTranscription: e.target.value })} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-slate-200 font-mono text-sm mb-6" rows={8} />
            <div className="flex justify-center">
              <button onClick={processTranscription} disabled={isProcessing} className="px-8 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl font-semibold flex items-center gap-3 transition-transform active:scale-95">
                {isProcessing ? <><Loader2 className="w-5 h-5 animate-spin" /> Processing...</> : <><Sparkles className="w-5 h-5" /> Process with AI</>}
              </button>
            </div>
          </div>
        )}

        {/* Processed Results History */}
        {s.processedResults.length > 0 && (
          <div className="space-y-4 mb-8">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold text-white flex items-center gap-3">
                <Sparkles className="w-6 h-6 text-purple-400" />
                AI Results
                <span className="text-xs font-normal text-slate-500 bg-slate-800 px-2 py-0.5 rounded-full">{s.processedResults.length} version{s.processedResults.length > 1 ? 's' : ''}</span>
              </h2>
              <button
                onClick={downloadAllVersions}
                className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 hover:text-white rounded-lg text-xs font-semibold transition-all"
              >
                <Download className="w-3.5 h-3.5" />
                Download All Versions
              </button>
            </div>

            {s.processedResults.map((v, i) => (
              <div key={v.id} className={`rounded-3xl p-6 border-2 relative ${i === s.processedResults.length - 1 ? 'bg-gradient-to-br from-purple-900/20 to-blue-900/20 border-purple-500/40' : 'bg-slate-900/60 border-slate-700/40'}`}>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded-full">v{i + 1}</span>
                    <span className="text-xs text-slate-500">{v.timestamp}</span>
                    {i === s.processedResults.length - 1 && (
                      <span className="text-xs font-semibold text-purple-300 bg-purple-500/20 px-2 py-0.5 rounded-full">Latest</span>
                    )}
                  </div>
                  <button
                    onClick={() => { navigator.clipboard.writeText(v.text); showNotification('success', 'Copied!'); }}
                    className="flex items-center gap-2 px-3 py-1.5 bg-purple-600/80 hover:bg-purple-600 text-white rounded-lg text-xs font-semibold transition-colors"
                  >
                    <Copy className="w-3 h-3" /> Copy
                  </button>
                </div>
                <div className={`rounded-lg px-4 py-4 text-slate-200 font-mono text-sm leading-relaxed border ${i === s.processedResults.length - 1 ? 'bg-slate-950/80 border-purple-500/10' : 'bg-slate-950/50 border-slate-800/50'}`}>
                  {v.text}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Tools */}
        {latestProcessed && (
          <div className="bg-slate-900 rounded-2xl p-6 border border-slate-700 mb-8">
            <h3 className="text-white font-semibold mb-4">Next Steps</h3>
            <select className="w-full bg-slate-800 border border-slate-700 text-white rounded-xl px-4 py-3 outline-none mb-4" value={s.selectedTool} onChange={(e) => update({ selectedTool: e.target.value, toolResult: null })}>
              <option value="">Select a tool...</option>
              <option value="email">Send Email</option>
              <option value="extract">Extract Keywords & Keypoints</option>
              <option value="translate">Translate</option>
            </select>

            {s.selectedTool === 'email' && (
              <div className="space-y-4 bg-slate-800/50 p-5 rounded-xl border border-slate-700">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <input type="email" placeholder="Recipient" className="bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white text-sm" value={s.emailData.to} onChange={e => update({ emailData: { ...s.emailData, to: e.target.value } })} />
                  <input type="text" placeholder="Sender" className="bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white text-sm" value={s.emailData.sender} onChange={e => update({ emailData: { ...s.emailData, sender: e.target.value } })} />
                </div>
                <input type="text" placeholder="Subject" className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white text-sm" value={s.emailData.subject} onChange={e => update({ emailData: { ...s.emailData, subject: e.target.value } })} />
                <textarea placeholder="Message" className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white text-sm" rows={3} value={s.emailData.user_message} onChange={e => update({ emailData: { ...s.emailData, user_message: e.target.value } })} />
                <div className="flex justify-center">
                  <button onClick={triggerEmail} disabled={isToolLoading} className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg text-sm font-semibold">
                    {isToolLoading ? <><Loader2 className="w-4 h-4 animate-spin" /> Sending...</> : 'Send Email'}
                  </button>
                </div>
              </div>
            )}

            {s.selectedTool === 'translate' && (
              <div className="space-y-4 bg-slate-800/50 p-5 rounded-xl border border-slate-700">
                <select className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white text-sm" value={s.targetLang} onChange={e => update({ targetLang: e.target.value })}>
                  {Object.entries(SUPPORTED_LANGS).map(([name, code]) => <option key={code} value={code}>{name}</option>)}
                </select>
                <div className="flex justify-center">
                  <button onClick={triggerTranslation} disabled={isToolLoading} className="flex items-center gap-2 px-6 py-2 bg-teal-600 text-white rounded-lg text-sm font-semibold">
                    {isToolLoading ? <><Loader2 className="w-4 h-4 animate-spin" /> Translating...</> : 'Translate'}
                  </button>
                </div>
              </div>
            )}

            {s.selectedTool === 'extract' && (
              <div className="flex justify-center p-5 bg-slate-800/50 rounded-xl border border-slate-700">
                <button onClick={triggerExtraction} disabled={isToolLoading} className="flex items-center gap-2 px-6 py-2 bg-indigo-600 text-white rounded-lg text-sm font-semibold">
                  {isToolLoading ? <><Loader2 className="w-4 h-4 animate-spin" /> Extracting...</> : 'Extract Insights'}
                </button>
              </div>
            )}

            {s.toolResult && (
              <div className="mt-6 p-5 bg-slate-950 rounded-xl border border-blue-500/30 text-sm">
                {s.toolResult.type === 'extraction' ? (
                  <div className="space-y-4">
                    <div className="flex flex-wrap gap-2">{s.toolResult.data.keywords.map((k, i) => <span key={i} className="bg-blue-500/10 text-blue-300 px-3 py-1 rounded-full text-xs">{k}</span>)}</div>
                    <ul className="list-disc list-inside text-slate-300 space-y-1">{s.toolResult.data.keypoints.map((p, i) => <li key={i}>{p}</li>)}</ul>
                  </div>
                ) : <p className="text-slate-300 whitespace-pre-wrap">{s.toolResult.data}</p>}
              </div>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="mt-4 mb-8 pt-8 border-t border-slate-800 text-center">
          <p className="text-slate-400 text-sm mb-4">I am Alee, a 17 year old Aspiring AI Engineer</p>
          <a href="https://github.com/AleeCodeAI/VoxFlow_AI" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 px-4 py-2 bg-slate-900 border border-slate-700 rounded-full text-slate-300 text-sm">
            <Github className="w-4 h-4" /> <span>@AleeCodeAI</span>
          </a>
        </div>
      </div>
    </div>
  );
};

// ─── Root App ────────────────────────────────────────────────────────────────
const App = () => {
  const [page, setPage] = useState('home');
  const [sessions, setSessions] = useState([createSession('New Session')]);
  const [activeId, setActiveId] = useState(sessions[0].id);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const activeSession = sessions.find(s => s.id === activeId) || sessions[0];

  const updateSession = (id, patch) => {
    setSessions(prev => prev.map(s => s.id === id ? { ...s, ...patch } : s));
  };

  const renameSession = (id, label) => {
    setSessions(prev => prev.map(s => s.id === id ? { ...s, label } : s));
  };

  const newSession = () => {
    const s = createSession('New Session');
    setSessions(prev => [s, ...prev]);
    setActiveId(s.id);
  };

  if (page === 'home') {
    return <HomePage onStart={() => setPage('tool')} />;
  }

  return (
    <div className="flex min-h-screen bg-slate-950">
      <Sidebar
        sessions={sessions}
        activeId={activeId}
        onSelect={setActiveId}
        onNew={newSession}
        onRename={renameSession}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(p => !p)}
      />
      <ToolPage
        session={activeSession}
        onUpdate={updateSession}
        onGoHome={() => setPage('home')}
      />
    </div>
  );
};

export default App;