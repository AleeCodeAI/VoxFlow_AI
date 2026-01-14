import React, { useState } from 'react';
import { Copy, Mic, Upload, FileText, Sparkles, Loader2, CheckCircle, XCircle, Github, Mail, Languages, ListTree, ChevronDown } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const SUPPORTED_LANGS = {
  'Albanian': 'sq', 'Arabic': 'ar', 'Armenian': 'hy', 'Assamese': 'as', 'Azerbaijani': 'az', 'Bengali': 'bn', 'Bhojpuri': 'bho', 'Bosnian': 'bs', 'Bulgarian': 'bg', 'Catalan': 'ca', 'Chinese (simplified)': 'zh-CN', 'Chinese (traditional)': 'zh-TW', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Dutch': 'nl', 'English': 'en', 'Estonian': 'et', 'Filipino': 'tl', 'Finnish': 'fi', 'French': 'fr', 'Georgian': 'ka', 'German': 'de', 'Greek': 'el', 'Gujarati': 'gu', 'Hindi': 'hi', 'Hungarian': 'hu', 'Icelandic': 'is', 'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it', 'Japanese': 'ja', 'Javanese': 'jw', 'Kannada': 'kn', 'Kazakh': 'kk', 'Khmer': 'km', 'Korean': 'ko', 'Lao': 'lo', 'Latvian': 'lv', 'Lithuanian': 'lt', 'Macedonian': 'mk', 'Malay': 'ms', 'Malayalam': 'ml', 'Marathi': 'mr', 'Mongolian': 'mn', 'Myanmar': 'my', 'Nepali': 'ne', 'Norwegian': 'no', 'Odia (oriya)': 'or', 'Pashto': 'ps', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt', 'Punjabi': 'pa', 'Romanian': 'ro', 'Russian': 'ru', 'Sanskrit': 'sa', 'Serbian': 'sr', 'Sindhi': 'sd', 'Sinhala': 'si', 'Slovak': 'sk', 'Slovenian': 'sl', 'Spanish': 'es', 'Swedish': 'sv', 'Tamil': 'ta', 'Telugu': 'te', 'Thai': 'th', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uzbek': 'uz', 'Vietnamese': 'vi', 'Welsh': 'cy'
};

const AudioPreprocessorUI = () => {
  const [liveAudio, setLiveAudio] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [manualText, setManualText] = useState('');
  const [manualName, setManualName] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcriptionResult, setTranscriptionResult] = useState(null);
  const [editedTranscription, setEditedTranscription] = useState('');
  const [processedResult, setProcessedResult] = useState(null);
  const [notification, setNotification] = useState(null);

  const [selectedTool, setSelectedTool] = useState('');
  const [isToolLoading, setIsToolLoading] = useState(false);
  const [toolResult, setToolResult] = useState(null);
  const [emailData, setEmailData] = useState({ to: '', subject: '', user_message: '', sender: '' });
  const [targetLang, setTargetLang] = useState('es');

  const showNotification = (type, message) => {
    setNotification({ type, message });
    setTimeout(() => setNotification(null), 4000);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks = [];
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        setLiveAudio(blob);
        stream.getTracks().forEach(track => track.stop());
        setUploadedFile(null);
        setManualText('');
      };
      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
    } catch (error) { showNotification('error', 'Could not access microphone'); }
  };

  const stopRecording = () => { if (mediaRecorder && mediaRecorder.state !== 'inactive') { mediaRecorder.stop(); setIsRecording(false); } };
  const toggleRecording = () => { if (isRecording) stopRecording(); else startRecording(); };
  
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) { setUploadedFile(file); setLiveAudio(null); setManualText(''); setTranscriptionResult(null); setProcessedResult(null); }
  };

  const transcribeAudio = async () => {
    const audioToTranscribe = liveAudio || uploadedFile;
    if (!audioToTranscribe) { showNotification('error', 'Please provide audio file'); return; }
    setIsTranscribing(true);
    try {
      const formData = new FormData();
      formData.append('file', audioToTranscribe, liveAudio ? 'recording.webm' : uploadedFile.name);
      const response = await fetch(`${API_BASE_URL}/transcribe/audio`, { method: 'POST', body: formData });
      const result = await response.json();
      if (response.ok) {
        setTranscriptionResult(result.data);
        setEditedTranscription(result.data.transcription);
        showNotification('success', result.message);
      } else { showNotification('error', result.detail?.message || 'Transcription failed'); }
    } catch (error) { showNotification('error', `Error: ${error.message}`); } finally { setIsTranscribing(false); }
  };

  const submitDirectText = async () => {
    if (!manualName.trim() || !manualText.trim()) { showNotification('error', 'Please provide name and text'); return; }
    setIsTranscribing(true);
    try {
      const response = await fetch(`${API_BASE_URL}/transcribe/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: manualName, transcription: manualText }),
      });
      const result = await response.json();
      if (response.ok) {
        setTranscriptionResult(result.data);
        setEditedTranscription(result.data.transcription);
        showNotification('success', result.message);
        setManualName(''); setManualText('');
      }
    } finally { setIsTranscribing(false); }
  };

  const processTranscription = async () => {
    if (!transcriptionResult) return;
    setIsProcessing(true);
    try {
      const response = await fetch(`${API_BASE_URL}/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: transcriptionResult.id, name: transcriptionResult.name, transcription: editedTranscription }),
      });
      const result = await response.json();
      if (response.ok) setProcessedResult(result.data);
    } finally { setIsProcessing(false); }
  };

  const triggerEmail = async () => {
    setIsToolLoading(true);
    try {
      await fetch(`${API_BASE_URL}/send-email`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...emailData, processed_data: processedResult.preprocessed_transcription }),
      });
      showNotification('success', 'Email sent!');
    } finally { setIsToolLoading(false); }
  };

  const triggerExtraction = async () => {
    setIsToolLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/extract-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ processed_data: processedResult.preprocessed_transcription }),
      });
      const result = await response.json();
      if (response.ok) setToolResult({ type: 'extraction', data: result.data });
    } finally { setIsToolLoading(false); }
  };

  const triggerTranslation = async () => {
    setIsToolLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language: targetLang, processed_data: processedResult.preprocessed_transcription }),
      });
      const result = await response.json();
      if (response.ok) setToolResult({ type: 'translation', data: result.translated_data });
    } finally { setIsToolLoading(false); }
  };

  const Divider = () => (
    <div className="flex items-center gap-4 my-2">
      <div className="flex-1 h-px bg-slate-700/50"></div>
      <span className="text-slate-500 font-bold text-xs tracking-widest">OR</span>
      <div className="flex-1 h-px bg-slate-700/50"></div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-4 md:p-8">
      <div className="max-w-3xl mx-auto">
        
        {notification && (
          <div className={`fixed top-8 right-8 z-50 px-6 py-4 rounded-xl shadow-2xl flex items-center gap-3 ${notification.type === 'success' ? 'bg-green-600' : 'bg-red-600'}`}>
            <span className="text-white font-medium">{notification.message}</span>
          </div>
        )}

        <div className="text-center mb-10">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-3 tracking-tight">VoxFlow AI</h1>
          <p className="text-slate-400 text-lg">Transcribe and preprocess audio with AI precision</p>
        </div>

        {/* Input Methods */}
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-3xl shadow-2xl p-6 border border-slate-700/50 mb-8">
          <div className="flex flex-col gap-4">
            
            <div className="bg-slate-900/50 rounded-2xl p-5 border border-slate-700/50 text-center">
              <div className="flex items-center gap-3 mb-4 text-left">
                <Mic className="w-5 h-5 text-blue-400" />
                <h3 className="text-white font-semibold">Live Recording</h3>
              </div>
              <button onClick={toggleRecording} className={`w-full border-2 border-dashed rounded-xl p-6 transition-all ${isRecording ? 'border-red-500 bg-red-500/10' : 'border-slate-600 hover:border-blue-500'}`}>
                <Mic className={`w-8 h-8 mx-auto mb-2 ${isRecording ? 'text-red-500 animate-pulse' : 'text-slate-500'}`} />
                <p className={`text-sm font-medium ${isRecording ? 'text-red-400' : 'text-slate-400'}`}>{isRecording ? 'Recording...' : 'Click to Record'}</p>
              </button>
              {liveAudio && (
                <div className="flex justify-center mt-4">
                  <button onClick={transcribeAudio} disabled={isTranscribing} className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg text-sm font-semibold hover:bg-blue-500 transition-colors">
                    {isTranscribing ? <><Loader2 className="w-4 h-4 animate-spin" /> Transcribing...</> : 'Transcribe Recording'}
                  </button>
                </div>
              )}
            </div>

            <Divider />

            <div className="bg-slate-900/50 rounded-2xl p-5 border border-slate-700/50 text-center">
              <div className="flex items-center gap-3 mb-4 text-left"><Upload className="w-5 h-5 text-blue-400" /><h3 className="text-white font-semibold">Upload Audio</h3></div>
              <label className="block cursor-pointer">
                <input type="file" accept="audio/*" onChange={handleFileUpload} className="hidden" />
                <div className="border-2 border-dashed border-slate-600 rounded-xl p-6 hover:border-blue-500 transition-colors">
                  <Upload className="w-8 h-8 text-slate-500 mx-auto mb-2" />
                  <p className="text-slate-400 text-sm">{uploadedFile ? uploadedFile.name : 'Choose audio file'}</p>
                </div>
              </label>
              {uploadedFile && (
                <div className="flex justify-center mt-4">
                  <button onClick={transcribeAudio} disabled={isTranscribing} className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg text-sm font-semibold hover:bg-blue-500 transition-colors">
                    {isTranscribing ? <><Loader2 className="w-4 h-4 animate-spin" /> Transcribing...</> : 'Transcribe File'}
                  </button>
                </div>
              )}
            </div>

            <Divider />

            <div className="bg-slate-900/50 rounded-2xl p-5 border border-slate-700/50 text-center">
              <div className="flex items-center gap-3 mb-4 text-left"><FileText className="w-5 h-5 text-blue-400" /><h3 className="text-white font-semibold">Direct Text</h3></div>
              <input type="text" value={manualName} onChange={(e) => setManualName(e.target.value)} className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white text-sm mb-3" placeholder="Session Name" />
              <textarea value={manualText} onChange={(e) => setManualText(e.target.value)} className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white text-sm" rows={4} placeholder="Paste your raw text here..." />
              {manualText && manualName && (
                <div className="flex justify-center mt-4">
                  <button onClick={submitDirectText} disabled={isTranscribing} className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg text-sm font-semibold hover:bg-blue-500 transition-colors">
                    {isTranscribing ? <><Loader2 className="w-4 h-4 animate-spin" /> Preparing...</> : 'Save & Prepare'}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Process Steps */}
        {transcriptionResult && (
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-3xl p-6 border border-slate-700/50 mb-8">
            <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-3"><FileText className="w-6 h-6 text-blue-400" />Original Transcription</h2>
            <textarea value={editedTranscription} onChange={(e) => setEditedTranscription(e.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-slate-200 font-mono text-sm mb-6" rows={8} />
            <div className="flex justify-center">
              <button onClick={processTranscription} disabled={isProcessing} className="px-8 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl font-semibold flex items-center gap-3 transition-transform active:scale-95">
                {isProcessing ? <><Loader2 className="w-5 h-5 animate-spin" /> Processing...</> : <><Sparkles className="w-5 h-5" /> Process with AI</>}
              </button>
            </div>
          </div>
        )}

        {processedResult && (
          <div className="space-y-6">
            <div className="bg-gradient-to-br from-purple-900/20 to-blue-900/20 rounded-3xl p-6 border-2 border-purple-500/40 relative">
              <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-3"><Sparkles className="w-6 h-6 text-purple-400" />AI Result</h2>
              <div className="bg-slate-950/80 rounded-lg px-4 py-4 text-slate-200 font-mono text-sm leading-relaxed border border-purple-500/10 mb-4">
                {processedResult.preprocessed_transcription}
              </div>
              <div className="flex justify-end">
                <button onClick={() => { navigator.clipboard.writeText(processedResult.preprocessed_transcription); showNotification('success', 'Copied!'); }} className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg text-xs font-semibold">
                  <Copy className="w-3 h-3" /> Copy Result
                </button>
              </div>
            </div>

            <div className="bg-slate-900 rounded-2xl p-6 border border-slate-700">
              <h3 className="text-white font-semibold mb-4 flex items-center gap-2">Next Steps</h3>
              <select className="w-full bg-slate-800 border border-slate-700 text-white rounded-xl px-4 py-3 outline-none mb-4" value={selectedTool} onChange={(e) => { setSelectedTool(e.target.value); setToolResult(null); }}>
                <option value="">Select a tool...</option>
                <option value="email">Send Email</option>
                <option value="extract">Extract Keywords & Keypoints</option>
                <option value="translate">Translate</option>
              </select>

              {selectedTool === 'email' && (
                <div className="space-y-4 bg-slate-800/50 p-5 rounded-xl border border-slate-700">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <input type="email" placeholder="Recipient" className="bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white text-sm" value={emailData.to} onChange={e => setEmailData({...emailData, to: e.target.value})} />
                    <input type="text" placeholder="Sender" className="bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white text-sm" value={emailData.sender} onChange={e => setEmailData({...emailData, sender: e.target.value})} />
                  </div>
                  <input type="text" placeholder="Subject" className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white text-sm" value={emailData.subject} onChange={e => setEmailData({...emailData, subject: e.target.value})} />
                  <textarea placeholder="Message" className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white text-sm" rows={3} value={emailData.user_message} onChange={e => setEmailData({...emailData, user_message: e.target.value})} />
                  <div className="flex justify-center">
                    <button onClick={triggerEmail} disabled={isToolLoading} className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg text-sm font-semibold">
                      {isToolLoading ? <><Loader2 className="w-4 h-4 animate-spin" /> Sending...</> : 'Send Email'}
                    </button>
                  </div>
                </div>
              )}

              {selectedTool === 'translate' && (
                <div className="space-y-4 bg-slate-800/50 p-5 rounded-xl border border-slate-700">
                  <select className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white text-sm" value={targetLang} onChange={e => setTargetLang(e.target.value)}>
                    {Object.entries(SUPPORTED_LANGS).map(([name, code]) => <option key={code} value={code}>{name}</option>)}
                  </select>
                  <div className="flex justify-center">
                    <button onClick={triggerTranslation} disabled={isToolLoading} className="flex items-center gap-2 px-6 py-2 bg-teal-600 text-white rounded-lg text-sm font-semibold">
                      {isToolLoading ? <><Loader2 className="w-4 h-4 animate-spin" /> Translating...</> : 'Translate'}
                    </button>
                  </div>
                </div>
              )}

              {selectedTool === 'extract' && (
                <div className="flex justify-center p-5 bg-slate-800/50 rounded-xl border border-slate-700">
                  <button onClick={triggerExtraction} disabled={isToolLoading} className="flex items-center gap-2 px-6 py-2 bg-indigo-600 text-white rounded-lg text-sm font-semibold">
                    {isToolLoading ? <><Loader2 className="w-4 h-4 animate-spin" /> Extracting...</> : 'Extract Insights'}
                  </button>
                </div>
              )}

              {toolResult && (
                <div className="mt-6 p-5 bg-slate-950 rounded-xl border border-blue-500/30 text-sm">
                  {toolResult.type === 'extraction' ? (
                    <div className="space-y-4">
                      <div className="flex flex-wrap gap-2">{toolResult.data.keywords.map((k, i) => <span key={i} className="bg-blue-500/10 text-blue-300 px-3 py-1 rounded-full text-xs">{k}</span>)}</div>
                      <ul className="list-disc list-inside text-slate-300 space-y-1">{toolResult.data.keypoints.map((p, i) => <li key={i}>{p}</li>)}</ul>
                    </div>
                  ) : <p className="text-slate-300 whitespace-pre-wrap">{toolResult.data}</p>}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Footer Description */}
        <div className="mt-12 mb-8 pt-8 border-t border-slate-800 text-center">
          <p className="text-slate-400 text-sm mb-4">I am Alee a 17 years old Aspiring AI Engineer</p>
          <a href="https://github.com/AleeCodeAI" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 px-4 py-2 bg-slate-900 border border-slate-700 rounded-full text-slate-300 text-sm">
            <Github className="w-4 h-4" /> <span>@AleeCodeAI</span>
          </a>
        </div>
      </div>
    </div>
  );
};

export default AudioPreprocessorUI;