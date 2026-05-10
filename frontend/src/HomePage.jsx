import React, { useEffect, useRef } from 'react';
import { Mic, Code2, SlidersHorizontal, Globe, Folder } from 'lucide-react';

const HomePage = ({ onStart }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animId;
    let t = 0;

    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const drawWave = (yBase, amp, freq, speed, opacity, color) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.globalAlpha = opacity;
      ctx.lineWidth = 1.5;
      for (let x = 0; x <= canvas.width; x += 2) {
        const y = yBase + Math.sin((x * freq + t * speed)) * amp * Math.sin(x * 0.003 + t * 0.2);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    };

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const cy = canvas.height / 2;
      drawWave(cy, 28, 0.012, 0.8, 0.15, '#3b82f6');
      drawWave(cy - 20, 18, 0.018, 1.1, 0.1, '#60a5fa');
      drawWave(cy + 20, 22, 0.015, 0.6, 0.12, '#1d4ed8');
      drawWave(cy, 40, 0.008, 0.5, 0.08, '#93c5fd');
      t += 0.03;
      animId = requestAnimationFrame(animate);
    };
    animate();
    return () => { cancelAnimationFrame(animId); window.removeEventListener('resize', resize); };
  }, []);

  // Floating code snippets data
  const leftCode = [
    '{ "session": "ax-9f2",',
    '  "audio": "meeting.mp3",',
    '  "lang": "en-US",',
    '  "confidence": 0.97,',
    '  "words": [',
    '    { "start": 0.0,',
    '      "end": 0.82,',
    '      "word": "Hello" },',
    '    { "start": 0.9,',
    '      "end": 1.4,',
    '      "word": "world" }',
    '  ],',
    '  "segments": 14,',
    '  "duration": 183.4',
    '}',
    'function transcribe(src) {',
    '  const buf = await',
    '    load(src);',
    '  return model',
    '    .infer(buf);',
    '}',
  ];

  const rightCode = [
    'preprocess({',
    '  noise_reduce: true,',
    '  normalize: -14,',
    '  vad_threshold: 0.4,',
    '  diarize: true,',
    '  speakers: "auto",',
    '  format: "wav",',
    '  sample_rate: 16000,',
    '  channels: 1,',
    '})',
    '.then(clean => {',
    '  emit("ready", clean);',
    '  pipeline.next(clean);',
    '})',
    'const eq = EQ.create({',
    '  bands: [80,250,1k,4k],',
    '  gains: [2,-1,3,1],',
    '})',
  ];

  return (
    <div className="relative min-h-screen bg-[#060c1a] overflow-hidden flex flex-col items-center justify-center select-none">

      {/* Ambient background glow */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-blue-600/5 blur-[120px]" />
        <div className="absolute top-1/3 left-1/4 w-[300px] h-[300px] rounded-full bg-blue-800/4 blur-[100px]" />
        <div className="absolute top-2/3 right-1/4 w-[250px] h-[250px] rounded-full bg-indigo-700/4 blur-[80px]" />
      </div>

      {/* Subtle grid overlay */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.025]"
        style={{
          backgroundImage: 'linear-gradient(rgba(96,165,250,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(96,165,250,0.5) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }}
      />

      {/* Left code column */}
      <div className="absolute left-0 top-0 h-full w-[280px] overflow-hidden pointer-events-none">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent to-[#060c1a] z-10" />
        <div className="absolute inset-0 bg-gradient-to-b from-[#060c1a] via-transparent to-[#060c1a] z-10" />
        <div className="pt-20 pl-4 space-y-1.5">
          {leftCode.map((line, i) => (
            <div key={i} className="font-mono text-[11px] text-slate-500/50 whitespace-nowrap tracking-tight">{line}</div>
          ))}
        </div>
        {/* Left soundwave bars */}
        <div className="absolute bottom-1/3 left-6 flex items-end gap-[3px]">
          {[12, 28, 18, 35, 22, 40, 15, 32, 25, 18, 38, 20, 30, 14, 26].map((h, i) => (
            <div key={i} className="w-[3px] rounded-full bg-blue-500/20" style={{ height: `${h}px` }} />
          ))}
        </div>
        {/* Left icon */}
        <div className="absolute top-1/2 -translate-y-1/2 left-8">
          <div className="w-16 h-16 rounded-2xl bg-slate-800/60 border border-slate-700/40 flex items-center justify-center backdrop-blur-sm">
            <Code2 className="w-7 h-7 text-slate-400/60" />
          </div>
        </div>
      </div>

      {/* Right code column */}
      <div className="absolute right-0 top-0 h-full w-[280px] overflow-hidden pointer-events-none">
        <div className="absolute inset-0 bg-gradient-to-l from-transparent to-[#060c1a] z-10" />
        <div className="absolute inset-0 bg-gradient-to-b from-[#060c1a] via-transparent to-[#060c1a] z-10" />
        <div className="pt-20 pr-4 space-y-1.5 text-right">
          {rightCode.map((line, i) => (
            <div key={i} className="font-mono text-[11px] text-slate-500/50 whitespace-nowrap tracking-tight">{line}</div>
          ))}
        </div>
        {/* Right soundwave bars */}
        <div className="absolute bottom-1/3 right-6 flex items-end gap-[3px]">
          {[20, 35, 15, 28, 40, 18, 33, 22, 38, 16, 30, 25, 20, 35, 12].map((h, i) => (
            <div key={i} className="w-[3px] rounded-full bg-blue-500/20" style={{ height: `${h}px` }} />
          ))}
        </div>
        {/* Right icon */}
        <div className="absolute top-1/2 -translate-y-1/2 right-8">
          <div className="w-16 h-16 rounded-2xl bg-slate-800/60 border border-slate-700/40 flex items-center justify-center backdrop-blur-sm">
            <SlidersHorizontal className="w-7 h-7 text-slate-400/60" />
          </div>
        </div>
      </div>

      {/* Center content */}
      <div className="relative z-10 flex flex-col items-center text-center px-8 max-w-2xl">

        {/* Mic logo at top */}
        <div className="mb-8">
          <div className="relative w-10 h-10 mx-auto">
            <div className="absolute inset-0 rounded-full bg-blue-500/10 animate-ping" style={{ animationDuration: '3s' }} />
            <Mic className="w-10 h-10 text-white/40 relative z-10" strokeWidth={1.5} />
          </div>
        </div>

        {/* Title */}
        <h1 className="text-7xl md:text-8xl font-black text-white mb-4 tracking-tight leading-none" style={{ fontFamily: "'Sora', 'DM Sans', sans-serif", textShadow: '0 0 80px rgba(59,130,246,0.3)' }}>
          VoxFlow AI
        </h1>

        {/* Tagline */}
        <p className="text-slate-400 text-lg md:text-xl mb-10 font-light tracking-wide" style={{ fontFamily: "'DM Sans', sans-serif" }}>
          Transcribe and preprocess audio with AI precision
        </p>

        {/* CTA Button */}
        <button
          onClick={onStart}
          className="relative group px-10 py-4 rounded-full text-white font-semibold text-lg tracking-wide transition-all duration-300 hover:scale-105 active:scale-95"
          style={{
            background: 'linear-gradient(135deg, #1d6fdb 0%, #2563eb 50%, #3b82f6 100%)',
            boxShadow: '0 0 30px rgba(59,130,246,0.4), 0 0 60px rgba(59,130,246,0.15), inset 0 1px 0 rgba(255,255,255,0.15)',
            fontFamily: "'DM Sans', sans-serif",
          }}
        >
          <span className="relative z-10">Start a Session</span>
          <div className="absolute inset-0 rounded-full bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        </button>

        {/* Waveform canvas */}
        <div className="w-full mt-14 mb-4" style={{ height: '60px' }}>
          <canvas ref={canvasRef} className="w-full h-full" />
        </div>

        {/* Bottom icons */}
        <div className="flex items-center gap-5 mb-10">
          {[
            { Icon: Mic, label: 'Record' },
            { Icon: Globe, label: 'Translate' },
            { Icon: Folder, label: 'Sessions' },
          ].map(({ Icon, label }) => (
            <div key={label} className="flex flex-col items-center gap-1.5">
              <div className="w-9 h-9 rounded-xl bg-slate-800/80 border border-slate-700/50 flex items-center justify-center">
                <Icon className="w-4 h-4 text-slate-400" strokeWidth={1.5} />
              </div>
            </div>
          ))}
        </div>

        {/* Nav links */}
        <div className="flex items-center gap-8 text-slate-500 text-sm tracking-widest uppercase" style={{ fontFamily: "'DM Sans', sans-serif", letterSpacing: '0.15em' }}>
          <a href="https://github.com/AleeCodeAI/VoxFlow_AI" target="_blank" rel="noopener noreferrer" className="hover:text-slate-300 transition-colors">GitHub</a>
          <span className="text-slate-700">·</span>
          <button onClick={onStart} className="hover:text-slate-300 transition-colors">Launch App</button>
        </div>
        <p className="text-slate-600 text-xs mt-6">Built by Alee — 17 year old aspiring AI Engineer</p>
        <p className="text-slate-700 text-xs mt-1">Hope you enjoy using VoxFlow AI ✨</p>
      </div>

      {/* Corner sparkle */}
      <div className="absolute bottom-8 right-8 pointer-events-none">
        <div className="w-3 h-3 bg-white/60 rotate-45 animate-pulse" style={{ clipPath: 'polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%)' }} />
      </div>
    </div>
  );
};

export default HomePage;