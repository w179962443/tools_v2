"""
Realtime microphone transcription primitives and a small Flask web app.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_REALTIME_CONFIG: dict[str, Any] = {
    "sample_rate": 16000,
    "chunk_duration": 0.5,
    "max_buffer_size": 50,
    "model_name": "base",
    "language": "auto",
    "transcribe_interval": 2.0,
    "use_gpu": False,
}

SUPPORTED_LANGUAGES: dict[str, str] = {
    "auto": "自动检测",
    "zh": "中文",
    "en": "English",
    "ja": "日本語",
    "ko": "한국어",
    "fr": "Français",
    "de": "Deutsch",
    "es": "Español",
    "ru": "Русский",
    "pt": "Português",
}


class AudioRecorder:
    """Record audio from the default input device using sounddevice."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,
        max_buffer_size: int = 50,
        device: int | str | None = None,
    ) -> None:
        try:
            import numpy as numpy_module
            import sounddevice as sounddevice_module
        except ImportError as import_error:
            raise ImportError(
                "Realtime recording requires numpy and sounddevice. "
                "Install with: pip install numpy sounddevice"
            ) from import_error

        from collections import deque

        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.max_buffer_size = max_buffer_size
        self.device = device
        self.audio_buffer = deque(maxlen=max_buffer_size)
        self.is_recording = False
        self.lock = threading.Lock()
        self.stream = None
        self._numpy = numpy_module
        self._sounddevice = sounddevice_module

    def audio_callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """Collect incoming mono audio blocks."""
        if status:
            print(f"Audio status: {status}")
        with self.lock:
            self.audio_buffer.append(indata[:, 0].copy())

    def start_recording(self, source: str = "mic") -> None:
        """Start recording from the selected audio source."""
        if self.is_recording:
            return

        if source in {"system", "both"}:
            print(
                "System audio capture needs an OS-level loopback or virtual audio "
                "device. Falling back to the selected/default input device."
            )

        self.is_recording = True
        self.clear_buffer()
        try:
            self.stream = self._sounddevice.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.chunk_size,
                callback=self.audio_callback,
                device=self.device,
            )
            self.stream.start()
            print(f"Recording started. Source: {source}")
        except Exception:
            self.is_recording = False
            raise

    def stop_recording(self) -> None:
        """Stop recording and close the input stream."""
        if not self.is_recording:
            return
        self.is_recording = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_audio_chunk(self) -> Any | None:
        """Return the current concatenated audio buffer."""
        with self.lock:
            if not self.audio_buffer:
                return None
            return self._numpy.concatenate(list(self.audio_buffer))

    def clear_buffer(self) -> None:
        """Clear buffered audio blocks."""
        with self.lock:
            self.audio_buffer.clear()


class WhisperTranscriber:
    """Small wrapper around OpenAI Whisper for realtime chunks."""

    def __init__(
        self,
        model_name: str = "base",
        language: str = "auto",
        model_dir: str | Path | None = None,
        use_gpu: bool = False,
    ) -> None:
        self.model_name = model_name
        self.language = None if language == "auto" else language
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self.model = self._load_model()

    def _load_model(self) -> Any:
        try:
            import whisper
        except ImportError as import_error:
            raise ImportError(
                "openai-whisper is required. Install with: pip install openai-whisper"
            ) from import_error

        device = None
        if self.use_gpu:
            device = "cuda"
        print(f"Loading Whisper model: {self.model_name}")
        return whisper.load_model(
            self.model_name,
            device=device,
            download_root=str(self.model_dir) if self.model_dir else None,
        )

    def transcribe_audio(self, audio_data: Any, language: str | None = None) -> dict[str, Any]:
        """Transcribe a numpy audio array or an audio file path."""
        transcribe_language = language if language != "auto" else None
        if transcribe_language is None:
            transcribe_language = self.language

        transcribe_kwargs: dict[str, Any] = {}
        if transcribe_language:
            transcribe_kwargs["language"] = transcribe_language

        try:
            result = self.model.transcribe(audio_data, **transcribe_kwargs)
            return {
                "text": result.get("text", "").strip(),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as exception:
            return {
                "text": "",
                "language": "unknown",
                "segments": [],
                "error": str(exception),
                "timestamp": datetime.now().isoformat(),
            }

    def transcribe_with_timestamps(self, audio_data: Any) -> list[dict[str, Any]]:
        """Return segment-level timestamps for a chunk or file."""
        result = self.transcribe_audio(audio_data)
        return [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment.get("text", "").strip(),
            }
            for segment in result.get("segments", [])
        ]


class TranscriptionLogger:
    """Persist realtime transcript entries to session text files."""

    def __init__(self, output_dir: str | Path = "recordings") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: dict[str, Any] | None = None
        self.current_file: Path | None = None

    def start_new_session(self) -> Path:
        """Create a new timestamped transcript file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcription_{timestamp}.txt"
        self.current_file = self.output_dir / filename
        self.current_session = {
            "start_time": datetime.now(),
            "filename": filename,
            "entries": [],
        }

        with self.current_file.open("w", encoding="utf-8") as handle:
            handle.write("=" * 60 + "\n")
            handle.write(
                "转录会话开始时间: "
                f"{self.current_session['start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            handle.write("=" * 60 + "\n\n")
        return self.current_file

    def log_transcription(
        self,
        text: str,
        language: str = "unknown",
        confidence: float = 0.0,
    ) -> None:
        """Append one transcript entry to the current session file."""
        if self.current_file is None:
            self.start_new_session()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        entry = {
            "timestamp": timestamp,
            "text": text,
            "language": language,
            "confidence": confidence,
        }
        assert self.current_session is not None
        self.current_session["entries"].append(entry)

        with self.current_file.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] [{language}] {text}\n")

    def get_session_summary(self) -> dict[str, Any] | None:
        """Return a JSON-friendly summary of the current session."""
        if self.current_session is None:
            return None
        return {
            "filename": self.current_session["filename"],
            "start_time": self.current_session["start_time"].isoformat(),
            "total_entries": len(self.current_session["entries"]),
            "entries": self.current_session["entries"],
        }


def _realtime_html() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>实时转录</title>
  <style>
    *{box-sizing:border-box} body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f4f6f8;color:#1f2933}
    .app{display:grid;grid-template-columns:280px 1fr;min-height:100vh} aside{background:#ffffff;border-right:1px solid #d9e2ec;padding:18px}
    main{display:flex;flex-direction:column;min-width:0}.toolbar{height:58px;background:#ffffff;border-bottom:1px solid #d9e2ec;display:flex;align-items:center;justify-content:space-between;padding:0 18px}
    h1{font-size:18px;margin:0}.field{margin:0 0 14px}.field label{display:block;font-size:12px;font-weight:600;margin-bottom:6px;color:#52606d}
    select,input,button{width:100%;height:36px;border:1px solid #bcccdc;border-radius:6px;background:#fff;color:#1f2933;font-size:14px}button{cursor:pointer;font-weight:600}.primary{background:#2563eb;color:#fff;border-color:#2563eb}.danger{background:#dc2626;color:#fff;border-color:#dc2626}.secondary{background:#edf2f7}
    button:disabled{opacity:.55;cursor:not-allowed}.actions{display:grid;gap:8px;margin:18px 0}.stats{display:grid;grid-template-columns:1fr 1fr;gap:8px}.stat{border:1px solid #d9e2ec;border-radius:6px;padding:10px;background:#f8fafc}.value{font-size:20px;font-weight:700}.label{font-size:12px;color:#627d98}
    #transcript{flex:1;overflow:auto;padding:18px}.entry{background:#fff;border:1px solid #d9e2ec;border-left:4px solid #2563eb;border-radius:6px;padding:10px;margin-bottom:10px}.meta{font-size:12px;color:#627d98;margin-bottom:5px}.text{white-space:pre-wrap;line-height:1.55}
    .status{display:flex;align-items:center;gap:8px}.dot{width:10px;height:10px;border-radius:50%;background:#9fb3c8}.dot.on{background:#16a34a}.message{font-size:13px;color:#b42318;min-height:20px}
    @media(max-width:760px){.app{grid-template-columns:1fr}aside{border-right:0;border-bottom:1px solid #d9e2ec}.toolbar{height:auto;padding:12px 18px}}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="field"><label>音源</label><select id="source"><option value="mic">麦克风</option><option value="system">系统声音</option><option value="both">麦克风 + 系统声音</option></select></div>
      <div class="field"><label>语言</label><select id="language"><option value="auto">自动检测</option><option value="zh">中文</option><option value="en">English</option><option value="ja">日本語</option><option value="ko">한국어</option></select></div>
      <div class="field"><label>模型</label><select id="model"><option value="tiny">tiny</option><option value="base" selected>base</option><option value="small">small</option><option value="medium">medium</option><option value="large">large</option></select></div>
      <div class="field"><label>转录间隔（秒）</label><input id="interval" type="number" min="1" max="30" step="0.5" value="2"></div>
      <div class="actions"><button id="start" class="primary">开始转录</button><button id="stop" class="danger" disabled>停止转录</button><button id="clear" class="secondary">清空屏幕</button><button id="download" class="secondary">下载文本</button></div>
      <div class="stats"><div class="stat"><div id="count" class="value">0</div><div class="label">文本条数</div></div><div class="stat"><div id="elapsed" class="value">00:00</div><div class="label">运行时间</div></div></div>
      <p id="message" class="message"></p>
    </aside>
    <main>
      <div class="toolbar"><h1>实时转录</h1><div class="status"><span id="dot" class="dot"></span><span id="status">就绪</span></div></div>
      <div id="transcript"></div>
    </main>
  </div>
  <script>
    let events=null, startedAt=null, count=0;
    const $=id=>document.getElementById(id);
    function setMessage(text){$('message').textContent=text||''}
    function setRunning(on){$('start').disabled=on;$('stop').disabled=!on;$('dot').classList.toggle('on',on);$('status').textContent=on?'录制中':'已停止'}
    function escapeHtml(text){return text.replace(/[&<>"']/g, ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]))}
    function tick(){if(!startedAt)return;const seconds=Math.floor((Date.now()-startedAt)/1000);$('elapsed').textContent=String(Math.floor(seconds/60)).padStart(2,'0')+':'+String(seconds%60).padStart(2,'0')} setInterval(tick,1000)
    function addEntry(data){const entry=document.createElement('div');entry.className='entry';entry.innerHTML='<div class="meta">'+new Date(data.timestamp).toLocaleTimeString()+' · '+(data.language||'unknown')+'</div><div class="text">'+escapeHtml(data.text)+'</div>';$('transcript').appendChild(entry);$('transcript').scrollTop=$('transcript').scrollHeight;count++;$('count').textContent=count}
    function connect(){events=new EventSource('/api/transcriptions');events.onmessage=e=>{const data=JSON.parse(e.data);if(data.type==='transcription')addEntry(data);if(data.type==='error')setMessage(data.message)}}
    $('start').onclick=async()=>{setMessage('');const response=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:$('source').value,model_name:$('model').value,language:$('language').value,transcribe_interval:Number($('interval').value)})});const data=await response.json();if(data.status==='success'){startedAt=Date.now();count=0;$('count').textContent='0';$('transcript').innerHTML='';setRunning(true);connect()}else setMessage(data.message)};
    $('stop').onclick=async()=>{const response=await fetch('/api/stop',{method:'POST'});const data=await response.json();if(events)events.close();setRunning(false);if(data.status!=='success')setMessage(data.message)};
    $('clear').onclick=()=>{$('transcript').innerHTML='';count=0;$('count').textContent='0'};
    $('download').onclick=async()=>{const response=await fetch('/api/status');const data=await response.json();if(data.current_session&&data.current_session.filename)location.href='/api/download/'+data.current_session.filename;else setMessage('没有可下载的文件')};
  </script>
</body>
</html>
"""


def create_realtime_app(
    config: dict[str, Any] | None = None,
    output_dir: str | Path = "recordings",
) -> Any:
    """Create a Flask app for realtime transcription."""
    try:
        from flask import Flask, jsonify, request, send_from_directory
        from flask_cors import CORS
    except ImportError as import_error:
        raise ImportError(
            "The realtime web app requires Flask and Flask-CORS. "
            "Install with: pip install Flask Flask-CORS"
        ) from import_error

    app_config = DEFAULT_REALTIME_CONFIG.copy()
    if config:
        app_config.update(config)

    app = Flask(__name__)
    CORS(app)

    state: dict[str, Any] = {
        "recorder": None,
        "transcriber": None,
        "logger": TranscriptionLogger(output_dir=output_dir),
        "queue": queue.Queue(),
        "is_running": False,
        "current_session": None,
    }

    def transcription_worker() -> None:
        logger: TranscriptionLogger = state["logger"]
        recorder: AudioRecorder = state["recorder"]
        transcriber: WhisperTranscriber = state["transcriber"]
        if logger.current_file is None:
            logger.start_new_session()
        state["current_session"] = logger.get_session_summary()
        last_transcribe_time = 0.0

        while state["is_running"]:
            try:
                current_time = time.time()
                if current_time - last_transcribe_time >= float(app_config["transcribe_interval"]):
                    audio_chunk = recorder.get_audio_chunk()
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        result = transcriber.transcribe_audio(audio_chunk)
                        if result.get("error"):
                            state["queue"].put({"type": "error", "message": result["error"]})
                        elif result.get("text"):
                            language = result.get("language", "unknown")
                            logger.log_transcription(result["text"], language=language, confidence=0.9)
                            state["current_session"] = logger.get_session_summary()
                            state["queue"].put(
                                {
                                    "type": "transcription",
                                    "text": result["text"],
                                    "language": language,
                                    "timestamp": datetime.now().isoformat(),
                                }
                            )
                        recorder.clear_buffer()
                    last_transcribe_time = current_time
                time.sleep(0.1)
            except Exception as exception:
                state["queue"].put({"type": "error", "message": str(exception)})

    @app.route("/")
    def index() -> str:
        return _realtime_html()

    @app.route("/api/start", methods=["POST"])
    def start_transcription() -> Any:
        if state["is_running"]:
            return jsonify({"status": "error", "message": "Already running"})

        updates = request.get_json(silent=True) or {}
        for key in ("model_name", "language", "transcribe_interval", "use_gpu"):
            if key in updates:
                app_config[key] = updates[key]

        try:
            state["recorder"] = AudioRecorder(
                sample_rate=int(app_config["sample_rate"]),
                chunk_duration=float(app_config["chunk_duration"]),
                max_buffer_size=int(app_config["max_buffer_size"]),
            )
            state["transcriber"] = WhisperTranscriber(
                model_name=str(app_config["model_name"]),
                language=str(app_config["language"]),
                use_gpu=bool(app_config["use_gpu"]),
            )
            state["logger"].start_new_session()
            state["current_session"] = state["logger"].get_session_summary()
            state["is_running"] = True
            state["recorder"].start_recording(source=updates.get("source", "mic"))
            threading.Thread(target=transcription_worker, daemon=True).start()
            return jsonify(
                {
                    "status": "success",
                    "message": "Transcription started",
                    "session": state["current_session"],
                }
            )
        except Exception as exception:
            state["is_running"] = False
            if state.get("recorder") is not None:
                try:
                    state["recorder"].stop_recording()
                except Exception:
                    pass
            return jsonify({"status": "error", "message": str(exception)})

    @app.route("/api/stop", methods=["POST"])
    def stop_transcription() -> Any:
        if not state["is_running"]:
            return jsonify({"status": "error", "message": "Not running"})
        state["is_running"] = False
        if state.get("recorder") is not None:
            state["recorder"].stop_recording()
        time.sleep(0.2)
        return jsonify(
            {
                "status": "success",
                "message": "Transcription stopped",
                "summary": state["logger"].get_session_summary(),
            }
        )

    @app.route("/api/status")
    def status() -> Any:
        return jsonify(
            {
                "is_running": state["is_running"],
                "current_session": state["logger"].get_session_summary(),
                "config": app_config,
            }
        )

    @app.route("/api/transcriptions")
    def transcriptions() -> Any:
        def generate() -> Any:
            while state["is_running"]:
                try:
                    data = state["queue"].get(timeout=1)
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"

        return app.response_class(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/download/<filename>")
    def download_file(filename: str) -> Any:
        logger: TranscriptionLogger = state["logger"]
        return send_from_directory(logger.output_dir, filename, as_attachment=True)

    @app.route("/api/sessions")
    def list_sessions() -> Any:
        logger: TranscriptionLogger = state["logger"]
        sessions = []
        for transcript_path in logger.output_dir.glob("transcription_*.txt"):
            try:
                content = transcript_path.read_text(encoding="utf-8")
                entry_count = sum(1 for line in content.splitlines() if line.startswith("["))
            except OSError:
                entry_count = 0
            sessions.append(
                {
                    "filename": transcript_path.name,
                    "created": datetime.fromtimestamp(transcript_path.stat().st_ctime).isoformat(),
                    "size": transcript_path.stat().st_size,
                    "entries": entry_count,
                }
            )
        return jsonify({"sessions": sorted(sessions, key=lambda item: item["created"], reverse=True)})

    return app


def run_realtime_server(
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
    output_dir: str | Path = "recordings",
    config: dict[str, Any] | None = None,
) -> None:
    """Run the realtime transcription Flask app."""
    app = create_realtime_app(config=config, output_dir=output_dir)
    app.run(host=host, port=port, debug=debug, use_reloader=False)