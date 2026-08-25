import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FaceDetector as VisionFaceDetector, FilesetResolver } from '@mediapipe/tasks-vision';

const API = 'api';
const api = async (path, init) => {
  const response = await fetch(`${API}${path}`, init);
  if (!response.ok) {
    let detail = `${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch { /* empty */ }
    throw new Error(detail);
  }
  return response.json();
};

const Icon = ({ children }) => <span className="icon" aria-hidden="true">{children}</span>;

function useCamera(enabled) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [state, setState] = useState({ ready: false, error: '' });
  const [requestVersion, requestCamera] = useState(0);
  useEffect(() => {
    let cancelled = false;
    if (!enabled) return undefined;
    navigator.mediaDevices?.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false })
      .then((stream) => {
        if (cancelled) return stream.getTracks().forEach((track) => track.stop());
        streamRef.current = stream;
        videoRef.current.srcObject = stream;
        return videoRef.current.play();
      })
      .then(() => !cancelled && setState({ ready: true, error: '' }))
      .catch((error) => !cancelled && setState({ ready: false, error: error.message }));
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, [enabled, requestVersion]);
  return { videoRef, requestCamera: () => requestCamera((value) => value + 1), ...state };
}

function descriptorFrom(canvas, box) {
  const sample = document.createElement('canvas');
  sample.width = 12; sample.height = 12;
  const ctx = sample.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(canvas, box.x, box.y, box.width, box.height, 0, 0, 12, 12);
  const pixels = ctx.getImageData(0, 0, 12, 12).data;
  const values = [];
  for (let i = 0; i < pixels.length; i += 4) values.push((pixels[i] * .299 + pixels[i + 1] * .587 + pixels[i + 2] * .114) / 255);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return values.map((value) => Number((value - mean).toFixed(4)));
}

function useFaceDetector() {
  const [detector, setDetector] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => {
    let disposed = false;
    FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm')
      .then((vision) => VisionFaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite' },
        runningMode: 'VIDEO', minDetectionConfidence: .68,
      }))
      .then((instance) => { if (disposed) instance.close(); else setDetector(instance); })
      .catch((reason) => !disposed && setError(reason.message));
    return () => { disposed = true; };
  }, []);
  return { detector, detectorError: error };
}

function CameraView({ interval, onCaptured, toast }) {
  const { videoRef, ready, error, requestCamera } = useCamera(true);
  const { detector, detectorError } = useFaceDetector();
  const canvasRef = useRef(null);
  const motionCanvasRef = useRef(null);
  const busy = useRef(false);
  const previousMotionFrame = useRef(null);
  const motionActiveUntil = useRef(0);
  const lastCaptureAt = useRef(0);
  const [paused, setPaused] = useState(false);
  const [signal, setSignal] = useState('Procurando um rosto…');
  const [motion, setMotion] = useState(false);

  const take = useCallback(async (manual = false) => {
    if (!ready || paused || busy.current || !videoRef.current?.videoWidth) return;
    if (!manual && (Date.now() > motionActiveUntil.current || Date.now() - lastCaptureAt.current < Math.max(3, interval) * 1000)) return;
    busy.current = true;
    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(video, 0, 0);
      if (!detector) { setSignal(detectorError ? 'Detector visual indisponível' : 'Preparando visão computacional…'); return; }
      const faces = detector.detectForVideo(video, performance.now()).detections || [];
      if (faces.length !== 1) { setSignal(faces.length ? 'Uma pessoa de cada vez' : 'Chegue um pouco mais perto'); return; }
      const detectedBox = faces[0].boundingBox;
      const box = { x: detectedBox.originX, y: detectedBox.originY, width: detectedBox.width, height: detectedBox.height };
      const coverage = (box.width * box.height) / (canvas.width * canvas.height);
      const image = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      let sum = 0, sum2 = 0, n = 0;
      for (let i = 0; i < image.length; i += 64) { const y = image[i] * .299 + image[i + 1] * .587 + image[i + 2] * .114; sum += y; sum2 += y * y; n += 1; }
      const brightness = sum / n, contrast = Math.sqrt(Math.max(0, sum2 / n - brightness * brightness));
      const quality = Math.min(1, coverage * 3.2) * Math.min(1, contrast / 42) * (brightness > 40 && brightness < 225 ? 1 : .5);
      if (quality < .42) { setSignal(brightness < 40 ? 'Está escuro — procure mais luz' : 'Fique parado e mais perto'); return; }
      setSignal('Ótimo — guardando este momento');
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', .9));
      const form = new FormData();
      form.append('photo', blob, `portrait-${Date.now()}.jpg`);
      form.append('descriptor_json', JSON.stringify(descriptorFrom(canvas, box)));
      form.append('quality', String(quality));
      const result = await api('/captures', { method: 'POST', body: form });
      lastCaptureAt.current = Date.now();
      onCaptured(result);
      setSignal(result.person ? `Oi, ${result.person.name}!` : 'Pessoa nova encontrada');
    } catch (e) { toast(e.message); setSignal('Não consegui salvar — vou tentar novamente'); }
    finally { busy.current = false; }
  }, [ready, paused, videoRef, onCaptured, toast, interval, detector, detectorError]);

  // A tiny frame-difference detector keeps all motion analysis on the tablet.
  // Motion only opens a five-second capture window; face + quality checks still
  // decide whether anything is uploaded.
  useEffect(() => {
    const detectMotion = () => {
      const video = videoRef.current;
      if (!ready || paused || !video?.videoWidth) return;
      const canvas = motionCanvasRef.current;
      canvas.width = 40; canvas.height = 30;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(video, 0, 0, 40, 30);
      const rgba = ctx.getImageData(0, 0, 40, 30).data;
      const frame = new Uint8Array(1200);
      let changed = 0;
      for (let pixel = 0, offset = 0; pixel < frame.length; pixel += 1, offset += 4) {
        frame[pixel] = (rgba[offset] * .299 + rgba[offset + 1] * .587 + rgba[offset + 2] * .114) | 0;
        if (previousMotionFrame.current && Math.abs(frame[pixel] - previousMotionFrame.current[pixel]) > 22) changed += 1;
      }
      previousMotionFrame.current = frame;
      const detected = changed / frame.length > .045;
      if (detected) {
        motionActiveUntil.current = Date.now() + 5000;
        setMotion(true);
        setSignal('Movimento detectado — procurando um rosto');
      } else if (Date.now() > motionActiveUntil.current) {
        setMotion(false);
      }
    };
    const id = setInterval(detectMotion, 650);
    return () => clearInterval(id);
  }, [ready, paused, videoRef]);

  useEffect(() => { const id = setInterval(() => take(false), 1200); return () => clearInterval(id); }, [take]);
  return <section className="cameraStage">
    <video ref={videoRef} playsInline muted />
    <canvas ref={canvasRef} hidden />
    <canvas ref={motionCanvasRef} hidden />
    <div className="cameraShade" />
    <div className="focusOval" />
    <div className="cameraTop"><span className={`liveDot ${ready ? '' : 'off'}`} /> {ready ? (motion ? 'MOVIMENTO DETECTADO' : 'OBSERVANDO O AMBIENTE') : 'CÂMERA'}</div>
    <div className="cameraMessage"><strong>{error ? 'Precisamos liberar a câmera' : signal}</strong><small>{error ? 'Toque abaixo para o navegador mostrar a permissão.' : 'Movimento → rosto → qualidade → galeria, tudo automático'}</small>{error && <button className="permissionButton" onClick={requestCamera}>Permitir câmera</button>}</div>
    <div className="cameraActions"><button onClick={() => take(true)} disabled={!ready}>Tirar foto agora</button><button onClick={() => setPaused((v) => !v)}>{paused ? 'Retomar automático' : 'Pausar automático'}</button></div>
  </section>;
}

function Gallery({ blocks, onSelect, selected }) {
  const images = blocks.flatMap((block) => block.images.map((image) => ({ ...image, source: block.source })));
  return <div className="galleryGrid">{images.map((image) => <button key={image.id} className={`photoCard ${selected === image.id ? 'selected' : ''}`} onClick={() => onSelect(image)}>
    <img src={image.url} alt="Retrato" loading="lazy" />
    <span className="chips">{image.tags?.slice(0, 2).map((tag) => <em key={tag.id}>{tag.name.replace('portrait:', '')}</em>)}</span>
  </button>)}</div>;
}

function PeoplePanel({ library, selectedImage, reload, toast }) {
  const [name, setName] = useState('');
  const [relation, setRelation] = useState({ source_id: '', target_id: '', kind: '' });
  const create = async () => { if (!name.trim()) return; try { await api('/people', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, image_id: selectedImage?.id }) }); setName(''); reload(); } catch (e) { toast(e.message); } };
  const link = async () => { try { await api('/relations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(relation) }); setRelation({ source_id: '', target_id: '', kind: '' }); reload(); } catch (e) { toast(e.message); } };
  return <div className="peoplePanel">
    <div className="panelCard"><h3>Quem é essa pessoa?</h3><p>{selectedImage ? 'Dê um nome ao rosto selecionado.' : 'Selecione uma foto nova na galeria.'}</p><div className="inline"><input value={name} onChange={(e) => setName(e.target.value)} placeholder="Nome da pessoa" /><button onClick={create} disabled={!selectedImage}>Salvar</button></div></div>
    <div className="peopleList">{library.people.map((person) => <div className="person" key={person.id}><div className="avatar">{person.photo_ids[0] ? <img src={`api/media/${person.photo_ids[0]}`} /> : person.name[0]}</div><div><strong>{person.name}</strong><small>{person.photo_ids.length} retrato(s)</small></div></div>)}</div>
    <div className="panelCard"><h3>Como eles se conhecem?</h3><div className="relationForm"><select value={relation.source_id} onChange={(e) => setRelation({ ...relation, source_id: e.target.value })}><option value="">Pessoa 1</option>{library.people.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select><input value={relation.kind} onChange={(e) => setRelation({ ...relation, kind: e.target.value })} placeholder="ex.: irmãos, amigos" /><select value={relation.target_id} onChange={(e) => setRelation({ ...relation, target_id: e.target.value })}><option value="">Pessoa 2</option>{library.people.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select><button onClick={link}>Criar relação</button></div></div>
  </div>;
}

function CreatePanel({ library, onGenerated, toast }) {
  const [prompt, setPrompt] = useState('');
  const [selected, setSelected] = useState([]);
  const [busy, setBusy] = useState(false);
  const ideas = ['num piquenique em Marte', 'como estrelas de um filme dos anos 80', 'numa aventura pirata elegante', 'celebrando juntos em um jardim mágico'];
  const run = async () => { setBusy(true); try { const result = await api('/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt, person_ids: selected }) }); onGenerated(result); setPrompt(''); } catch (e) { toast(e.message); } finally { setBusy(false); } };
  return <div className="createPanel"><header><span className="eyebrow">ESTÚDIO CRIATIVO</span><h2>Onde vamos levar todo mundo hoje?</h2><p>Escolha as pessoas e descreva a cena. A identidade vem dos retratos aprovados.</p></header><div className="cast">{library.people.map((person) => <button key={person.id} className={selected.includes(person.id) ? 'picked' : ''} onClick={() => setSelected((ids) => ids.includes(person.id) ? ids.filter((id) => id !== person.id) : [...ids, person.id])}><div className="avatar">{person.photo_ids[0] ? <img src={`api/media/${person.photo_ids[0]}`} /> : person.name[0]}</div>{person.name}</button>)}</div><div className="promptBox"><textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Ex.: Frederico e Ana explorando uma cidade flutuante, luz dourada, fotografia cinematográfica…" /><div className="ideas">{ideas.map((idea) => <button key={idea} onClick={() => setPrompt(idea)}>{idea}</button>)}</div><button className="generate" disabled={busy || !prompt || !selected.length} onClick={run}>{busy ? 'Criando…' : 'Criar novo retrato ✦'}</button></div></div>;
}

export default function App() {
  const [view, setView] = useState('frame');
  const [status, setStatus] = useState({ capture_interval_seconds: 12 });
  const [library, setLibrary] = useState({ people: [], relations: [], generation_history: [] });
  const [blocks, setBlocks] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [notice, setNotice] = useState('');
  const toast = useCallback((message) => { setNotice(message); setTimeout(() => setNotice(''), 5000); }, []);
  const reload = useCallback(async () => { try { const [s, l] = await Promise.all([api('/status'), api('/library')]); setStatus(s); setLibrary(l); if (s.gallery_configured) setBlocks((await api('/gallery')).blocks); } catch (e) { toast(e.message); } }, [toast]);
  useEffect(() => { reload(); }, [reload]);
  const displayImages = useMemo(() => blocks.flatMap((b) => b.images).filter((img) => img.tags?.some((tag) => ['portrait:generated', 'portrait:display-ready'].includes(tag.name))), [blocks]);
  const [slide, setSlide] = useState(0);
  useEffect(() => { const id = setInterval(() => setSlide((n) => displayImages.length ? (n + 1) % displayImages.length : 0), 18000); return () => clearInterval(id); }, [displayImages.length]);
  return <main>
    <nav><button className="brand" onClick={() => setView('frame')}><Icon>✦</Icon><span>AI Portrait<small>MEMÓRIAS VIVAS</small></span></button><div className="navTabs">{[['frame','Porta-retrato'],['camera','Câmera'],['people','Pessoas'],['create','Criar']].map(([id,label]) => <button key={id} className={view === id ? 'active' : ''} onClick={() => setView(id)}>{label}</button>)}</div><button className="settings" onClick={() => setView('people')}>⚙</button></nav>
    {view === 'frame' && <section className="frameView">{displayImages.length ? <><img src={displayImages[slide]?.url} /><div className="frameGradient" /><div className="frameCaption"><span>RETRATO DO MOMENTO</span><strong>Uma memória que nunca aconteceu.<br />Mas deveria.</strong></div><div className="dots">{displayImages.map((_, i) => <i className={i === slide ? 'on' : ''} key={i} />)}</div></> : <div className="emptyFrame"><Icon>✦</Icon><h1>Seu porta-retrato está pronto.</h1><p>Conheça algumas pessoas pela câmera e crie a primeira memória impossível.</p><button onClick={() => setView('camera')}>Abrir a câmera</button></div>}</section>}
    <div className={view === 'camera' ? '' : 'cameraBackground'}><CameraView interval={status.capture_interval_seconds} toast={toast} onCaptured={(result) => { reload(); if (!result.person) toast('Pessoa nova encontrada — abra Pessoas para dar um nome.'); }} /></div>
    {view === 'people' && <section className="workspace"><div className="sectionHead"><span className="eyebrow">GALERIA & IDENTIDADE</span><h2>As pessoas por trás das histórias</h2><p>Selecione uma captura, dê um nome e ensine relações. Nada é publicado sem sua configuração.</p></div><div className="split"><div><Gallery blocks={blocks} selected={selectedImage?.id} onSelect={setSelectedImage} /></div><PeoplePanel library={library} selectedImage={selectedImage} reload={reload} toast={toast} /></div></section>}
    {view === 'create' && <CreatePanel library={library} toast={toast} onGenerated={() => { reload(); toast('Novo retrato criado e adicionado ao porta-retrato.'); }} />}
    {notice && <div className="toast">{notice}</div>}
  </main>;
}
