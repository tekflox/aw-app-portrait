import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FaceDetector as VisionFaceDetector, FilesetResolver } from '@mediapipe/tasks-vision';
import './ambient.css';
import './gallery.css';

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

function CameraView({ interval, captureReady, captureBlocker, onCaptured, toast }) {
  const { videoRef, ready, error, requestCamera } = useCamera(true);
  const { detector, detectorError } = useFaceDetector();
  const canvasRef = useRef(null);
  const busy = useRef(false);
  const stablePresence = useRef(0);
  const lastCaptureAt = useRef(0);
  const [paused, setPaused] = useState(false);
  const [signal, setSignal] = useState('Looking for a face…');
  const [showcase, setShowcase] = useState(null);

  const take = useCallback(async (manual = false) => {
    if (!captureReady) {
      setSignal(captureBlocker || 'Photo capture is not configured');
      return;
    }
    if (!ready || paused || busy.current || !videoRef.current?.videoWidth) return;
    if (!manual && Date.now() - lastCaptureAt.current < Math.max(3, interval) * 1000) return;
    busy.current = true;
    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(video, 0, 0);
      if (!detector) { setSignal(detectorError ? 'Vision detector unavailable' : 'Preparing computer vision…'); return; }
      const faces = detector.detectForVideo(video, performance.now()).detections || [];
      if (faces.length !== 1) { stablePresence.current = 0; setSignal(faces.length ? 'One person at a time' : 'Waiting quietly for someone…'); return; }
      const detectedBox = faces[0].boundingBox;
      const box = { x: detectedBox.originX, y: detectedBox.originY, width: detectedBox.width, height: detectedBox.height };
      const points = faces[0].keypoints || [];
      const eyeSpan = points.length >= 2 ? Math.abs(points[0].x - points[1].x) : 0;
      const noseCentered = points.length < 3 || (points[2].x > Math.min(points[0].x, points[1].x) && points[2].x < Math.max(points[0].x, points[1].x));
      const looking = eyeSpan > .055 && noseCentered;
      stablePresence.current = looking ? stablePresence.current + 1 : 0;
      if (!manual && stablePresence.current < 2) { setSignal(looking ? 'Hello — stay for a moment' : 'Look this way when you are ready'); return; }
      const coverage = (box.width * box.height) / (canvas.width * canvas.height);
      const image = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      let sum = 0, sum2 = 0, n = 0;
      for (let i = 0; i < image.length; i += 64) { const y = image[i] * .299 + image[i + 1] * .587 + image[i + 2] * .114; sum += y; sum2 += y * y; n += 1; }
      const brightness = sum / n, contrast = Math.sqrt(Math.max(0, sum2 / n - brightness * brightness));
      const quality = Math.min(1, coverage * 3.2) * Math.min(1, contrast / 42) * (brightness > 40 && brightness < 225 ? 1 : .5);
      if (quality < .42) { setSignal(brightness < 40 ? 'It is too dark — find more light' : 'Stay still and come closer'); return; }
      setSignal('Great — saving this moment');
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', .9));
      const form = new FormData();
      form.append('photo', blob, `portrait-${Date.now()}.jpg`);
      form.append('descriptor_json', JSON.stringify(descriptorFrom(canvas, box)));
      form.append('face_box_json', JSON.stringify([box.x / canvas.width, box.y / canvas.height, box.width / canvas.width, box.height / canvas.height]));
      form.append('quality', String(quality));
      const result = await api('/captures', { method: 'POST', body: form });
      lastCaptureAt.current = Date.now();
      onCaptured(result);
      if (result.photo_ids?.length) {
        setShowcase({ name: result.person?.name || 'You', photos: result.photo_ids });
        setTimeout(() => setShowcase(null), 9000);
      }
      setSignal(result.collection_complete ? 'I already have ten good moments' : result.person ? `Hi, ${result.person.name}!` : `${result.sample_count || 1} of 10 moments collected`);
    } catch (e) { toast(e.message); setSignal('Could not save it — I will try again'); }
    finally { busy.current = false; }
  }, [captureReady, captureBlocker, ready, paused, videoRef, onCaptured, toast, interval, detector, detectorError]);

  useEffect(() => { const id = setInterval(() => take(false), 1200); return () => clearInterval(id); }, [take]);
  return <section className="cameraStage">
    <video ref={videoRef} playsInline muted />
    <canvas ref={canvasRef} hidden />
    {showcase && <div className="selfShowcase"><div className="selfPhotos">{showcase.photos.slice(-10).map((id) => <img key={id} src={`api/media/${id}`} alt="A moment of you" />)}</div><strong>Hi, {showcase.name}.</strong><small>These are the moments I remember.</small></div>}
    <div className="cameraTop"><span className={`liveDot ${ready && captureReady ? '' : 'off'}`} /> {!captureReady ? 'SETUP REQUIRED' : ready ? 'AMBIENT CAPTURE' : 'CAMERA'}</div>
    {!showcase && <div className={`cameraMessage ${!captureReady ? 'captureBlocked' : ''}`}><strong>{!captureReady ? 'Photo saving is not configured' : error ? 'Camera access is required' : signal}</strong><small>{!captureReady ? (captureBlocker || 'Open the app settings and configure the gallery.') : error ? 'Tap below to open the browser permission prompt.' : 'Presence → server validation → person crop → private gallery'}</small>{error && captureReady && <button className="permissionButton" onClick={requestCamera}>Allow camera</button>}</div>}
    <div className="cameraActions"><button onClick={() => take(true)} disabled={!ready || !captureReady}>Take a photo now</button><button onClick={() => setPaused((v) => !v)} disabled={!captureReady}>{paused ? 'Resume automatic capture' : 'Pause automatic capture'}</button></div>
  </section>;
}

function Gallery({ blocks, library, onSelect, selected }) {
  const [filter, setFilter] = useState('captures');
  const images = blocks.flatMap((block) => block.images.map((image) => ({ ...image, source: block.source })));
  const tagged = (image, name) => image.tags?.some((tag) => tag.name === name);
  const captures = images.filter((image) => tagged(image, 'portrait:capture'));
  const visible = filter === 'all' ? images : filter === 'unnamed' ? captures.filter((image) => tagged(image, 'portrait:unknown')) : captures;
  const collectionCount = new Set(captures.flatMap((image) => image.tags?.map((tag) => tag.name.startsWith('portrait:collection:') ? tag.name : null).filter(Boolean) || [])).size;
  return <div className="captureGallery">
    <div className="galleryToolbar"><div><strong>{captures.length} captured photo{captures.length === 1 ? '' : 's'}</strong><small>{collectionCount || library.unknown_clusters?.length || 0} people collection{collectionCount === 1 ? '' : 's'} · updates automatically</small></div><div className="galleryFilters">{[['captures', 'Captured'], ['unnamed', 'Needs names'], ['all', 'All photos']].map(([id, label]) => <button key={id} className={filter === id ? 'active' : ''} onClick={() => setFilter(id)}>{label}</button>)}</div></div>
    {visible.length ? <div className="galleryGrid">{visible.map((image) => <button key={image.id} className={`photoCard ${selected === image.id ? 'selected' : ''}`} onClick={() => onSelect(image)}>
      <img src={`${image.url}?v=${image.id}`} alt="Captured portrait" loading="lazy" />
      <span className="chips"><em>{tagged(image, 'portrait:unknown') ? 'Needs a name' : tagged(image, 'portrait:generated') ? 'Generated' : 'Captured'}</em></span>
    </button>)}</div> : <div className="galleryEmpty"><strong>No photos in this view yet.</strong><span>Accepted camera captures will appear here automatically.</span></div>}
  </div>;
}

function PeoplePanel({ library, selectedImage, reload, toast }) {
  const [name, setName] = useState('');
  const [relation, setRelation] = useState({ source_id: '', target_id: '', kind: '' });
  const create = async () => { if (!name.trim()) return; try { await api('/people', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, image_id: selectedImage?.id }) }); setName(''); reload(); } catch (e) { toast(e.message); } };
  const link = async () => { try { await api('/relations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(relation) }); setRelation({ source_id: '', target_id: '', kind: '' }); reload(); } catch (e) { toast(e.message); } };
  return <div className="peoplePanel">
    <div className="panelCard"><h3>Who is this person?</h3><p>{selectedImage ? 'Name the selected face.' : 'Select a new photo from the gallery.'}</p><div className="inline"><input value={name} onChange={(e) => setName(e.target.value)} placeholder="Person's name" /><button onClick={create} disabled={!selectedImage}>Save</button></div></div>
    <div className="peopleList">{library.people.map((person) => <div className="person" key={person.id}><div className="avatar">{person.photo_ids[0] ? <img src={`api/media/${person.photo_ids[0]}`} /> : person.name[0]}</div><div><strong>{person.name}</strong><small>{person.photo_ids.length} portrait(s)</small></div></div>)}</div>
    <div className="panelCard"><h3>How do they know each other?</h3><div className="relationForm"><select value={relation.source_id} onChange={(e) => setRelation({ ...relation, source_id: e.target.value })}><option value="">Person 1</option>{library.people.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select><input value={relation.kind} onChange={(e) => setRelation({ ...relation, kind: e.target.value })} placeholder="e.g. siblings, friends" /><select value={relation.target_id} onChange={(e) => setRelation({ ...relation, target_id: e.target.value })}><option value="">Person 2</option>{library.people.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select><button onClick={link}>Create relationship</button></div></div>
  </div>;
}

function CreatePanel({ library, onGenerated, toast }) {
  const [prompt, setPrompt] = useState('');
  const [selected, setSelected] = useState([]);
  const [busy, setBusy] = useState(false);
  const ideas = ['having a picnic on Mars', 'as stars in an eighties movie', 'on an elegant pirate adventure', 'celebrating together in a magical garden'];
  const run = async () => { setBusy(true); try { const result = await api('/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt, person_ids: selected }) }); onGenerated(result); setPrompt(''); } catch (e) { toast(e.message); } finally { setBusy(false); } };
  return <div className="createPanel"><header><span className="eyebrow">CREATIVE STUDIO</span><h2>Where should we take everyone today?</h2><p>Choose the people and describe the scene. Identity comes from approved portraits.</p></header><div className="cast">{library.people.map((person) => <button key={person.id} className={selected.includes(person.id) ? 'picked' : ''} onClick={() => setSelected((ids) => ids.includes(person.id) ? ids.filter((id) => id !== person.id) : [...ids, person.id])}><div className="avatar">{person.photo_ids[0] ? <img src={`api/media/${person.photo_ids[0]}`} /> : person.name[0]}</div>{person.name}</button>)}</div><div className="promptBox"><textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Example: Fred and Ana exploring a floating city, golden light, cinematic photography…" /><div className="ideas">{ideas.map((idea) => <button key={idea} onClick={() => setPrompt(idea)}>{idea}</button>)}</div><button className="generate" disabled={busy || !prompt || !selected.length} onClick={run}>{busy ? 'Creating…' : 'Create a new portrait ✦'}</button></div></div>;
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
  useEffect(() => { if (view !== 'people') return undefined; const id = setInterval(reload, 5000); return () => clearInterval(id); }, [view, reload]);
  const displayImages = useMemo(() => blocks.flatMap((b) => b.images).filter((img) => img.tags?.some((tag) => ['portrait:generated', 'portrait:display-ready'].includes(tag.name))), [blocks]);
  const [slide, setSlide] = useState(0);
  useEffect(() => { const id = setInterval(() => setSlide((n) => displayImages.length ? (n + 1) % displayImages.length : 0), 18000); return () => clearInterval(id); }, [displayImages.length]);
  return <main>
    <nav><button className="brand" onClick={() => setView('frame')}><Icon>✦</Icon><span>AI Portrait<small>LIVING MEMORIES</small></span></button><div className="navTabs">{[['frame','Photo frame'],['camera','Camera'],['people','People'],['create','Create']].map(([id,label]) => <button key={id} className={view === id ? 'active' : ''} onClick={() => setView(id)}>{label}</button>)}</div><button className="settings" onClick={() => setView('people')}>⚙</button></nav>
    {view === 'frame' && <section className="frameView">{displayImages.length ? <><img src={displayImages[slide]?.url} /><div className="frameGradient" /><div className="frameCaption"><span>PORTRAIT OF THE MOMENT</span><strong>A memory that never happened.<br />But should have.</strong></div><div className="dots">{displayImages.map((_, i) => <i className={i === slide ? 'on' : ''} key={i} />)}</div></> : <div className="emptyFrame"><Icon>✦</Icon><h1>Your photo frame is ready.</h1><p>Meet a few people through the camera and create the first impossible memory.</p><button onClick={() => setView('camera')}>Open camera</button></div>}</section>}
    <div className={view === 'camera' ? '' : 'cameraBackground'}><CameraView interval={status.capture_interval_seconds} captureReady={status.capture_ready} captureBlocker={status.capture_blocker} toast={toast} onCaptured={(result) => { reload(); if (view === 'frame' && result.photo_ids?.length) { setView('camera'); setTimeout(() => setView((current) => current === 'camera' ? 'frame' : current), 9500); } if (!result.person && result.saved) toast('New person found — open People to give them a name.'); }} /></div>
    {view === 'people' && <section className="workspace"><div className="sectionHead"><span className="eyebrow">GALLERY & IDENTITY</span><h2>The people behind the stories</h2><p>Every accepted ambient capture appears here. Select one to name the person and teach relationships.</p></div><div className="split"><div><Gallery blocks={blocks} library={library} selected={selectedImage?.id} onSelect={setSelectedImage} /></div><PeoplePanel library={library} selectedImage={selectedImage} reload={reload} toast={toast} /></div></section>}
    {view === 'create' && <CreatePanel library={library} toast={toast} onGenerated={() => { reload(); toast('New portrait created and added to the photo frame.'); }} />}
    {notice && <div className="toast">{notice}</div>}
  </main>;
}
