# FRAME70 beta — instant composition camera

**Status:** functional camera/capture/crop proof-of-concept; NOT the PhotoGuider trained model or a native Android camera.

## Try on phone

Host this folder over HTTPS (Render static site, GitHub Pages, or any HTTPS static host). Open in Android Chrome, grant camera access, then shoot. Compare corrected / original and use Save/Share. Tap the primary subject if automatic saliency misses it. It does not upload any camera images or call an LLM.

## Features

- Rear/front live camera via `getUserMedia`, no network inference.
- Full-resolution photo via `ImageCapture.takePhoto` when supported; high resolution video-frame fallback.
- Low-resolution on-device analysis for composition; manual main-subject selection; optional browser FaceDetector API.
- Simple horizontal-line estimation; rule-based framing guides for camera pan/tilt.
- Geometry-aware crop optimization; conservative optional roll correction, side-by-side original vs corrected (toggle).
- Save to local downloads or Android native share sheet; offline service worker after initial load.
- No API keys, no backend, no photo upload.

## Limitations / known gaps

- The PhotoGuider released pipeline requires multiple vision-model environments with GPU; this code uses original lightweight heuristics inspired by its Subject/Composition/Integrity evaluation, not its actual trained ranker/checkpoints.
- Saliency centroid isn't a full object detector; it can target a bright or high-detail region rather than the intended subject. Tap to indicate the subject.
- The browser `FaceDetector` API is optional/rare; many Android versions have no face detector available.
- No parallax-based *physical camera position* guidance yet; pan/tilt guides are approximate. Background collision advice is not provided.
- Camera `ImageCapture` support, hardware ISP quality, EXIF retention, offline install support and share/save behavior vary by phone/browser. Browser camera cannot promise native HDR/night mode parity. Captured corrected JPEG is re-encoded without original EXIF; unedited original remains available for download.
- No real-device camera performance verified in this environment. Test Chrome Android in good lighting with portrait-held phone.

## Run locally

`python -m http.server 8080` then open `http://localhost:8080` on the same device (camera APIs require secure context: HTTPS or localhost). A LAN http://IP:8080 URL is not enough for camera permissions.

## Suggested follow-up

Native Android CameraX + ML Kit face/object + motion sensors/ARCore, with instrumentation tests measuring capture latency and camera motion guidance, once the core product UX passes user evaluation.

Research: [PhotoGuider (Apache-2.0)](https://github.com/NTUneillee/PhotoGuider), [CameraX](https://developer.android.com/media/camera/camerax/architecture).