'use strict';
/** FRAME70 experiment. All image processing stays on the device; no model/API calls. */
const $ = id => document.getElementById(id);
const UI={video:$('camera'),overlay:$('guides'),sample:$('sample'),view:$('viewfinder'),status:$('statusText'),chip:$('statusChip'),bubble:$('guideBubble'),bubbleText:$('guideText'),bubbleIcon:$('guideIcon'),focusLabel:$('focusLabel'),analyze:$('analyzeStatus'),permission:$('permissionPanel'),shutter:$('shutter'),cameraPage:$('cameraPage'),resultPage:$('resultPage'),result:$('resultImage'),resultBadge:$('resultBadge'),summary:$('resultSummary'),toast:$('toast'),help:$('helpModal')};
const S={stream:null,track:null,imageCapture:null,facing:'environment',started:false,guideOn:true,processing:false,lastAnalysis:null,manualTarget:null,faceDetector:null,facePending:false,faceLast:0,faceBox:null,lastFrame:0,analysisEvery:185,guideCandidate:'',candidateSince:0,guideShown:'',guideLast:0,rollHistory:[],shot:null,urls:[],fileImport:false};
const LIMIT=(v,a,b)=>Math.max(a,Math.min(b,v));
const lerp=(a,b,t)=>a+(b-a)*t;
const showToast=(msg)=>{UI.toast.textContent=msg;UI.toast.classList.add('show');clearTimeout(showToast.timer);showToast.timer=setTimeout(()=>UI.toast.classList.remove('show'),2400)};
function clearPhotoUrls(){for(const u of S.urls)URL.revokeObjectURL(u);S.urls=[]}
function setPermission(title,copy){UI.permission.hidden=false;$('permissionTitle').textContent=title;$('permissionCopy').textContent=copy;UI.status.textContent='待機中'}
function clearPermission(){UI.permission.hidden=true}
function setStatus(msg){UI.status.textContent=msg}
function setGuide(icon,msg){if(!S.guideOn){UI.bubble.classList.remove('visible');return}UI.bubbleIcon.textContent=icon;UI.bubbleText.textContent=msg;UI.bubble.classList.add('visible')}
function hideGuide(){UI.bubble.classList.remove('visible')}
function stopCamera(){if(S.stream){S.stream.getTracks().forEach(t=>t.stop());S.stream=null;S.track=null;S.started=false}UI.video.srcObject=null}
async function startCamera(){
  if(S.processing)return;stopCamera();clearPermission();setStatus('カメラを起動中');$('shutter').disabled=true;
  if(!navigator.mediaDevices?.getUserMedia){setPermission('この環境ではカメラを開けません','Android ChromeなどのブラウザーでHTTPSのページとして開いてください。写真を選んで補正を試すこともできます。');return}
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:S.facing},width:{ideal:1920},height:{ideal:1440},frameRate:{ideal:30,max:30}}});
    S.stream=stream;S.track=stream.getVideoTracks()[0];UI.video.srcObject=stream;
    await UI.video.play();S.imageCapture=typeof ImageCapture==='function'?new ImageCapture(S.track):null;
    S.started=true;S.lastFrame=0;S.faceBox=null;S.lastAnalysis=null;S.manualTarget=null;S.candidateSince=0;
    UI.video.style.transform=S.facing==='user'?'scaleX(-1)':'none';
    setStatus('撮影できます');UI.analyze.textContent='構図を解析中';$('shutter').disabled=false;
    initFaceDetector();requestAnimationFrame(loop);
  }catch(e){console.warn('Camera unavailable',e);const msg=e?.name==='NotAllowedError'?'ブラウザーのカメラ許可をONにしてください。':e?.name==='NotFoundError'?'利用可能なカメラが見つかりません。':'カメラが起動できませんでした。HTTPSで開き直すか、写真を読み込んでください。';setPermission('カメラにアクセスできません',msg);$('shutter').disabled=true}
}
function initFaceDetector(){try{if(!S.faceDetector && typeof FaceDetector!=='undefined'){S.faceDetector=new FaceDetector({maxDetectedFaces:2,fastMode:true})}}catch(e){S.faceDetector=null}}
/** Browser preview uses object-fit:cover. Map sampling to the exact visible source rectangle. */
function visibleVideoRect(){const w=UI.video.videoWidth,h=UI.video.videoHeight;const dw=UI.view.clientWidth,dh=UI.view.clientHeight;if(!w||!h||!dw||!dh)return null;const ratio=dw/dh,srcRatio=w/h;if(srcRatio>ratio){const sw=h*ratio;return {x:(w-sw)/2,y:0,w:sw,h}}const sh=w/ratio;return {x:0,y:(h-sh)/2,w,h:sh}}
function drawVisible(ctx,w,h){const r=visibleVideoRect();if(!r)return false;ctx.drawImage(UI.video,r.x,r.y,r.w,r.h,0,0,w,h);return true}
function computeImageFeatures(image){
  const {data,width:w,height:h}=image;const count=w*h;const L=new Float32Array(count),sat=new Float32Array(count);
  let mean=0;for(let i=0,j=0;i<count;i++,j+=4){const r=data[j],g=data[j+1],b=data[j+2];const y=.2126*r+.7152*g+.0722*b;L[i]=y;mean+=y;sat[i]=Math.max(r,g,b)-Math.min(r,g,b)}mean/=count;
  const cols=10,rows=14,cell=new Float32Array(cols*rows),energy=new Float32Array(cols*rows);let maxGradient=0;const edges=[];
  for(let y=2;y<h-2;y+=2){for(let x=2;x<w-2;x+=2){const i=y*w+x,dx=L[i+1]-L[i-1],dy=L[i+w]-L[i-w];const grad=Math.abs(dx)+Math.abs(dy);const ci=LIMIT(Math.floor(x/w*cols),0,cols-1),ri=LIMIT(Math.floor(y/h*rows),0,rows-1),k=ri*cols+ci;
    const contrast=Math.abs(L[i]-mean),color=sat[i];const score=.44*LIMIT(grad,0,130)+.35*contrast+.21*color;
    cell[k]+=score;energy[k]+=1;maxGradient=Math.max(maxGradient,grad);
    if(Math.abs(dy)>32&&Math.abs(dy)>Math.abs(dx)*1.45&&x%4===0&&y%4===0)edges.push({x,y,weight:LIMIT(Math.abs(dy),0,110)});
  }}
  let avg=0;for(let k=0;k<cell.length;k++){cell[k]/=Math.max(1,energy[k]);avg+=cell[k]}avg/=cell.length;
  // Smooth a small saliency neighborhood so isolated highlights don't dominate.
  let best=-1,bx=.5,by=.5,sum=0,wx=0,wy=0;
  for(let r=1;r<rows-1;r++)for(let c=1;c<cols-1;c++){let local=0;for(let v=-1;v<=1;v++)for(let u=-1;u<=1;u++)local+=cell[(r+v)*cols+c+u];local/=9;const cx=(c+.5)/cols,cy=(r+.5)/rows;const centerBias=1-.22*Math.hypot((cx-.5)*1.4,(cy-.48)*1.0);
    const v=local*centerBias;if(v>best){best=v;bx=cx;by=cy}const sal=Math.max(0,local-avg*.68);sum+=sal;wx+=sal*cx;wy+=sal*cy;
  }
  const target={x:lerp(bx,sum?wx/sum:.5,.25),y:lerp(by,sum?wy/sum:.5,.25),source:'saliency',confidence:LIMIT((best/Math.max(1,avg)-1)*.5,0,.65)};
  // Hough estimate of prominent near-horizontal lines; don't claim a horizon if weak.
  const slopes=[-.18,-.14,-.10,-.07,-.045,-.025,0,.025,.045,.07,.10,.14,.18],acc=slopes.map(()=>new Float32Array(h+50));
  for(const e of edges){for(let s=0;s<slopes.length;s++){const bin=Math.round(e.y-slopes[s]*(e.x-w/2))+25;if(bin>=0&&bin<h+50)acc[s][bin]+=e.weight}}
  let highest=0,second=0,sl=0,ab=0;for(let s=0;s<slopes.length;s++)for(let b=0;b<h+50;b++){const val=acc[s][b];if(val>highest){second=highest;highest=val;sl=s;ab=b}else if(val>second&&Math.abs(slopes[s]-slopes[sl])>.05)second=val}
  // At least ~35% of image width must support a single line.
  const strong=highest>w*13;
  const degree=strong?Math.atan(slopes[sl])*180/Math.PI:0;
  return{target,lines:{degree,confidence:strong?LIMIT((highest/Math.max(1,w*18)-.6)*.7,0,.92):0},saliency:{cell:Array.from(cell),rows,cols,average:avg},brightness:mean};
}
async function detectFaces(bitmap,w,h){if(!S.faceDetector||S.facePending||performance.now()-S.faceLast<700)return;S.facePending=true;S.faceLast=performance.now();try{const faces=await S.faceDetector.detect(bitmap);if(faces?.length){const f=faces[0].boundingBox;S.faceBox={x:(f.x+f.width/2)/w,y:(f.y+f.height/2)/h,w:f.width/w,h:f.height/h,t:performance.now()}}else S.faceBox=null}catch(e){S.faceDetector=null}finally{S.facePending=false}}
function decideGuidance(features){const t=features.target;let target=S.manualTarget?{...S.manualTarget,confidence:1,source:'tap'}:t;
 if(!S.manualTarget&&S.faceBox&&performance.now()-S.faceBox.t<1800)target={x:S.faceBox.x,y:S.faceBox.y,confidence:.95,source:'face',width:S.faceBox.w,height:S.faceBox.h};
 const roll=features.lines;let action=null;
 if(roll.confidence>.52&&Math.abs(roll.degree)>2.7&&Math.abs(roll.degree)<10)action={key:roll.degree>0?'level-left':'level-right',icon:'⤾',message:'スマホの傾きを少し直す',kind:'level',degree:roll.degree};
 if(!action&&target.confidence>.38){
   // Only recommend when the subject is noticeably close to an edge.
   if(target.x>.78)action={key:'right',icon:'→',message:'少し右を向けて',kind:'pan'};
   else if(target.x<.22)action={key:'left',icon:'←',message:'少し左を向けて',kind:'pan'};
   else if(target.y<.17)action={key:'up',icon:'↑',message:'少し上を向けて',kind:'tilt'};
   else if(target.y>.84)action={key:'down',icon:'↓',message:'少し下を向けて',kind:'tilt'};
 }
 return{target,action};
}
function renderOverlay(target,action){const canvas=UI.overlay,rect=UI.view.getBoundingClientRect();const dpr=Math.min(window.devicePixelRatio||1,2);if(canvas.width!==Math.round(rect.width*dpr)||canvas.height!==Math.round(rect.height*dpr)){canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr)}const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,rect.width,rect.height);if(!S.guideOn||!S.started)return;
 const w=rect.width,h=rect.height;
 // Lightweight composition guide, mostly invisible unless it provides an actionable cue.
 if(action){c.strokeStyle='rgba(177,244,205,.55)';c.setLineDash([5,8]);c.lineWidth=1;c.beginPath();c.moveTo(w/3,h*.22);c.lineTo(w/3,h*.78);c.moveTo(w*2/3,h*.22);c.lineTo(w*2/3,h*.78);c.moveTo(w*.12,h/3);c.lineTo(w*.88,h/3);c.moveTo(w*.12,h*2/3);c.lineTo(w*.88,h*2/3);c.stroke();c.setLineDash([])}
 if(target&&target.confidence>.38){const x=target.x*w,y=target.y*h;c.beginPath();c.arc(x,y,17,0,Math.PI*2);c.strokeStyle=S.manualTarget?'#b1f4cd':'rgba(177,244,205,.7)';c.lineWidth=1.5;c.stroke();c.beginPath();c.arc(x,y,3,0,Math.PI*2);c.fillStyle='#b1f4cd';c.fill()}
}
function updateGuide(g){const now=performance.now(),key=g.action?.key||'none';if(S.guideCandidate!==key){S.guideCandidate=key;S.candidateSince=now}if(now-S.candidateSince<480)return;if(key==='none'){if(now-S.guideLast>850){hideGuide();S.guideShown=''}return}if(S.guideShown!==key){S.guideShown=key;S.guideLast=now;setGuide(g.action.icon,g.action.message)}}
function loop(t){if(!S.started)return;requestAnimationFrame(loop);if(S.processing||UI.cameraPage.hidden||t-S.lastFrame<S.analysisEvery)return;S.lastFrame=t;
 const canvas=UI.sample,ctx=canvas.getContext('2d',{willReadFrequently:true});const rect=visibleVideoRect();if(!rect)return;const maxSide=160;const sc=Math.min(1,maxSide/Math.max(rect.w,rect.h));const w=Math.max(70,Math.round(rect.w*sc)),h=Math.max(70,Math.round(rect.h*sc));canvas.width=w;canvas.height=h;
 if(!drawVisible(ctx,w,h))return;const img=ctx.getImageData(0,0,w,h);const feat=computeImageFeatures(img);if(S.faceDetector&&!S.facePending){try{createImageBitmap(canvas).then(b=>{detectFaces(b,w,h).finally(()=>b.close())})}catch(e){}}
 const g=decideGuidance(feat);S.lastAnalysis={...feat,...g,timestamp:t,visibleAspect:rect.w/rect.h};renderOverlay(g.target,g.action);updateGuide(g);UI.analyze.textContent=g.target.source==='face'?'顔を認識中':S.manualTarget?'主役を指定中':'構図を解析中';
}
function getSelectedRect(bitmap){const aspect=S.lastAnalysis?.visibleAspect||UI.view.clientWidth/Math.max(1,UI.view.clientHeight);const vw=bitmap.width,vh=bitmap.height;if(vw/vh>aspect){const rw=vh*aspect;return{x:(vw-rw)/2,y:0,w:rw,h:vh}}const rh=vw/aspect;return{x:0,y:(vh-rh)/2,w:vw,h:rh}}
function cropOptimizer(features){
 const target=features?.target||{x:.5,y:.5,confidence:0};if(target.confidence<.35)return{rect:{x:0,y:0,w:1,h:1},gain:0};
 const candidates=[{x:0,y:0,w:1,h:1}];for(const zoom of [.93,.87])for(const tx of [.35,.5,.65])for(const ty of [.35,.5,.65]){const x=LIMIT(target.x-tx*zoom,0,1-zoom),y=LIMIT(target.y-ty*zoom,0,1-zoom);candidates.push({x,y,w:zoom,h:zoom})}
 const sal=features?.saliency,score=rect=>{
   const posX=(target.x-rect.x)/rect.w,posY=(target.y-rect.y)/rect.h;
   const placement=Math.max(...[.33,.5,.67].map(v=>1-Math.abs(posX-v)*2.0)) + .34*(1-Math.abs(posY-.45));
   const borderPenalty=Math.max(0,.16-posX)*4+Math.max(0,posX-.84)*4+Math.max(0,.12-posY)*3+Math.max(0,posY-.9)*3;
   let integrity=0;if(sal){let keep=0,total=0;for(let y=0;y<sal.rows;y++)for(let x=0;x<sal.cols;x++){const weight=Math.max(0,sal.cell[y*sal.cols+x]-sal.average*.62),px=(x+.5)/sal.cols,py=(y+.5)/sal.rows;total+=weight;if(px>=rect.x&&px<=rect.x+rect.w&&py>=rect.y&&py<=rect.y+rect.h)keep+=weight}integrity=total>0?(1-keep/total)*.58:0}
   return .60*placement+(.22*(1-borderPenalty))-integrity-2.1*(1-rect.w);
 };
 const baseline=score(candidates[0]);let best=candidates[0],bestScore=baseline;for(const r of candidates.slice(1)){const v=score(r);if(v>bestScore){best=r;bestScore=v}}const gain=bestScore-baseline;
 if(gain>.06)return{rect:best,gain};
 // Explicitly selected subject: offer a conservative tighter composition only
 // when it remains safely away from the frame boundary.
 if(target.source==='tap'&&target.x>.20&&target.x<.80&&target.y>.20&&target.y<.80){
   const z=.90,rx=LIMIT(target.x-z*.5,0,1-z),ry=LIMIT(target.y-z*.5,0,1-z);
   const safer={x:rx,y:ry,w:z,h:z};const pX=(target.x-rx)/z,pY=(target.y-ry)/z;
   if(pX>.18&&pX<.82&&pY>.18&&pY<.82)return{rect:safer,gain:.07};
 }
 return{rect:candidates[0],gain:0};
}
async function captureFromCamera(){if(S.processing)return;S.processing=true;UI.shutter.classList.add('busy');$('shutter').disabled=true;setStatus('撮影中');
 try{let blob=null;
  if(S.imageCapture){try{blob=await S.imageCapture.takePhoto()}catch(e){console.info('Using full video-frame capture fallback:',e?.name)}}
  if(!blob){const width=UI.video.videoWidth,height=UI.video.videoHeight;if(!width||!height)throw new Error('カメラ映像がありません');const c=document.createElement('canvas');c.width=width;c.height=height;c.getContext('2d').drawImage(UI.video,0,0,width,height);blob=await new Promise((resolve,reject)=>c.toBlob(x=>x?resolve(x):reject(new Error('保存用画像を作成できません')),'image/jpeg',.97))}
  await processPhoto(blob,true);
 }catch(e){console.error(e);showToast('撮影できませんでした。もう一度お試しください。');setStatus('撮影できます')}finally{UI.shutter.classList.remove('busy');$('shutter').disabled=false;S.processing=false}
}
async function processPhoto(blob,fromCamera){clearPhotoUrls();const bitmap=await createImageBitmap(blob);try{
  // Analyze the actual captured image; high resolution stays local.
  const low=document.createElement('canvas');const ratio=bitmap.width/bitmap.height;low.width=ratio>=1?176:Math.round(176*ratio);low.height=ratio>=1?Math.round(176/ratio):176;const cx=low.getContext('2d',{willReadFrequently:true});cx.drawImage(bitmap,0,0,low.width,low.height);const feature=computeImageFeatures(cx.getImageData(0,0,low.width,low.height));
  let photoTarget=S.manualTarget||S.lastAnalysis?.target||feature.target;
  if(fromCamera&&photoTarget){const visible=getSelectedRect(bitmap);photoTarget={...photoTarget,x:(visible.x+photoTarget.x*visible.w)/bitmap.width,y:(visible.y+photoTarget.y*visible.h)/bitmap.height}}
  if(!fromCamera)photoTarget=feature.target;
  const params={...feature,target:photoTarget};const suggest=cropOptimizer(params);
  const roll=feature.lines.confidence>.6&&Math.abs(feature.lines.degree)>2.8&&Math.abs(feature.lines.degree)<6?feature.lines.degree:0;
  // Rotate only if confidence is high; use a conservative safe zoom to prevent corners.
  const angle=roll*Math.PI/180;let rect={...suggest.rect};if(roll){const rw=bitmap.width,rh=bitmap.height;const safe=Math.max(Math.abs(Math.cos(angle))+rh/rw*Math.abs(Math.sin(angle)),Math.abs(Math.cos(angle))+rw/rh*Math.abs(Math.sin(angle)));const z=1/safe;if(z>=.84){rect={x:(1-z*rect.w)/2+z*rect.x,y:(1-z*rect.h)/2+z*rect.y,w:rect.w*z,h:rect.h*z}}}
  const changed=Math.abs(rect.w-1)>.01||Math.abs(roll)>0.01;
  // Keep true original as a Blob URL; only render corrected image when necessary.
  const origUrl=URL.createObjectURL(blob);S.urls.push(origUrl);
  let adjustedBlob=blob;
  if(changed){const cw=Math.max(1,Math.round(bitmap.width*rect.w)),ch=Math.max(1,Math.round(bitmap.height*rect.h));let maxSize=4200,scale=Math.min(1,maxSize/Math.max(cw,ch));const out=document.createElement('canvas');out.width=Math.max(1,Math.round(cw*scale));out.height=Math.max(1,Math.round(ch*scale));const c=out.getContext('2d');c.imageSmoothingQuality='high';
    if(roll){c.translate(out.width/2,out.height/2);c.rotate(-angle);c.drawImage(bitmap,-(rect.x+rect.w/2)*bitmap.width*scale,-(rect.y+rect.h/2)*bitmap.height*scale,bitmap.width*scale,bitmap.height*scale)}
    else c.drawImage(bitmap,rect.x*bitmap.width,rect.y*bitmap.height,rect.w*bitmap.width,rect.h*bitmap.height,0,0,out.width,out.height);
    adjustedBlob=await new Promise((resolve,reject)=>out.toBlob(x=>x?resolve(x):reject(new Error('補正画像を作成できません')),'image/jpeg',.95));
  }
  const adjUrl=changed?URL.createObjectURL(adjustedBlob):origUrl;if(changed)S.urls.push(adjUrl);S.shot={original:blob,adjusted:adjustedBlob,origUrl,adjUrl,changed,roll,zoom:rect.w,feat:params};
  UI.cameraPage.hidden=true;UI.resultPage.hidden=false;UI.result.src=adjUrl;UI.resultBadge.textContent=changed?'構図を自動補正':'オリジナル構図';
  UI.summary.textContent=changed?`${rect.w<.99?'フレーミング補正':''}${rect.w<.99&&roll?' ＋ ':''}${roll?'水平補正':''}を適用しました`:'不確かな補正は行わず、元画像を保持しました';showAdjusted();stopCamera();
 }finally{bitmap.close?.()}}
function showAdjusted(){if(!S.shot)return;UI.result.src=S.shot.adjUrl;$('showAdjusted').classList.add('selected');$('showOriginal').classList.remove('selected');UI.resultBadge.textContent=S.shot.changed?'自動補正後':'オリジナル構図'}
function showOriginal(){if(!S.shot)return;UI.result.src=S.shot.origUrl;$('showOriginal').classList.add('selected');$('showAdjusted').classList.remove('selected');UI.resultBadge.textContent='元画像（無加工）'}
async function sharePhoto(){if(!S.shot)return;const original=$('showOriginal').classList.contains('selected');const blob=original?S.shot.original:S.shot.adjusted;const date=new Date().toISOString().replaceAll(':','-').replace('T','_').slice(0,19);const file=new File([blob],`FRAME70_${date}${original?'_original':''}.jpg`,{type:'image/jpeg'});
 try{if(navigator.share&&navigator.canShare?.({files:[file]})){await navigator.share({files:[file],title:'FRAME70で撮影した写真'});return}}
 catch(e){if(e?.name==='AbortError')return;console.info('Share unavailable',e)}
 const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download=file.name;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);showToast('写真をダウンロードしました')}
async function retake(){UI.resultPage.hidden=true;UI.cameraPage.hidden=false;S.shot=null;clearPhotoUrls();await startCamera()}
UI.view.addEventListener('click',e=>{if(!S.started||e.target.closest('button')||e.target.closest('.permissionPanel'))return;const r=UI.view.getBoundingClientRect(),x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height;if(y<.15||y>.9)return;S.manualTarget={x:S.facing==='user'?1-x:x,y,source:'tap',confidence:1};UI.focusLabel.textContent='主役を指定しました（再タップで変更）';showToast('主役を指定しました')});
$('shutter').addEventListener('click',captureFromCamera);$('retryCamera').addEventListener('click',startCamera);$('flipButton').addEventListener('click',async()=>{S.facing=S.facing==='environment'?'user':'environment';await startCamera()});
$('modeButton').addEventListener('click',()=>{S.guideOn=!S.guideOn;$('modeLabel').textContent=S.guideOn?'ガイドON':'ガイドOFF';if(!S.guideOn){hideGuide();UI.overlay.getContext('2d').clearRect(0,0,UI.overlay.width,UI.overlay.height)}else if(S.lastAnalysis)renderOverlay(S.lastAnalysis.target,S.lastAnalysis.action);showToast(S.guideOn?'構図ガイド ON':'構図ガイド OFF')});
$('showAdjusted').addEventListener('click',showAdjusted);$('showOriginal').addEventListener('click',showOriginal);$('saveButton').addEventListener('click',sharePhoto);$('retakeButton').addEventListener('click',retake);$('backButton').addEventListener('click',retake);
$('infoBtn').addEventListener('click',()=>{if(!S.shot)return;showToast(S.shot.changed?'補正前の写真も「元画像」から保存可能です':'不確かな補正は実施していません')});
$('helpButton').addEventListener('click',()=>UI.help.hidden=false);$('closeHelp').addEventListener('click',()=>UI.help.hidden=true);$('helpDone').addEventListener('click',()=>UI.help.hidden=true);UI.help.addEventListener('click',e=>{if(e.target===UI.help)UI.help.hidden=true});
$('fileFallback').addEventListener('click',()=>$('importFile').click());$('importFile').addEventListener('change',async e=>{const file=e.target.files?.[0];if(!file)return;S.processing=true;try{await processPhoto(file,false)}catch(ex){console.error(ex);showToast('画像を読み込めませんでした')}finally{S.processing=false;e.target.value=''}});
window.addEventListener('pagehide',()=>{stopCamera();clearPhotoUrls()});document.addEventListener('visibilitychange',()=>{if(document.hidden&&S.started)stopCamera();else if(!document.hidden&&!S.started&&!UI.cameraPage.hidden)startCamera()});
if('serviceWorker' in navigator&&location.protocol==='https:'){navigator.serviceWorker.register('./sw.js').catch(console.warn)}
startCamera();