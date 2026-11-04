"use strict";
const $ = id => document.getElementById(id);
let state = null, photo = null, image = null, anchor = [.5,.5], force = [.3,0];
let busy = false, dragging = false, output = null, pinned = null, active = null, operation = 0;
let playing = false;
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
async function api(path, data) {
  const response = await fetch(path, {method:data === undefined ? "GET" : "POST",
    headers:data === undefined ? {} : {"Content-Type":"application/json","X-Demo-Token":state.token},
    body:data === undefined ? undefined : JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw Error(result.error || "request failed");
  return result;
}
function message(text, error=false) {$("status").textContent=text; $("status").classList.toggle("error",error);}
function controls() {
  for (const id of ["open-photo","empty-open","photos","recordings","strength","reverse","zero","caption","seed"])
    $(id).disabled=busy;
  $("generate").disabled=busy || !photo || !$("caption").value.trim() || !state?.model?.identity;
  $("cancel").hidden=!busy;
}
function dirty() {if(output) $("source").textContent="previous action - generate to update"; controls();}
function draw() {
  const canvas=$("input"),ctx=canvas.getContext("2d");
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!image)return;
  ctx.drawImage(image,0,0,canvas.width,canvas.height);
  const w=canvas.width,h=canvas.height,x=anchor[0]*w,y=anchor[1]*h,ex=x+force[0]*w/4,ey=y+force[1]*w/4;
  ctx.lineWidth=3;ctx.strokeStyle="#d9f098";ctx.fillStyle="#d9f098";
  ctx.shadowColor="#111";ctx.shadowBlur=4;
  ctx.beginPath();ctx.arc(x,y,9,0,Math.PI*2);ctx.stroke();
  if(Math.hypot(...force)>0){
    ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(ex,ey);ctx.stroke();
    const a=Math.atan2(ey-y,ex-x);ctx.beginPath();ctx.moveTo(ex,ey);
    ctx.lineTo(ex-14*Math.cos(a-.4),ey-14*Math.sin(a-.4));
    ctx.lineTo(ex-14*Math.cos(a+.4),ey-14*Math.sin(a+.4));ctx.closePath();ctx.fill();
  }
  ctx.shadowBlur=0;
  const percent=Math.round(Math.hypot(...force)*100);
  $("strength").value=percent;$("strength-value").textContent=percent+"%";
}
function stop() {playing=false;for(const v of document.querySelectorAll("video"))v.pause();$("play").textContent="play";}
function clearResults() {
  stop();output=null;pinned=null;document.body.classList.remove("has-pinned");$("pinned").hidden=true;$("pinned-results").replaceChildren();
  $("results").replaceChildren();$("results").classList.add("empty-futures");
  for(let i=0;i<4;i++){const d=document.createElement("div");d.className="future-empty";
    const s=document.createElement("span");s.textContent=String(i+1).padStart(2,"0");
    const p=document.createElement("p");p.textContent="future "+(i+1);d.append(s,p);$("results").append(d);}
  for(const id of ["play","timeline","pin"])$(id).disabled=true;
  $("source").textContent="awaiting a push";$("timeline").value=0;$("time").textContent="0.00s";
}
async function selectPhoto(value) {
  busy=true;controls();
  try {
  stop();const next=await api("/api/photos/"+value);const nextImage=new Image();
  nextImage.src=next.image;await nextImage.decode();photo=next;image=nextImage;
  clearResults();anchor=next.anchor || [.5,.5];force=next.force || [.3,0];
  $("caption").value=next.caption || "";$("input").hidden=false;$("input-empty").hidden=true;
  $("point-help").textContent="click where to apply the push, then drag to aim it.";
  draw();message("set the push and describe the scene.");
  } finally {busy=false;controls();}
}
function photoOptions() {
  const selected=photo?.id;$("photos").replaceChildren();
  state.photos.forEach((p,i)=>{const option=document.createElement("option");option.value=p.id;
    option.textContent=p.label || "photo "+(i+1);$("photos").append(option);});
  $("photos").hidden=state.photos.length<2;if(selected)$("photos").value=selected;
}
function recordingOptions() {
  const selected=$("recordings").value;$("recordings").replaceChildren();
  const empty=document.createElement("option");empty.value="";empty.textContent="open a recorded run...";$("recordings").append(empty);
  for(const row of state.recordings){const option=document.createElement("option");option.value=row.id;
    option.textContent="["+row.force.map(v=>v.toFixed(2)).join(", ")+"] "+row.prompt.slice(0,60);$("recordings").append(option);}
  $("recordings").hidden=!state.recordings.length;if(selected)$("recordings").value=selected;
}
async function refresh() {
  state=await api("/api/state");
  $("input").width=state.size[0];$("input").height=state.size[1];draw();
  document.body.classList.toggle("backend-cogvideo",state.backend.startsWith("force-prompting"));
  $("zero").textContent=state.minimum_label;
  $("attribution").textContent=(state.backend.startsWith("force-prompting") ? "force prompting / external pretrained model / " : "streamforce / external pretrained model / ")+state.size.join(" x ")+" / "+state.steps+" steps";
  $("worker").textContent=state.ready ? "gpu worker ready" : state.stage;
  $("worker").classList.toggle("ready",state.ready);photoOptions();recordingOptions();controls();
}
function point(event) {
  const box=$("input").getBoundingClientRect();
  return [Math.max(0,Math.min(1,(event.clientX-box.left)/box.width)),
    Math.max(0,Math.min(1,(event.clientY-box.top)/box.height))];
}
$("input").addEventListener("pointerdown",event=>{
  if(busy||!photo)return;anchor=point(event);dragging=true;$("input").setPointerCapture(event.pointerId);dirty();draw();
});
$("input").addEventListener("pointermove",event=>{
  if(!dragging)return;const p=point(event);let x=(p[0]-anchor[0])*4,y=(p[1]-anchor[1])*$("input").height/($("input").width/4);
  const n=Math.hypot(x,y);if(n>1){x/=n;y/=n;}force=[x,y];draw();
});
$("input").addEventListener("pointerup",()=>{dragging=false;});
$("input").addEventListener("pointercancel",()=>{dragging=false;});
$("input").addEventListener("keydown",event=>{
  if(busy||!photo||!event.key.startsWith("Arrow"))return;event.preventDefault();
  const step=event.shiftKey ? .01 : .05;const v={ArrowRight:[step,0],ArrowLeft:[-step,0],ArrowDown:[0,step],ArrowUp:[0,-step]}[event.key];
  let [x,y]=[force[0]+v[0],force[1]+v[1]],n=Math.hypot(x,y);if(n>1){x/=n;y/=n;}
  force=[x,y];dirty();draw();
});
$("strength").oninput=()=>{const n=Math.hypot(...force),p=Number($("strength").value)/100;
  force=n ? force.map(v=>v/n*p) : [p,0];dirty();draw();};
$("reverse").onclick=()=>{force=force.map(v=>-v);dirty();draw();};
$("zero").onclick=()=>{force=[0,0];dirty();draw();};
$("caption").oninput=dirty;$("seed").oninput=dirty;
$("open-photo").onclick=$("empty-open").onclick=()=>$("upload").click();
$("photos").onchange=()=>selectPhoto($("photos").value).catch(e=>message(e.message,true));
$("upload").onchange=async()=>{
  const file=$("upload").files[0];if(!file)return;
  busy=true;controls();
  try {
    if(file.size>12*1024*1024)throw Error("choose a photograph up to 12 MiB");
    const data=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=()=>reject(Error("could not read the photo"));r.readAsDataURL(file);});
    const value=await api("/api/photos",{image:data});await refresh();await selectPhoto(value.id);$("photos").value=value.id;
  }catch(error){message(error.message,true);}finally{$("upload").value="";busy=false;controls();}
};
function cards(target,item) {
  target.replaceChildren();target.classList.remove("empty-futures");
  item.result.videos.forEach((row,i)=>{
    const card=document.createElement("div");card.className="future";
    const video=document.createElement("video");video.src="/media/"+item.id+"/"+row.file;
    video.muted=true;video.playsInline=true;video.preload="auto";
    video.setAttribute("aria-label","future "+(i+1)+", seed "+row.seed);
    const footer=document.createElement("div");footer.className="video-footer";
    const title=document.createElement("b");title.textContent="future "+(i+1);
    const seed=document.createElement("span");seed.textContent="seed "+row.seed;
    const expand=document.createElement("button");expand.textContent="expand";
    expand.onclick=()=>{if(video.requestFullscreen)video.requestFullscreen().catch(()=>{});};
    video.addEventListener("fullscreenchange",()=>{video.controls=document.fullscreenElement===video;});
    footer.append(title,seed,expand);card.append(video,footer);target.append(card);
  });
}
function showResult(job,source) {
  if(job.result.videos.length!==4||!job.result.videos.every((v,i)=>v.seed===job.result.request.seeds[i]))throw Error("the worker did not return the declared four samples");
  stop();output={id:job.id,result:job.result};cards($("results"),output);
  $("source").textContent=source;
  for(const id of ["play","timeline","pin"])$(id).disabled=false;
  $("timeline").max=job.result.frames-1;$("timeline").value=0;$("time").textContent="0.00s";
  $("details").textContent=JSON.stringify(job.result,null,2);
  message("four futures ready. pin this action, then change the push to compare.");
  pairMessage();
  if(!reduced.matches){const video=$("results").querySelector("video");if(video)video.addEventListener("canplay",()=>{if(output?.id===job.id&&!playing)play();},{once:true});}
}
function pairMessage() {
  let note="four declared seeds. every output stays visible.";
  if(pinned&&output){
    const a=pinned.result.request,b=output.result.request;
    note=a.prompt!==b.prompt ? "the caption changed; this pair changes more than the push." :
      JSON.stringify(a.seeds)!==JSON.stringify(b.seeds) ? "the noise seeds changed between these actions." :
      "same caption and noise seeds; compare the actions without assuming a shared hidden world.";
  }
  $("pair-note").textContent=note;
}
async function openRecording(id) {
  busy=true;controls();
  try {
    const job=await api("/api/jobs/"+id);if(job.status!=="complete")throw Error(job.stage || "recording is incomplete");
    const request=job.result.request;
    if(!photo||photo.id!==request.photo)await selectPhoto(request.photo);
    anchor=request.anchor.slice();force=request.force.slice();$("caption").value=request.prompt;$("seed").value=request.seeds[0];
    draw();showResult(job,"recorded replay");$("recordings").value=id;$("photos").value=request.photo;
  } finally {busy=false;controls();}
}
$("recordings").onchange=()=>{if($("recordings").value)openRecording($("recordings").value).catch(e=>message(e.message,true));};
function seek(seconds) {
  stop();for(const video of document.querySelectorAll("video")){
    if(video.readyState>=1)video.currentTime=Math.min(seconds,Math.max(0,video.duration-.001));
  }
  $("time").textContent=seconds.toFixed(2)+"s";
}
$("timeline").oninput=()=>seek(Number($("timeline").value)/(output?.result.fps || state.fps));
async function play() {
  if(playing){stop();return;}
  const videos=[...document.querySelectorAll("video")];if(!videos.length)return;
  if(Number($("timeline").value)>=Number($("timeline").max))seek(0);
  try{await Promise.all(videos.map(v=>v.play()));playing=true;$("play").textContent="pause";}
  catch(error){stop();message("press play when all four videos finish loading.");}
}
$("play").onclick=play;
function animate() {
  if(playing){const videos=[...document.querySelectorAll("video")],primary=videos[0];
    if(primary){const t=primary.currentTime;$("timeline").value=Math.min(Number($("timeline").max),Math.round(t*(output?.result.fps || state.fps)));$("time").textContent=t.toFixed(2)+"s";
      for(const v of videos.slice(1))if(v.readyState>=2&&Math.abs(v.currentTime-t)>.12)v.currentTime=t;
      if(primary.ended)stop();}}
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
$("pin").onclick=()=>{if(!output)return;stop();pinned=output;cards($("pinned-results"),pinned);$("pinned").hidden=false;document.body.classList.add("has-pinned");
  $("pinned-action").textContent="push ["+pinned.result.request.force.map(v=>v.toFixed(2)).join(", ")+"] / seeds "+pinned.result.request.seeds.join(", ");
  pairMessage();};
$("clear-pin").onclick=()=>{stop();pinned=null;document.body.classList.remove("has-pinned");$("pinned").hidden=true;$("pinned-results").replaceChildren();pairMessage();};
$("generate").onclick=async()=>{
  const mine=++operation;let receipt;
  try{
    const seed=Number($("seed").value);if(!Number.isInteger(seed)||seed<0||seed>2147483644)throw Error("use an integer seed between 0 and 2147483644");
    busy=true;controls();message("sending the four-seed request...");
    receipt=await api("/api/generate",{photo:photo.id,prompt:$("caption").value.trim(),anchor:anchor.slice(),force:force.slice(),seeds:[seed,seed+1,seed+2,seed+3]});
    if(mine!==operation){await api("/api/jobs/"+receipt.id+"/cancel",{});return;}
    active=receipt.id;
    while(mine===operation){
      const job=await api("/api/jobs/"+receipt.id);message(job.stage || job.status);
      if(job.status==="complete"){
        showResult(job,receipt.source==="cached replay" ? "cached replay" : "live model output");
        break;
      }
      if(job.status==="error")throw Error(job.error || job.stage || "generation failed");
      if(job.status==="cancelled"){message("generation cancelled.");break;}
      await new Promise(resolve=>setTimeout(resolve,800));
    }
  }catch(error){if(mine===operation)message(error.message,true);}
  finally{if(mine===operation){busy=false;active=null;controls();}}
};
$("cancel").onclick=async()=>{++operation;const id=active;active=null;busy=false;controls();
  try{if(id)await api("/api/jobs/"+id+"/cancel",{});message("cancellation requested; the worker keeps any completed records.");}
  catch(error){message(error.message,true);}};
reduced.addEventListener("change",()=>{if(reduced.matches)stop();});
(async()=>{try{await refresh();if(state.recordings.length){await openRecording(state.recordings[0].id);}
  else if(state.photos.length){await selectPhoto(state.photos[0].id);$("photos").value=state.photos[0].id;}
  setInterval(()=>refresh().catch(()=>{$("worker").textContent="worker connection unavailable";}),5000);
}catch(error){message(error.message,true);}})();
