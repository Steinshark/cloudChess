const PIECES = {K:'♔',Q:'♕',R:'♖',B:'♗',N:'♘',P:'♙',k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟'};
const state = { game:null, selected:null, models:[], pendingPromotions:null };
const $ = id => document.getElementById(id);

async function api(path, options={}) {
  const response = await fetch(path, {headers:{'Content-Type':'application/json'}, ...options});
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { const body = await response.json(); detail = body.detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}
function showError(message='') { const el=$('error'); el.textContent=message; el.classList.toggle('show', !!message); }
function fmtBytes(n){ if(!n)return '—'; const u=['B','KB','MB','GB']; let i=0,v=n; while(v>=1024&&i<u.length-1){v/=1024;i++} return `${v.toFixed(i?1:0)} ${u[i]}`; }
function fmtNum(n,d=2){ return Number.isFinite(Number(n)) ? Number(n).toFixed(d) : '—'; }
function parseFen(fen){
  const placement=fen.split(' ')[0], board={}; let rank=8,file=0;
  for(const ch of placement){ if(ch==='/'){rank--;file=0;continue} if(/\d/.test(ch)){file+=Number(ch);continue} board['abcdefgh'[file]+rank]=ch;file++; }
  return board;
}
function orientedSquares(color){
  const files=color==='white'?'abcdefgh'.split(''):'hgfedcba'.split('');
  const ranks=color==='white'?[8,7,6,5,4,3,2,1]:[1,2,3,4,5,6,7,8];
  return ranks.flatMap(r=>files.map(f=>f+r));
}
function renderBoard(){
  const boardEl=$('board'); boardEl.innerHTML='';
  const game=state.game, orientation=game?.human_color || $('side').value;
  const pieces=game?parseFen(game.fen):parseFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const legal=game?.legal_moves || [], last=game?.last_move || '';
  const targets=state.selected ? legal.filter(m=>m.slice(0,2)===state.selected).map(m=>m.slice(2,4)) : [];
  const squares=orientedSquares(orientation);
  squares.forEach((sq,index)=>{
    const file='abcdefgh'.indexOf(sq[0]), rank=Number(sq[1]);
    const el=document.createElement('button'); el.className=`square ${(file+rank)%2?'light':'dark'}`; el.dataset.square=sq;
    if(state.selected===sq)el.classList.add('selected');
    if(last && (last.slice(0,2)===sq || last.slice(2,4)===sq))el.classList.add('last');
    if(targets.includes(sq))el.classList.add(pieces[sq]?'capture':'legal');
    if (pieces[sq]) {
      const p = document.createElement('span');
      const pieceCode = pieces[sq];
      const isWhitePiece = pieceCode === pieceCode.toUpperCase();
      p.className = `piece ${isWhitePiece ? 'white-piece' : 'black-piece'}`;
      p.textContent = PIECES[pieceCode];
      el.appendChild(p);
    }
    const col=index%8,row=Math.floor(index/8);
    if(col===0){const c=document.createElement('span');c.className='coord rank';c.textContent=sq[1];el.appendChild(c)}
    if(row===7){const c=document.createElement('span');c.className='coord file';c.textContent=sq[0];el.appendChild(c)}
    el.addEventListener('click',()=>clickSquare(sq,pieces)); boardEl.appendChild(el);
  });
}
function clickSquare(square,pieces){
  const game=state.game; if(!game || !game.human_to_move || game.game_over) return;
  const legal=game.legal_moves;
  if(state.selected){
    const candidates=legal.filter(m=>m.slice(0,2)===state.selected && m.slice(2,4)===square);
    if(candidates.length){
      if(candidates.length>1) return choosePromotion(candidates);
      state.selected=null; renderBoard(); return submitMove(candidates[0]);
    }
  }
  const selectable=legal.some(m=>m.slice(0,2)===square);
  state.selected=selectable?square:null; renderBoard();
}
function choosePromotion(candidates){
  state.pendingPromotions=candidates; const box=$('promotion-options'); box.innerHTML='';
  const symbols={q:'♛',r:'♜',b:'♝',n:'♞'};
  candidates.forEach(uci=>{const b=document.createElement('button');b.textContent=symbols[uci[4]];b.onclick=()=>{closePromotion();submitMove(uci)};box.appendChild(b)});
  $('promotion-modal').classList.add('show');
}
function closePromotion(){ $('promotion-modal').classList.remove('show'); state.pendingPromotions=null; state.selected=null; renderBoard(); }
async function submitMove(uci){
  setThinking(true); showError();
  try { state.game=await api(`/api/games/${state.game.id}/moves`,{method:'POST',body:JSON.stringify({uci})}); renderAll(); }
  catch(e){showError(e.message)} finally{setThinking(false)}
}
function setThinking(value){ $('thinking').classList.toggle('show',value); $('new-game').disabled=value; $('refresh-models').disabled=value; }
function statusText(game){
  if(!game)return 'Select a model';
  if(game.game_over){ const r=game.result==='1-0'?'White wins':game.result==='0-1'?'Black wins':'Draw'; return `${r} · ${game.termination}`; }
  if(game.in_check)return `${game.turn[0].toUpperCase()+game.turn.slice(1)} is in check`;
  return game.human_to_move?'Your move':'Model to move';
}
function renderMeta(game){
  const el=$('model-meta'); if(!game){el.innerHTML='';return}
  const m=game.model;
  el.innerHTML=`<div class="metric-mini"><span class="label">Blocks</span><strong>${m.residual_blocks??'—'}</strong></div><div class="metric-mini"><span class="label">Generation</span><strong>${m.generation??'Bootstrap'}</strong></div><div class="metric-mini"><span class="label">Size</span><strong>${fmtBytes(m.size_bytes)}</strong></div>`;
  const ai=$('ai-details');
  if(game.last_ai){ai.style.display='grid';ai.innerHTML=`<div><span class="label">Last search</span><strong>${game.last_ai.simulations} sims</strong></div><div><span class="label">Root value</span><strong>${fmtNum(game.last_ai.root_value,3)}</strong></div><div><span class="label">Time</span><strong>${fmtNum(game.last_ai.elapsed_seconds,2)}s</strong></div>`} else ai.style.display='none';
}
function renderMoves(game){
  const el=$('move-list'); if(!game || !game.moves.length){el.innerHTML='<div class="empty" style="grid-column:1/-1">No moves yet.</div>';return}
  let html=''; for(let i=0;i<game.moves.length;i+=2){const no=i/2+1,w=game.moves[i],b=game.moves[i+1];html+=`<div class="move-num">${no}.</div><div class="move-san ${i===game.moves.length-1?'latest':''}">${w?.san||''}</div><div class="move-san ${i+1===game.moves.length-1?'latest':''}">${b?.san||''}</div>`} el.innerHTML=html; el.scrollTop=el.scrollHeight;
}
function renderAll(){
  renderBoard(); const g=state.game; $('status').textContent=statusText(g);
  $('turn-badge').innerHTML=`<span class="dot"></span>${g?(g.game_over?'Finished':`${g.turn[0].toUpperCase()+g.turn.slice(1)} to move`):'Idle'}`;
  renderMeta(g); renderMoves(g);
}
async function loadModels(refresh=false){
  showError(); try{
    const data=await api(`/api/models?refresh=${refresh}`); state.models=data.models.filter(m=>!m.inspect_error); const select=$('model'); const previous=select.value; select.innerHTML='';
    state.models.forEach(m=>{const o=document.createElement('option');o.value=m.id;o.textContent=m.name;select.appendChild(o)});
    if(previous && state.models.some(m=>m.id===previous))select.value=previous;
    if(!state.models.length)showError('No readable .pt checkpoints were found. Update models.roots in web_config.yaml.');
    const standard=[32,128,400,800].filter(v=>v>=data.simulation_limits.minimum&&v<=data.simulation_limits.maximum); $('simulations').innerHTML=standard.map(v=>`<option value="${v}" ${v===data.simulation_limits.default?'selected':''}>${v}${v===32?' · Quick':v===128?' · Standard':v===400?' · Strong':' · Maximum'}</option>`).join('');
  }catch(e){showError(e.message)}
}
async function newGame(){
  const model_id=$('model').value; if(!model_id)return showError('Select a checkpoint first.'); setThinking(true); showError(); state.selected=null;
  try{ state.game=await api('/api/games',{method:'POST',body:JSON.stringify({model_id,human_color:$('side').value,simulations:Number($('simulations').value)})}); renderAll(); }
  catch(e){showError(e.message)} finally{setThinking(false)}
}
$('new-game').addEventListener('click',newGame); $('refresh-models').addEventListener('click',()=>loadModels(true)); $('side').addEventListener('change',()=>{if(!state.game)renderBoard()}); $('promotion-modal').addEventListener('click',e=>{if(e.target===$('promotion-modal'))closePromotion()});
loadModels().then(renderAll);
