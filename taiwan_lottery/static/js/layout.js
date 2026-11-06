/* Offline OCR is a draft. Never insert server strings as HTML. */
(() => {
  const el = id => document.getElementById(id);
  let corners = [], image = null, runId = null, busy = false, ocrResult = null;
  const coordinate = x => x ? 'R'+x.row+'C'+x.col : '—';
  const dimensions = () => ({rows: Number(el('rows').value), cols: Number(el('cols').value)});
  function build(cells) {
    const {rows, cols} = dimensions();
    if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows<1 || cols<1 || rows>100 || cols>100 || rows*cols>400) throw Error('最多 400 格');
    el('grid').replaceChildren();
    el('grid').style.gridTemplateColumns = 'repeat('+cols+', minmax(42px,1fr))';
    for (let i=0; i<rows*cols; i++) {
      const input = document.createElement('input');
      input.type='number'; input.min=1; input.max=rows*cols;
      input.className='form-control px-1 text-center'; input.value=cells?.[i] ?? '';
      input.setAttribute('aria-label', 'R'+(Math.floor(i/cols)+1)+'C'+(i%cols+1));
      input.addEventListener('input', () => {el('confirmed').checked=false;});
      el('grid').append(input);
    }
    el('confirmed').checked=false;
  }
  const board = () => {
    const dims=dimensions();
    if (el('grid').children.length!==dims.rows*dims.cols) throw Error('尺寸已更改，請重新建立表格');
    return {...dims,cells:Array.from(el('grid').children, i=>i.value==='' ? null : Number(i.value))};
  };
  async function api(path, body) {
    const options = body instanceof FormData ? {method:'POST',body} : body===undefined ? {} :
      {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)};
    const r = await fetch('/api/v1/layout'+path,options);
    const data = await r.json();
    if (!r.ok) throw Error(typeof data.detail==='string' ? data.detail : JSON.stringify(data.detail));
    return data;
  }
  function action(id, fn) {
    el(id).onclick = async () => {
      if(busy) return; busy=true; el(id).disabled=true; el('status').textContent='處理中…';
      try {await fn(); el('status').textContent='完成';}
      catch(e) {el('status').textContent=e.message;}
      finally {busy=false; el(id).disabled=false;}
    };
  }
  action('create',()=>{build();ocrResult=null;el('ocrReview').hidden=true;});
  el('photo').onchange=()=>{
    const file=el('photo').files[0]; if(!file) return;
    const url=URL.createObjectURL(file); image=new Image();
    image.onload=()=>{corners=[];draw();URL.revokeObjectURL(url);}; image.src=url;
  };
  function draw() {
    const canvas=el('photoCanvas'); canvas.hidden=false;
    canvas.width=image.width;canvas.height=image.height;
    const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);
    corners.forEach(([x,y],i)=>{ctx.fillStyle='red';ctx.beginPath();ctx.arc(x,y,Math.max(image.width/100,4),0,Math.PI*2);ctx.fill();ctx.font=(image.width/30)+'px sans-serif';ctx.fillText(i+1,x,y);});
    el('ocrStatus').textContent='已選 '+corners.length+' / 4 個表格外角。辨識後請校正所有可見數字。';
  }
  el('photoCanvas').onclick=e=>{
    if(!image || corners.length>=4) return;
    const box=e.target.getBoundingClientRect();
    corners.push([(e.clientX-box.left)*image.width/box.width,(e.clientY-box.top)*image.height/box.height]);draw();
  };
  el('resetCorners').onclick=()=>{corners=[];if(image) draw();};
  action('recognize',async()=>{
    const file=el('photo').files[0];if(!file) throw Error('請先選擇照片');
    if(corners.length!==0 && corners.length!==4) throw Error('請選滿四個角或清除角點');
    const dims=dimensions(), data=new FormData();data.append('file',file);
    data.append('rows',dims.rows);data.append('cols',dims.cols);
    data.append('retain',el('retainOCR').checked);
    if(corners.length===4) data.append('corners',JSON.stringify(corners));
    const result=await api('/recognize',data);
    if(result.status!=='needs_confirmation') throw Error(result.reason);
    build(result.cells);ocrResult=result;renderOCR();el('ocrStatus').textContent='辨識草稿已填入；未開/不確定格保持空白。請核對每格及全表校正。';
  });
  action('predict',async()=>{
    if(!el('confirmed').checked) throw Error('請先確認表格');
    const r=await api('/predict',{...board(),target:Number(el('target').value)});
    runId=r.run_id;el('result').hidden=false;
    el('ranking').replaceChildren();el('models').replaceChildren();
    el('trace').textContent=JSON.stringify(r,null,2);
    if(r.status==='already_revealed') {el('summary').textContent='目標已揭露：'+coordinate(r);return;}
    const c=r.consensus;
    el('summary').textContent='共識：'+c.strength+'；分歧 '+(c.disagreement_level*100).toFixed(0)+'%；隨機 Top-1 基準 '+(r.random_baseline.top1*100).toFixed(1)+'%；全表分配 '+coordinate(c.target_assignment)+(c.assignment_disagrees_with_top1?'（與單一排名不同）':'')+'。討論效果尚未證實。';
    [['Top 1',[c.top1]],['Top 3',c.top3],['Top 5',c.top5]].forEach(([label,items])=>{
      const group=document.createElement('div');group.className='w-100';
      const title=document.createElement('strong');title.textContent=label+' ';group.append(title);
      items.forEach((x,i)=>{const badge=document.createElement('span');badge.className='badge bg-secondary p-2 me-1 mb-1';badge.textContent=(i+1)+'. '+coordinate(x)+' 分數 '+x.score.toFixed(4);group.append(badge);});
      el('ranking').append(group);
    });
    r.round1.forEach(p=>{
      const review=r.round2.find(x=>x.model===p.model), tr=document.createElement('tr');
      [p.model,p.status==='ready' ? coordinate(p.top1)+' → '+coordinate(review.after_top1) : 'not_ready',
        p.status==='ready' ? c.effective_vote_weight[p.model].toFixed(3) : '—',
        review ? review.disagreement_reasons.join(', ') : p.reason].forEach(value=>{
          const td=document.createElement('td');td.textContent=value;tr.append(td);
        });el('models').append(tr);
    });
    const d=r.devils_advocate;
    el('advocate').textContent='支持 '+d.support_models+' / 反對 '+d.oppose_models+'；替代候選 '+coordinate(d.alternative)+'；風險 '+d.risk+'：'+d.reasons.join(', ');
  });
  action('saveHistory',async()=>{
    if(!el('confirmed').checked) throw Error('請先確認完整表格');
    await api('/history',{...board(),source:el('source').value});
  });
  action('resolve',async()=>{
    if(!runId) throw Error('請先預測');
    const r=await api('/runs/'+runId+'/resolve',{row:Number(el('actualRow').value),col:Number(el('actualCol').value)});
    el('trace').textContent+='\nActual outcome:\n'+JSON.stringify(r,null,2);
  });
  function showEvaluation(r) {
    el('evaluation').textContent=JSON.stringify({status:r.status,aggregate:r.aggregate,
      individual_models:r.individual_models,discussion_gain:r.discussion_gain,
      discussion_improved:r.discussion_improved,warning:r.warning},null,2);
  }
  action('evaluate',async()=>showEvaluation(await api('/evaluation',{...dimensions(),max_folds:8})));
  action('getEvaluation',async()=>showEvaluation(await api('/evaluation?'+new URLSearchParams(dimensions()))));

  function renderOCR() {
    if(!ocrResult) return;
    el('ocrReview').hidden=false;el('ocrCells').replaceChildren();
    const perf=ocrResult.performance;
    el('ocrPerformance').textContent='整表 '+(perf.whole_board_ms/1000).toFixed(2)+' 秒；平均每格 '+perf.per_cell_ms.toFixed(0)+' ms；記憶體 '+(perf.rss_bytes ? (perf.rss_bytes/1048576).toFixed(1)+' MiB' : '未提供')+'。分數未校準，不是正確率。';
    ocrResult.proposals.forEach((p,index)=>{
      if(el('disputedOnly').checked && !p.disputed) return;
      const tr=document.createElement('tr');if(p.disputed)tr.className='table-warning';
      const td=document.createElement('td');td.textContent=coordinate(p);tr.append(td);
      const state=document.createElement('td'),select=document.createElement('select');select.className='form-select form-select-sm';select.dataset.index=index;
      ['OPENED','UNOPENED','UNCERTAIN'].forEach(value=>{const opt=document.createElement('option');opt.value=value;opt.textContent=value;select.append(opt);});
      select.value=p.manualState || p.state;select.onchange=()=>{p.manualState=select.value;el('confirmed').checked=false;};
      state.append(select);tr.append(state);
      const values=[String(p.value ?? 'unknown')+' / '+p.confidence.toFixed(3),
        p.raw_engines.map(x=>x.engine+': '+(x.candidates.map(c=>c.value+' ('+c.score.toFixed(2)+')').join(' / ')||x.status)).join('; '),
        p.corrected ? p.correction_reason : '—'];
      values.forEach(value=>{const cell=document.createElement('td');cell.textContent=value;tr.append(cell);});
      el('ocrCells').append(tr);
    });
    const engines=ocrResult.engines.map(x=>x.engine+': '+x.status+'（載入 '+x.load_seconds.toFixed(2)+' 秒）').join('; ');
    el('ocrStatus').textContent=engines;
  }
  el('disputedOnly').onchange=renderOCR;
  action('saveCorrections',async()=>{
    if(!ocrResult?.capture_id) throw Error('此照片未保留格子圖片，請勾選保留並重新辨識');
    if(!el('confirmed').checked) throw Error('請先確認數字與開格狀態');
    const b=board();
    if(b.rows!==ocrResult.rows || b.cols!==ocrResult.cols) throw Error('尺寸已變更，請重新辨識');
    const source=el('ocrSourceBoard').value.trim();if(!source) throw Error('請填寫實體表格來源 ID');
    const cells=ocrResult.proposals.map((p,index)=>({index,label:b.cells[index],
      state:b.cells[index]!==null ? 'OPENED' : (p.manualState || p.state),
      scenarios:Array.from(el('ocrScenarios').selectedOptions,o=>o.value)}));
    const r=await api('/ocr/corrections',{capture_id:ocrResult.capture_id,source_board:source,cells,auto_train:true});
    el('ocrBenchmarkResult').textContent=JSON.stringify(r,null,2);
  });
  action('ocrBenchmark',async()=>{
    const r=await api('/ocr/benchmark',{});
    el('ocrBenchmarkResult').textContent=JSON.stringify(r,null,2);
  });

  build();
})();
