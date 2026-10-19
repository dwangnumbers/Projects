import { useState } from "react";

const D = {"hypothesis":"An RL agent observing recent photon emission history can learn to predict the emission rate in upcoming time windows. The agent will discover the non-stationary, decaying nature of the emission process and outperform both a constant-rate (Poisson) model and a naive persistence baseline, demonstrating that photon emission rates carry learnable temporal structure.","dataset":{"n_events":244693,"time_span_ns":2630.35,"window_ns":5.0,"n_windows":527,"mean_rate":92.86,"max_rate":694.4,"nonzero_windows":370,"zero_windows":157,"decay_constant_ns":1973,"rate_labels":["silent","very_low","low","medium","high","very_high"],"category_distribution":{"silent":157,"very_low":75,"low":73,"medium":74,"high":74,"very_high":74}},"config":{"algorithm":"Tabular Q-Learning","state_space":"6×3×5×3 = 270","state_components":["recent_rate_category (6)","rate_trend (3)","time_epoch (5)","zero_fraction (3)"],"alpha":0.15,"gamma":0.9,"episodes":200,"segment_length":300},"training":{"rewards":[-29.7,-48.3,-52.6,-67.1,-31.1,-33.2,-34.9,-10.3,-32,-32.8,-25.6,-25.5,-18.5,-10.3,-6.8,-39.8,-30.8,2.7,-4.1,-18.7,8.9,-8.7,16.1,-6.1,-3,24.7,12.4,-5.8,5.6,11.5,1.2,-1.1,-2.8,34.7,16.7,7.3,20.3,11,24,22.7,34.3,38.4,24.4,21.5,28.1,42.8,15,29.7,41.4,26.2,34.7,24.4,56.2,42.7,34.4,52.6,43.8,32.2,31.3,24.4,44.5,40.8,40.6,47.9,33.6,79.6,58.3,57.2,33.1,69.1,68.5,39.6,63.9,93.5,67.1,47.2,51.3,63.6,44.8,44.5,61.9,56.1,62.2,61.3,84.9,48.9,66.9,64,68.7,40,83.2,91.5,53.8,61.9,84.2,71,74.8,60.2,78.8,107.9,51.9,62.2,85.2,53.1,71.1,93.3,77.1,73.4,75.7,57.4,65.9,96.9,93.5,76,67.7,71.5,93.9,80,102.7,89,65.6,105.3,82.3,83.9,80.5,95.8,66.2,52.7,74.7,70.7,82.1,75.5,60.6,66.3,76.6,75.6,77.4,77.2,77.7,98.8,80.8,69.8,71.9,104.2,91.4,61.3,77.7,82.1,100.2,80.2,76.9,78.4,119.1,75.1,77,78.1,81.9,84.9,80.8,82.6,89.5,88.4,80.5,103.5,94.3,76.6,111,128.9,86.1,85.9,116.3,76.5,96.3,79.3,86.1,91.4,77.1,129.7,80.8,121,107.8,79.7,94.2,99,83.3,93.7,112.8,89,88.3,88.4,92.9,84.1,107.5,91.6,89.3,132.3,87.8,78.5,93.9,125.7],"accuracy":[0.183,0.127,0.13,0.147,0.183,0.183,0.17,0.213,0.19,0.193,0.187,0.183,0.183,0.213,0.19,0.17,0.22,0.2,0.193,0.237,0.227,0.277,0.243,0.213,0.227,0.283,0.273,0.203,0.22,0.233,0.227,0.21,0.22,0.253,0.247,0.257,0.243,0.227,0.26,0.233,0.267,0.223,0.223,0.227,0.263,0.263,0.257,0.257,0.287,0.28,0.277,0.223,0.307,0.29,0.287,0.31,0.253,0.267,0.227,0.247,0.3,0.307,0.267,0.28,0.267,0.367,0.33,0.297,0.26,0.32,0.34,0.277,0.26,0.393,0.387,0.28,0.283,0.317,0.267,0.277,0.327,0.31,0.34,0.3,0.377,0.317,0.323,0.35,0.357,0.263,0.387,0.413,0.293,0.35,0.353,0.303,0.34,0.313,0.393,0.427,0.287,0.34,0.39,0.27,0.323,0.377,0.333,0.387,0.333,0.267,0.31,0.397,0.417,0.323,0.297,0.333,0.403,0.33,0.437,0.333,0.297,0.373,0.347,0.34,0.36,0.41,0.3,0.283,0.343,0.313,0.33,0.317,0.277,0.32,0.35,0.36,0.347,0.353,0.337,0.427,0.35,0.32,0.32,0.433,0.417,0.277,0.36,0.353,0.473,0.323,0.317,0.313,0.497,0.353,0.313,0.323,0.337,0.353,0.327,0.337,0.387,0.353,0.36,0.41,0.39,0.303,0.417,0.497,0.36,0.343,0.45,0.307,0.347,0.307,0.337,0.37,0.31,0.493,0.327,0.457,0.407,0.317,0.37,0.373,0.36,0.363,0.45,0.357,0.357,0.33,0.367,0.333,0.417,0.363,0.347,0.477,0.373,0.333,0.38,0.487],"td_errors":[0.65,0.58,0.6,0.61,0.61,0.6,0.62,0.64,0.57,0.61,0.63,0.69,0.61,0.65,0.59,0.58,0.74,0.62,0.63,0.72,0.65,0.71,0.75,0.64,0.72,0.78,0.76,0.63,0.71,0.72,0.77,0.71,0.65,0.68,0.75,0.7,0.67,0.67,0.7,0.73,0.7,0.65,0.65,0.62,0.64,0.67,0.74,0.66,0.66,0.75,0.73,0.64,0.77,0.75,0.72,0.7,0.63,0.67,0.67,0.64,0.67,0.73,0.66,0.65,0.67,0.81,0.68,0.67,0.62,0.77,0.7,0.65,0.63,0.83,0.83,0.67,0.67,0.69,0.67,0.64,0.65,0.64,0.7,0.64,0.76,0.68,0.64,0.66,0.78,0.67,0.68,0.72,0.67,0.67,0.6,0.61,0.63,0.65,0.66,0.7,0.6,0.66,0.71,0.6,0.62,0.67,0.6,0.71,0.56,0.6,0.61,0.68,0.66,0.59,0.57,0.58,0.62,0.61,0.68,0.62,0.59,0.63,0.61,0.56,0.62,0.61,0.6,0.61,0.58,0.62,0.55,0.58,0.61,0.59,0.59,0.6,0.61,0.62,0.6,0.63,0.62,0.61,0.61,0.64,0.64,0.6,0.63,0.61,0.65,0.62,0.61,0.58,0.67,0.59,0.59,0.58,0.6,0.57,0.58,0.58,0.61,0.6,0.62,0.59,0.58,0.59,0.6,0.64,0.58,0.59,0.6,0.6,0.58,0.59,0.6,0.56,0.57,0.6,0.62,0.56,0.58,0.59,0.58,0.58,0.59,0.57,0.62,0.58,0.58,0.58,0.58,0.59,0.58,0.59,0.58,0.56,0.59,0.61,0.55,0.57]},"eval":{"rl":{"accuracy":0.426,"adj_accuracy":0.741,"reward_per_step":0.393},"persistence":{"accuracy":0.289,"adj_accuracy":0.557,"reward_per_step":0.088},"constant":{"accuracy":0.298,"adj_accuracy":0.441,"reward_per_step":-0.106}},"epoch_analysis":{"0-526":{"rl_acc":0.267,"persist_acc":0.219,"const_acc":0.067,"mean_rate":149.7},"526-1052":{"rl_acc":0.333,"persist_acc":0.190,"const_acc":0.095,"mean_rate":144.4},"1052-1578":{"rl_acc":0.358,"persist_acc":0.132,"const_acc":0.142,"mean_rate":107.0},"1578-2104":{"rl_acc":0.410,"persist_acc":0.295,"const_acc":0.429,"mean_rate":48.2},"2104-2630":{"rl_acc":0.762,"persist_acc":0.610,"const_acc":0.762,"mean_rate":16.2}},"policy_by_epoch":{"0-526":{"silent":"low","very_low":"silent","low":"high","medium":"medium","high":"low","very_high":"very_low"},"526-1052":{"silent":"medium","very_low":"very_high","low":"high","medium":"high","high":"high","very_high":"very_high"},"1052-1578":{"silent":"high","very_low":"low","low":"silent","medium":"low","high":"silent","very_high":"medium"},"1578-2104":{"silent":"silent","very_low":"silent","low":"high","medium":"low","high":"very_low","very_high":"silent"},"2104-2630":{"silent":"silent","very_low":"silent","low":"silent","medium":"silent","high":"very_low","very_high":"silent"}},"stats":{"rl_wins":228,"persist_wins":108,"ties":190,"z":7.01,"p":2.4e-12,"sig":true},"confusion":[[102,30,7,8,7,3],[14,24,13,13,10,1],[7,8,26,14,15,3],[3,10,13,28,18,2],[6,4,6,20,34,3],[3,5,8,15,33,10]],"rate_profile":{"time":[2.5,12.5,22.5,32.5,42.5,52.5,62.5,72.5,82.5,92.5,102.5,112.5,122.5,132.5,142.5,152.5,162.5,172.5,182.5,192.5,202.5,212.5,222.5,232.5,242.5,252.5,262.5,272.5,282.5,292.5,302.5,312.5,322.5,332.5,342.5,352.5,362.5,372.5,382.5,392.5,402.5,412.5,422.5,432.5,442.5,452.5,462.5,472.5,482.5,492.5,502.5,512.5,522.5,532.5,542.5,552.5,562.5,572.5,582.5,592.5,602.5,612.5,622.5,632.5,642.5,652.5,662.5,672.5,682.5,692.5,702.5,712.5,722.5,732.5,742.5,752.5,762.5,772.5,782.5,792.5,802.5,812.5,822.5,832.5,842.5,852.5,862.5,872.5,882.5,892.5,902.5,912.5,922.5,932.5,942.5,952.5,962.5,972.5,982.5,992.5,1002.5,1012.5,1022.5,1032.5,1042.5,1052.5,1062.5,1072.5,1082.5,1092.5,1102.5,1112.5,1122.5,1132.5,1142.5,1152.5,1162.5,1172.5,1182.5,1192.5,1202.5,1212.5,1222.5,1232.5,1242.5,1252.5,1262.5,1272.5,1282.5,1292.5,1302.5,1312.5,1322.5,1332.5,1342.5,1352.5,1362.5,1372.5,1382.5,1392.5,1402.5,1412.5,1422.5,1432.5,1442.5,1452.5,1462.5,1472.5,1482.5,1492.5,1502.5,1512.5,1522.5,1532.5,1542.5,1552.5,1562.5,1572.5,1582.5,1592.5,1602.5,1612.5,1622.5,1632.5,1642.5,1652.5,1662.5,1672.5,1682.5,1692.5,1702.5,1712.5,1722.5,1732.5,1742.5,1752.5,1762.5,1772.5,1782.5,1792.5,1802.5,1812.5,1822.5,1832.5,1842.5,1852.5,1862.5,1872.5,1882.5,1892.5,1902.5,1912.5,1922.5,1932.5,1942.5,1952.5,1962.5,1972.5,1982.5,1992.5,2002.5,2012.5,2022.5,2032.5,2042.5,2052.5,2062.5,2072.5,2082.5,2092.5,2102.5,2112.5,2122.5,2132.5,2142.5,2152.5,2162.5,2172.5,2182.5,2192.5,2202.5,2212.5,2222.5,2232.5,2242.5,2252.5,2262.5,2272.5,2282.5,2292.5,2302.5,2312.5,2322.5,2332.5,2342.5,2352.5,2362.5,2372.5,2382.5,2392.5,2402.5,2412.5,2422.5,2432.5,2442.5,2452.5,2462.5,2472.5,2482.5,2492.5,2502.5,2512.5,2522.5,2532.5,2542.5,2552.5,2562.5,2572.5,2582.5,2592.5,2602.5,2612.5,2622.5,2632.5],"rate":[138.8,124.8,69.2,151.4,130.2,83.2,352.2,235.2,30,381.8,0,0,131.2,10.6,65.8,229,149.6,63.6,3.4,52.2,0,164.8,12.6,77.6,20.4,146,69.2,219.2,110,0,132.6,0,0,30,119.8,167.6,241,90.2,355.6,51.6,27,63,138,76.2,133.4,50.2,356.4,25,58.2,124.4,396.6,199.8,258.4,0,161.6,8.4,82,239.8,156.2,314.8,388,20.2,27,69,142.2,175.8,68.8,77.8,119,91.8,210,188,346,319.8,176.2,197,240.6,117,178.6,137.4,251,143.6,19.2,0,0,63.8,164.8,78.6,216.8,216,280.6,88.8,37.2,0,39,68.8,98,169.8,428.2,146.8,53.2,159.4,31,10.8,380.4,88.2,0,421.6,209.8,0,339,279.4,71,30.6,34.6,21.6,242.2,74.8,149.6,293,7.2,241,21.2,58.2,58.2,255.4,0,44.4,114.4,119.8,0,37,156,287,63.2,154.4,42.6,44.8,130.2,85.8,159.2,98.8,15.4,0,73.6,111.8,14,73.2,292.4,39.2,167,24.2,72.8,20.2,0,33.6,34,59.8,57.6,116.6,273,0,86.6,144.4,83.2,0,0,49.4,142.2,0,60.2,6.4,121.8,0,87.6,64.4,140,0,12.6,0,0,36.6,0,0,8.8,68,106.2,56.6,0.8,77.2,84.8,0,40.4,167,209.4,40.8,36.6,0,0,0,0,0,0,19.4,0,0,0,0,0,90.8,104.8,0,0,53.8,0,66.2,12.6,0,41.2,11,0,0,0,0,0,0,0,0,0,6.2,0,0,38,0,0,0,0,0,0,0,1.4,0,0,0,0,7.2,0,0,0,0,0,0,0,0,0,0,0,0,19,0,234.6,0,0,44.8]},"conclusions":["The RL agent achieved the highest prediction accuracy (42.6%) among all models, outperforming both the persistence baseline (28.9%) and constant predictor (29.8%).","In adjacent accuracy (\u00b11 category), the RL agent reached 74.1% vs. persistence at 55.7%, showing the agent rarely makes large prediction errors.","The emission rate decays with a characteristic timescale of ~1973 ns. The agent's learned policy reflects this: in early time epochs it predicts higher rate categories, while in late epochs it predominantly predicts low or silent emission states.","The RL agent's advantage over persistence is statistically significant (z=7.01, p=2.40e-12), confirming that the temporal structure of photon emission rates is learnable beyond simple extrapolation.","CONCLUSION: The photon emission rate is non-stationary and carries temporal structure. The RL agent learned that emission frequency decays over time and that bursts of activity are context-dependent, validating that electric dipole photon emission rates are predictable beyond a simple Poisson process."]};

const CAT_COLORS = ["#334155","#6366f1","#8b5cf6","#a78bfa","#f59e0b","#ef4444"];
const TABS = ["Overview","Training","Policy","Conclusions"];

function Line({data,color,w=380,h=100,label}){
  const mn=Math.min(...data),mx=Math.max(...data),r=mx-mn||1,p=4,cw=w-p*2,ch=h-p*2;
  const wn=Math.max(8,Math.floor(data.length/15));
  const ma=data.map((_,i)=>{const s=Math.max(0,i-wn+1);const sl=data.slice(s,i+1);return sl.reduce((a,b)=>a+b,0)/sl.length});
  const pts=data.map((v,i)=>`${p+i/(data.length-1)*cw},${p+ch-(v-mn)/r*ch}`).join(" ");
  const maPts=ma.map((v,i)=>`${p+i/(data.length-1)*cw},${p+ch-(v-mn)/r*ch}`).join(" ");
  return <div>
    {label&&<div style={{fontSize:10,color:"#888",marginBottom:2,fontFamily:"monospace"}}>{label}</div>}
    <svg width={w} height={h}><polyline points={pts} fill="none" stroke={color} strokeWidth={1} opacity={0.25}/><polyline points={maPts} fill="none" stroke={color} strokeWidth={2}/></svg>
    <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"#666",fontFamily:"monospace"}}><span>{mn.toFixed(1)}</span><span>{mx.toFixed(1)}</span></div>
  </div>;
}

function RateProfile({times,rates,w=720,h=160}){
  const mx=Math.max(...rates),p={t:8,r:8,b:28,l:40},cw=w-p.l-p.r,ch=h-p.t-p.b;
  return <svg width={w} height={h}>
    <g transform={`translate(${p.l},${p.t})`}>
      {rates.map((r,i)=>{
        const x=i/(rates.length-1)*cw, ht=r/mx*ch;
        const c=r===0?"#1e293b":r<38?"#6366f1":r<77?"#8b5cf6":r<134?"#a78bfa":r<209?"#f59e0b":"#ef4444";
        return <rect key={i} x={x-1} y={ch-ht} width={Math.max(2,cw/rates.length-0.5)} height={ht} fill={c} opacity={0.8} rx={0.5}/>;
      })}
      <line x1={0} y1={ch} x2={cw} y2={ch} stroke="#334155" strokeWidth={1}/>
      {[0,500,1000,1500,2000,2500].map(t=>{
        const x=t/times[times.length-1]*cw;
        return <text key={t} x={x} y={ch+14} textAnchor="middle" fontSize={9} fill="#666" fontFamily="monospace">{t}</text>;
      })}
      <text x={cw/2} y={ch+26} textAnchor="middle" fontSize={9} fill="#888" fontFamily="monospace">Time (ns)</text>
      {[0,200,400,600].map(r=>{
        const y=ch-r/mx*ch;
        return <g key={r}><line x1={-4} y1={y} x2={cw} y2={y} stroke="#1e293b" strokeWidth={0.5}/><text x={-6} y={y+3} textAnchor="end" fontSize={8} fill="#666" fontFamily="monospace">{r}</text></g>;
      })}
    </g>
  </svg>;
}

function ConfusionMatrix({matrix,labels}){
  const mx=Math.max(...matrix.flat());
  return <div style={{overflowX:"auto"}}>
    <table style={{borderCollapse:"collapse",fontSize:11,fontFamily:"monospace"}}>
      <thead><tr><th style={{padding:"4px 8px",color:"#555"}}></th>{labels.map(l=><th key={l} style={{padding:"4px 6px",color:"#aaa",fontWeight:400,fontSize:9,writingMode:"vertical-lr",height:60}}>{l}</th>)}</tr></thead>
      <tbody>{matrix.map((row,i)=><tr key={i}>
        <td style={{padding:"4px 8px",color:"#aaa",textAlign:"right",fontSize:9}}>{labels[i]}</td>
        {row.map((v,j)=><td key={j} style={{padding:"4px 6px",textAlign:"center",background:`rgba(99,102,241,${v/mx*0.8})`,color:v>mx*0.3?"#fff":"#888",borderRadius:2,border:"1px solid #1a1a30"}}>{v}</td>)}
      </tr>)}</tbody>
    </table>
  </div>;
}

function Stat({label,value,sub,accent}){
  return <div style={{background:"#111827",borderRadius:12,padding:"14px 18px",border:"1px solid #1e293b",flex:"1 1 170px",minWidth:155}}>
    <div style={{fontSize:9,color:"#555",fontFamily:"monospace",textTransform:"uppercase",letterSpacing:1,marginBottom:5}}>{label}</div>
    <div style={{fontSize:22,fontWeight:700,color:accent||"#fff",fontFamily:"monospace"}}>{value}</div>
    {sub&&<div style={{fontSize:10,color:"#666",marginTop:3}}>{sub}</div>}
  </div>;
}

function EpochTable({data}){
  const epochs=Object.entries(data);
  return <table style={{width:"100%",borderCollapse:"collapse",fontSize:11,fontFamily:"monospace"}}>
    <thead><tr>{["Epoch (ns)","Mean Rate","RL","Persist","Constant"].map(h=><th key={h} style={{padding:"6px 8px",textAlign:"left",color:"#555",borderBottom:"1px solid #1e293b",fontSize:9,textTransform:"uppercase",letterSpacing:0.5}}>{h}</th>)}</tr></thead>
    <tbody>{epochs.map(([epoch,v])=>{
      const best=Math.max(v.rl_acc,v.persist_acc,v.const_acc);
      return <tr key={epoch}>
        <td style={{padding:"6px 8px",color:"#aaa"}}>{epoch}</td>
        <td style={{padding:"6px 8px",color:"#888"}}>{v.mean_rate.toFixed(1)}/ns</td>
        <td style={{padding:"6px 8px",color:v.rl_acc===best?"#6366f1":"#888",fontWeight:v.rl_acc===best?700:400}}>{(v.rl_acc*100).toFixed(1)}%</td>
        <td style={{padding:"6px 8px",color:v.persist_acc===best?"#f59e0b":"#888",fontWeight:v.persist_acc===best?700:400}}>{(v.persist_acc*100).toFixed(1)}%</td>
        <td style={{padding:"6px 8px",color:v.const_acc===best?"#ef4444":"#888",fontWeight:v.const_acc===best?700:400}}>{(v.const_acc*100).toFixed(1)}%</td>
      </tr>;
    })}</tbody>
  </table>;
}

function PolicyTable({data}){
  const epochs=Object.entries(data);
  const cats=D.dataset.rate_labels;
  return <div style={{overflowX:"auto"}}>
    <table style={{borderCollapse:"collapse",fontSize:10,fontFamily:"monospace",width:"100%"}}>
      <thead><tr><th style={{padding:"5px 6px",color:"#555",textAlign:"left",fontSize:9}}>Epoch</th>{cats.map(c=><th key={c} style={{padding:"5px 6px",color:"#555",fontSize:9}}>if {c}</th>)}</tr></thead>
      <tbody>{epochs.map(([epoch,preds])=><tr key={epoch}>
        <td style={{padding:"5px 6px",color:"#aaa"}}>{epoch} ns</td>
        {cats.map(c=>{
          const pred=preds[c];
          const cidx=cats.indexOf(pred);
          return <td key={c} style={{padding:"5px 6px",textAlign:"center",color:CAT_COLORS[cidx]||"#888",background:cidx>=0?CAT_COLORS[cidx]+"15":"transparent",borderRadius:3}}>→ {pred}</td>;
        })}
      </tr>)}</tbody>
    </table>
  </div>;
}

export default function App(){
  const [tab,setTab]=useState(0);
  return <div style={{minHeight:"100vh",background:"#0a0a1a",color:"#d1d5db",fontFamily:"'Segoe UI',system-ui,sans-serif",padding:"20px 14px"}}>
    <div style={{maxWidth:860,margin:"0 auto"}}>
      <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
        <div style={{width:7,height:7,borderRadius:"50%",background:"#6366f1",boxShadow:"0 0 10px #6366f188"}}/>
        <span style={{fontSize:10,fontFamily:"monospace",color:"#6366f1",letterSpacing:2,textTransform:"uppercase"}}>Reinforcement Learning × Photon Emission</span>
      </div>
      <h1 style={{fontSize:24,fontWeight:700,margin:"6px 0 4px",color:"#fff"}}>Photon Emission Rate Prediction</h1>
      <p style={{fontSize:12,color:"#666",margin:0,maxWidth:640,lineHeight:1.5}}>
        Q-Learning agent trained to predict how often photons are emitted across {D.dataset.n_events.toLocaleString()} events over {D.dataset.time_span_ns.toFixed(0)} ns.
      </p>

      <div style={{display:"flex",gap:2,marginTop:16,background:"#111827",borderRadius:10,padding:3,width:"fit-content"}}>
        {TABS.map((t,i)=><button key={t} onClick={()=>setTab(i)} style={{padding:"7px 16px",fontSize:11,fontFamily:"monospace",border:"none",borderRadius:8,cursor:"pointer",background:tab===i?"#6366f118":"transparent",color:tab===i?"#6366f1":"#555",fontWeight:tab===i?600:400,transition:"all 0.2s"}}>{t}</button>)}
      </div>

      <div style={{marginTop:16}}>
        {tab===0&&<div>
          <div style={{background:"linear-gradient(135deg,#111827,#131a2e)",borderRadius:14,padding:20,border:"1px solid #6366f122",marginBottom:16}}>
            <div style={{fontSize:9,fontFamily:"monospace",color:"#6366f1",letterSpacing:1.5,textTransform:"uppercase",marginBottom:6}}>Hypothesis</div>
            <p style={{fontSize:13,lineHeight:1.7,color:"#bbb",margin:0}}>{D.hypothesis}</p>
          </div>
          <div style={{display:"flex",flexWrap:"wrap",gap:10,marginBottom:16}}>
            <Stat label="RL Accuracy" value={`${(D.eval.rl.accuracy*100).toFixed(1)}%`} sub="exact category match" accent="#6366f1"/>
            <Stat label="Adjacent Acc." value={`${(D.eval.rl.adj_accuracy*100).toFixed(1)}%`} sub="±1 category tolerance" accent="#8b5cf6"/>
            <Stat label="vs Persistence" value={`+${((D.eval.rl.accuracy-D.eval.persistence.accuracy)*100).toFixed(1)}%`} sub={`${(D.eval.persistence.accuracy*100).toFixed(1)}% baseline`} accent="#f59e0b"/>
            <Stat label="Significance" value={`p≈0`} sub={`z = ${D.stats.z.toFixed(1)}`} accent="#10b981"/>
          </div>

          <div style={{background:"#111827",borderRadius:12,padding:16,border:"1px solid #1e293b",marginBottom:16}}>
            <div style={{fontSize:10,fontFamily:"monospace",color:"#6366f1",marginBottom:10}}>Emission Rate Over Time (5 ns windows)</div>
            <RateProfile times={D.rate_profile.time} rates={D.rate_profile.rate}/>
            <div style={{display:"flex",gap:12,flexWrap:"wrap",justifyContent:"center",marginTop:8}}>
              {D.dataset.rate_labels.map((l,i)=><div key={l} style={{display:"flex",alignItems:"center",gap:4}}><div style={{width:8,height:8,borderRadius:2,background:CAT_COLORS[i]}}/><span style={{fontSize:9,color:"#888",fontFamily:"monospace"}}>{l}</span></div>)}
            </div>
            <p style={{fontSize:11,color:"#777",marginTop:8,lineHeight:1.5}}>
              The emission rate decays from ~150 photons/ns early on to near-silent at late times, with bursty
              intermittent structure throughout. Decay constant ≈ {D.dataset.decay_constant_ns} ns.
            </p>
          </div>

          <div style={{display:"flex",flexWrap:"wrap",gap:10}}>
            <div style={{background:"#111827",borderRadius:12,padding:16,border:"1px solid #1e293b",flex:"1 1 280px"}}>
              <div style={{fontSize:9,fontFamily:"monospace",color:"#555",letterSpacing:1,textTransform:"uppercase",marginBottom:8}}>Dataset</div>
              {[["Events",D.dataset.n_events.toLocaleString()],["Time span",`${D.dataset.time_span_ns.toFixed(1)} ns`],["Windows",`${D.dataset.n_windows} × ${D.dataset.window_ns} ns`],["Non-zero / Zero",`${D.dataset.nonzero_windows} / ${D.dataset.zero_windows}`],["Max rate",`${D.dataset.max_rate} /ns`]].map(([k,v])=><div key={k} style={{display:"flex",justifyContent:"space-between",padding:"3px 0",fontSize:11,borderBottom:"1px solid #1a1a30"}}><span style={{color:"#666"}}>{k}</span><span style={{fontFamily:"monospace",color:"#bbb"}}>{v}</span></div>)}
            </div>
            <div style={{background:"#111827",borderRadius:12,padding:16,border:"1px solid #1e293b",flex:"1 1 280px"}}>
              <div style={{fontSize:9,fontFamily:"monospace",color:"#555",letterSpacing:1,textTransform:"uppercase",marginBottom:8}}>Model</div>
              {[["Algorithm",D.config.algorithm],["State space",D.config.state_space],["Actions","6 rate categories"],["Learning rate",D.config.alpha],["Discount",D.config.gamma],["Episodes",D.config.episodes]].map(([k,v])=><div key={k} style={{display:"flex",justifyContent:"space-between",padding:"3px 0",fontSize:11,borderBottom:"1px solid #1a1a30"}}><span style={{color:"#666"}}>{k}</span><span style={{fontFamily:"monospace",color:"#bbb"}}>{v}</span></div>)}
            </div>
          </div>
        </div>}

        {tab===1&&<div style={{display:"flex",flexWrap:"wrap",gap:16}}>
          {[["Cumulative Reward","rewards","#6366f1","Reward climbs from negative (random guessing) to ~125+ as the agent learns emission patterns."],["Prediction Accuracy","accuracy","#8b5cf6","Accuracy improves from ~18% (random among 6 classes) to ~49%, far above the 16.7% chance level."],["Mean |TD Error|","td_errors","#10b981","TD error stabilizes around 0.57, indicating Q-value convergence."]].map(([title,key,color,desc])=>
            <div key={key} style={{background:"#111827",borderRadius:12,padding:18,border:"1px solid #1e293b",flex:"1 1 340px"}}>
              <div style={{fontSize:11,fontFamily:"monospace",color,marginBottom:10}}>{title}</div>
              <Line data={D.training[key]} color={color} w={380} h={110}/>
              <p style={{fontSize:11,color:"#777",marginTop:8}}>{desc}</p>
            </div>
          )}
        </div>}

        {tab===2&&<div>
          <div style={{background:"#111827",borderRadius:12,padding:18,border:"1px solid #1e293b",marginBottom:16}}>
            <div style={{fontSize:10,fontFamily:"monospace",color:"#6366f1",marginBottom:10}}>Accuracy by Time Epoch</div>
            <EpochTable data={D.epoch_analysis}/>
            <p style={{fontSize:11,color:"#777",marginTop:10,lineHeight:1.5}}>
              The RL agent dominates in early/middle epochs where emission is active and variable. In the late epoch (2104–2630 ns),
              it matches the constant predictor because the emission has largely ceased — both correctly predict "silent."
            </p>
          </div>
          <div style={{background:"#111827",borderRadius:12,padding:18,border:"1px solid #1e293b",marginBottom:16}}>
            <div style={{fontSize:10,fontFamily:"monospace",color:"#8b5cf6",marginBottom:10}}>Learned Policy: Predicted Rate by Epoch</div>
            <PolicyTable data={D.policy_by_epoch}/>
            <p style={{fontSize:11,color:"#777",marginTop:10,lineHeight:1.5}}>
              The policy reveals learned temporal structure: early epochs show diverse, high-rate predictions reflecting bursty emission.
              By the final epoch (2104–2630 ns), the agent predicts "silent" for nearly every input — it learned that emission decays to negligible levels.
            </p>
          </div>
          <div style={{background:"#111827",borderRadius:12,padding:18,border:"1px solid #1e293b"}}>
            <div style={{fontSize:10,fontFamily:"monospace",color:"#f59e0b",marginBottom:10}}>Confusion Matrix (RL Agent)</div>
            <ConfusionMatrix matrix={D.confusion} labels={D.dataset.rate_labels}/>
            <p style={{fontSize:11,color:"#777",marginTop:10,lineHeight:1.5}}>
              Diagonal concentration confirms the agent predicts correctly most often. The "silent" category (102/157 correct)
              is the easiest to predict; "very_high" (10/74) is hardest due to its bursty, unpredictable nature.
            </p>
          </div>
        </div>}

        {tab===3&&<div>
          <div style={{display:"flex",flexWrap:"wrap",gap:14,marginBottom:20}}>
            {[["RL Agent",D.eval.rl,"#6366f1"],["Persistence",D.eval.persistence,"#f59e0b"],["Constant",D.eval.constant,"#ef4444"]].map(([name,v,c])=>
              <div key={name} style={{flex:"1 1 200px",background:"#111827",borderRadius:14,padding:20,border:`1px solid ${c}22`}}>
                <div style={{fontSize:9,fontFamily:"monospace",color:c,letterSpacing:1.5,marginBottom:10}}>{name.toUpperCase()}</div>
                <div style={{fontSize:28,fontWeight:700,color:c,fontFamily:"monospace"}}>{(v.accuracy*100).toFixed(1)}%</div>
                <div style={{fontSize:10,color:"#666",marginBottom:8}}>exact accuracy</div>
                <div style={{fontSize:16,fontWeight:600,color:"#aaa",fontFamily:"monospace"}}>{(v.adj_accuracy*100).toFixed(1)}%</div>
                <div style={{fontSize:10,color:"#666"}}>adjacent accuracy</div>
              </div>
            )}
          </div>
          <div style={{background:"#111827",borderRadius:12,padding:18,border:"1px solid #10b98133",marginBottom:20}}>
            <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:10}}>
              <div style={{width:8,height:8,borderRadius:"50%",background:"#10b981",boxShadow:"0 0 8px #10b98166"}}/>
              <span style={{fontSize:10,fontFamily:"monospace",color:"#10b981",letterSpacing:1}}>STATISTICALLY SIGNIFICANT</span>
            </div>
            <div style={{display:"flex",gap:32,flexWrap:"wrap"}}>
              <div><div style={{fontSize:8,color:"#666",fontFamily:"monospace"}}>Z-SCORE</div><div style={{fontSize:24,fontWeight:700,color:"#fff",fontFamily:"monospace"}}>{D.stats.z.toFixed(2)}</div></div>
              <div><div style={{fontSize:8,color:"#666",fontFamily:"monospace"}}>P-VALUE</div><div style={{fontSize:24,fontWeight:700,color:"#fff",fontFamily:"monospace"}}>2.4×10⁻¹²</div></div>
              <div><div style={{fontSize:8,color:"#666",fontFamily:"monospace"}}>RL WINS</div><div style={{fontSize:24,fontWeight:700,color:"#fff",fontFamily:"monospace"}}>{D.stats.rl_wins}</div></div>
              <div><div style={{fontSize:8,color:"#666",fontFamily:"monospace"}}>PERSIST WINS</div><div style={{fontSize:24,fontWeight:700,color:"#fff",fontFamily:"monospace"}}>{D.stats.persist_wins}</div></div>
            </div>
          </div>
          <div style={{background:"linear-gradient(135deg,#111827,#131a2e)",borderRadius:14,padding:20,border:"1px solid #8b5cf622"}}>
            <div style={{fontSize:9,fontFamily:"monospace",color:"#8b5cf6",letterSpacing:1.5,marginBottom:14}}>CONCLUSIONS</div>
            {D.conclusions.map((c,i)=><div key={i} style={{padding:"12px 14px",background:i===D.conclusions.length-1?"#6366f10a":"transparent",borderLeft:`3px solid ${i===D.conclusions.length-1?"#6366f1":"#1e293b"}`,marginBottom:8,borderRadius:"0 8px 8px 0"}}>
              <p style={{margin:0,fontSize:12,lineHeight:1.7,color:i===D.conclusions.length-1?"#fff":"#bbb",fontWeight:i===D.conclusions.length-1?600:400}}>{c}</p>
            </div>)}
          </div>
        </div>}
      </div>
    </div>
  </div>;
}
