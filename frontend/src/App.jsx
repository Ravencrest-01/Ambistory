import React, { useState, useEffect, useRef } from "react";
import {
  BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell, Tooltip,
  ReferenceLine, LabelList,
} from "recharts";
import {
  Sparkles, ArrowRight, RotateCcw, Layers, Quote, ScanSearch,
  CheckCircle2, AlertTriangle, ChevronRight,
} from "lucide-react";

/* Real development-set examples with the model's actual predictions. */
const EXAMPLES = [
  { homonym: "track", pre: "The detectives arrived at the abandoned train station. They were looking for signs of the missing artifact. A faint trail caught their attention.", sent: "They followed the track.", end: "They found interesting clues that helped them solve the problem.", sense: "evidence pointing to a possible solution", pred: 4, human: 4.2, sd: 0.84 },
  { homonym: "suit", pre: "John woke up on the wrong side of the bed. His mind was racing with thoughts from the day before. He had been expecting a delivery and it was finally here.", sent: "John's morning could hardly have gone much worse; an awful suit had arrived at his front door.", end: "As he opened the envelope and scanned the letterhead, his heart sank as he recognized the law firm's name.", sense: "a set of garments for outerwear, all of the same fabric and color", pred: 2, human: 1.6, sd: 1.34 },
  { homonym: "stars", pre: "Tom had spent months planning his special trip. He packed his bags, making sure to include his telescope and binoculars. As the day approached, his excitement grew.", sent: "He looked forward to seeing the stars.", end: "This would be the first time he'd slept out on a clear night with just his sleeping bag.", sense: "an actor who plays a principal role", pred: 2, human: 1.6, sd: 1.34 },
  { homonym: "coached", pre: "Tom had always loved sports and vintage vehicles. When the team asked for his help, he couldn't refuse. He saw this as a chance to combine his two passions.", sent: "He coached the team all the way to the championship.", end: "He always loved to drive his team around.", sense: "drive a coach", pred: 2, human: 2.4, sd: 1.67 },
  { homonym: "suits", pre: "John struggled through a tough year. Every week seemed to bring a new challenge. He found himself constantly attending meetings downtown.", sent: "He had many suits last year.", end: "It was tiring for him, but he was determined to clear his name and get justice.", sense: "a proceeding in a court of law in which an individual seeks a legal remedy", pred: 4, human: 4.0, sd: 0.71 },
  { homonym: "mean", pre: "Dr. Aldridge was preparing for the annual research presentation. Two groups of students took part in the experiment last week. Each group had very different reactions.", sent: "The results of the test were higher in the mean cohort.", end: "He really wanted the results to be better in the nicer group.", sense: "approximating the statistical norm, average, or expected value", pred: 4, human: 3.8, sd: 1.79 },
  { homonym: "reservation", pre: "Anna hesitated when she saw the invitation from her coworkers. She wasn't sure if she should go because she liked her evenings quiet. They always had a great time, but she felt overwhelmed.", sent: "She agreed to join them for dinner, but only after making a reservation.", end: "She didn't want to waste time waiting for an available table.", sense: "a statement that limits or restricts some claim", pred: 2, human: 2.0, sd: 1.73 },
];

const MODELS = [
  { name: "TF-IDF baseline", sp: 0.11, acc: 0.62, mae: 1.02 },
  { name: "RoBERTa-base", sp: 0.37, acc: 0.71, mae: 0.96 },
  { name: "DeBERTa-v3", sp: 0.49, acc: 0.77, mae: 0.84 },
];
const BUCKETS = [
  { r: "1", acc: 0.56 }, { r: "2", acc: 0.79 }, { r: "3", acc: 0.95 },
  { r: "4", acc: 0.81 }, { r: "5", acc: 0.57 },
];

/* Cool-to-warm spectrum: 1 = implausible (indigo), 5 = plausible (amber). */
function scaleColor(v) {
  const stops = [
    [1, [99, 102, 241]], [2, [56, 152, 219]], [3, [20, 184, 166]],
    [4, [245, 176, 32]], [5, [251, 146, 60]],
  ];
  const x = Math.max(1, Math.min(5, v));
  let a = stops[0], b = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (x >= stops[i][0] && x <= stops[i + 1][0]) { a = stops[i]; b = stops[i + 1]; break; }
  }
  const t = (x - a[0]) / (b[0] - a[0] || 1);
  const c = a[1].map((ai, i) => Math.round(ai + (b[1][i] - ai) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}
const pct = (v) => ((v - 1) / 4) * 100;

function useReveal() {
  const ref = useRef(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current; if (!el) return;
    const io = new IntersectionObserver(
      ([e]) => e.isIntersecting && setSeen(true),
      { threshold: 0.18 }
    );
    io.observe(el); return () => io.disconnect();
  }, []);
  return [ref, seen];
}

function Spectrum({ pred, human, revealed }) {
  return (
    <div className="spectrum">
      <div className="spectrum-track" />
      <div className="spectrum-ticks">
        {[1, 2, 3, 4, 5].map((n) => <span key={n}>{n}</span>)}
      </div>
      <div className={`marker human ${revealed ? "on" : ""}`} style={{ left: `${pct(human)}%` }}>
        <div className="dot ghost" />
        <div className="tag">humans&nbsp;{human.toFixed(1)}</div>
      </div>
      <div className={`marker model ${revealed ? "on" : ""}`} style={{ left: `${pct(pred)}%` }}>
        <div className="dot" style={{ background: scaleColor(pred), boxShadow: `0 0 18px ${scaleColor(pred)}` }} />
        <div className="tag strong">model&nbsp;{pred}</div>
      </div>
    </div>
  );
}

export default function App() {
  const [i, setI] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const ex = EXAMPLES[i];
  const agree = Math.abs(ex.pred - ex.human) <= Math.max(ex.sd, 1);

  const next = () => { setRevealed(false); setI((i + 1) % EXAMPLES.length); };

  const [r1, s1] = useReveal();
  const [r2, s2] = useReveal();

  const parts = ex.sent.split(new RegExp(`(${ex.homonym})`, "i"));

  return (
    <div className="amb-root">
      <style>{CSS}</style>

      {/* atmosphere */}
      <div className="aurora" aria-hidden />

      {/* nav */}
      <header className="nav">
        <div className="brand">
          <span className="brand-mark">Ambi<span className="grad-text">Story</span></span>
          <span className="brand-sub">word-sense plausibility</span>
        </div>
        <div className="nav-stat">
          <span className="nav-stat-num grad-text">0.49</span>
          <span className="nav-stat-lab">Spearman&nbsp;·&nbsp;dev</span>
        </div>
      </header>

      {/* hero */}
      <section className="hero">
        <div className="hero-copy">
          <div className="eyebrow"><Sparkles size={13} /> SemEval&nbsp;2026 · Task&nbsp;5</div>
          <h1>
            One word.<br />Two meanings.<br />
            <span className="grad-text">How plausible is each?</span>
          </h1>
          <p className="lede">
            A homonym shifts meaning with the story around it. We trained a model to
            rate on a 1&ndash;5 scale how plausible a given sense is, matching
            how real readers judged it. Read a story and see what it decides.
          </p>
          <div className="hero-tags">
            <span>DeBERTa-v3 cross-encoder</span>
            <span>graded, not binary</span>
            <span>2,280 stories</span>
          </div>
        </div>

        {/* the lab — signature element */}
        <div className="lab">
          <div className="lab-head">
            <div className="lab-title"><ScanSearch size={15} /> Plausibility Lab</div>
            <div className="lab-count">{i + 1} / {EXAMPLES.length}</div>
          </div>

          <div className="story">
            <Quote size={16} className="qmark" />
            <p className="pre">{ex.pre}</p>
            <p className="sent">
              {parts.map((p, k) =>
                p.toLowerCase() === ex.homonym.toLowerCase()
                  ? <mark key={k}>{p}</mark> : <span key={k}>{p}</span>
              )}
            </p>
            {ex.end && <p className="end">{ex.end}</p>}
          </div>

          <div className="sense">
            <span className="sense-lab">Is <b>“{ex.homonym}”</b> being used to mean…</span>
            <span className="sense-gloss">{ex.sense}</span>
          </div>

          <Spectrum pred={ex.pred} human={ex.human} revealed={revealed} />

          <div className="lab-actions">
            {!revealed ? (
              <button className="btn primary" onClick={() => setRevealed(true)}>
                Reveal judgement <ArrowRight size={15} />
              </button>
            ) : (
              <div className={`verdict ${agree ? "ok" : "miss"}`}>
                {agree ? <CheckCircle2 size={15} /> : <AlertTriangle size={15} />}
                {agree
                  ? "Within the band of human disagreement"
                  : "Outside the human band — a miss"}
              </div>
            )}
            <button className="btn ghost" onClick={next}>
              Next story <RotateCcw size={14} />
            </button>
          </div>
        </div>
      </section>

      {/* pipeline */}
      <section className="band" ref={r1}>
        <div className={`section-head ${s1 ? "in" : ""}`}>
          <span className="kicker">how it reads a story</span>
          <h2>Story and sense, weighed together</h2>
          <p>
            Both the narrative and the candidate meaning go into one cross-encoder,
            so the model can judge how the words actually interact &mdash; the way an
            ending can tip a word from one reading to another.
          </p>
        </div>
        <div className={`pipe ${s1 ? "in" : ""}`}>
          <div className="pipe-node">
            <span className="pipe-lab">input</span>
            <strong>The story</strong>
            <em>precontext · sentence · ending</em>
          </div>
          <ChevronRight className="pipe-arrow" />
          <div className="pipe-node">
            <span className="pipe-lab">input</span>
            <strong>A candidate sense</strong>
            <em>gloss · usage example</em>
          </div>
          <ChevronRight className="pipe-arrow" />
          <div className="pipe-node accent">
            <span className="pipe-lab"><Layers size={12} /> model</span>
            <strong>DeBERTa-v3</strong>
            <em>cross-attention · regression head</em>
          </div>
          <ChevronRight className="pipe-arrow" />
          <div className="pipe-node out">
            <span className="pipe-lab">output</span>
            <strong className="grad-text">1 – 5 score</strong>
            <em>predicted plausibility</em>
          </div>
        </div>
      </section>

      {/* results */}
      <section className="band" ref={r2}>
        <div className={`section-head ${s2 ? "in" : ""}`}>
          <span className="kicker">results · development set</span>
          <h2>From lexical guessing to genuine reading</h2>
          <p>
            Bag-of-words features barely move past chance because dictionary
            definitions and story text share so few words. The transformer reasons
            about meaning instead, roughly quadrupling rank agreement with humans.
          </p>
        </div>

        <div className={`charts ${s2 ? "in" : ""}`}>
          <div className="card chart-card">
            <div className="card-lab">Spearman correlation, by model</div>
            <ResponsiveContainer width="100%" height={230}>
              <BarChart data={MODELS} margin={{ top: 18, right: 8, left: -18, bottom: 0 }}>
                <XAxis dataKey="name" tick={{ fill: "#a9a3c9", fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 0.6]} tick={{ fill: "#6f6a90", fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} contentStyle={tipStyle} itemStyle={{ color: "#efeaf7" }} />
                <Bar dataKey="sp" radius={[6, 6, 0, 0]}>
                  {MODELS.map((m, k) => (
                    <Cell key={k} fill={k === 2 ? "url(#g)" : "#3b3766"} />
                  ))}
                  <LabelList dataKey="sp" position="top" fill="#efeaf7" fontSize={12} formatter={(v) => v.toFixed(2)} />
                </Bar>
                <defs>
                  <linearGradient id="g" x1="0" y1="1" x2="0" y2="0">
                    <stop offset="0%" stopColor="#6366f1" />
                    <stop offset="100%" stopColor="#fbbf24" />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card chart-card">
            <div className="card-lab">Accuracy within stdev, by true rating</div>
            <ResponsiveContainer width="100%" height={230}>
              <BarChart data={BUCKETS} margin={{ top: 18, right: 8, left: -18, bottom: 0 }}>
                <XAxis dataKey="r" tick={{ fill: "#a9a3c9", fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 1]} tick={{ fill: "#6f6a90", fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} contentStyle={tipStyle} itemStyle={{ color: "#efeaf7" }} />
                <ReferenceLine y={0.77} stroke="#6f6a90" strokeDasharray="4 4" />
                <Bar dataKey="acc" radius={[6, 6, 0, 0]}>
                  {BUCKETS.map((b, k) => (
                    <Cell key={k} fill={scaleColor(Number(b.r))} fillOpacity={0.92} />
                  ))}
                  <LabelList dataKey="acc" position="top" fill="#efeaf7" fontSize={12} formatter={(v) => v.toFixed(2)} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="chart-note">
              Strong in the middle, weaker at the extremes — the model hedges away
              from 1s and 5s, the main thing left to fix.
            </div>
          </div>
        </div>

        <div className={`stat-row ${s2 ? "in" : ""}`}>
          {[
            ["0.49", "Spearman ρ"], ["0.77", "within stdev"],
            ["0.84", "mean abs. error"], ["220", "homonyms"],
          ].map(([n, l]) => (
            <div className="stat" key={l}>
              <div className="stat-num grad-text">{n}</div>
              <div className="stat-lab">{l}</div>
            </div>
          ))}
        </div>
      </section>

      <footer className="foot">
        <div className="foot-l">
          <span className="brand-mark">Ambi<span className="grad-text">Story</span></span>
          <p>Rating word-sense plausibility in ambiguous narratives.</p>
        </div>
        <div className="foot-tech">
          {["PyTorch", "Transformers", "DeBERTa-v3", "scikit-learn"].map((t) => (
            <span key={t}>{t}</span>
          ))}
        </div>
      </footer>
    </div>
  );
}

const tipStyle = {
  background: "#1c1840", border: "1px solid #34305e",
  borderRadius: 10, color: "#efeaf7", fontSize: 12,
};

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

.amb-root{
  --ink:#100e26; --ink2:#16133073; --panel:#1a1640; --panel2:#221d52;
  --line:#332e5c; --text:#efeaf7; --mut:#a9a3c9; --dim:#6f6a90;
  --violet:#a78bfa; --amber:#fbbf24;
  position:relative; min-height:100vh; background:#100e26; color:var(--text);
  font-family:'Inter',system-ui,sans-serif; overflow-x:hidden;
  -webkit-font-smoothing:antialiased;
}
.amb-root *{box-sizing:border-box;}
.grad-text{
  background:linear-gradient(95deg,#8b7cf6 0%,#22b8cf 38%,#34d399 60%,#fbbf24 100%);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.aurora{
  position:fixed; inset:0; z-index:0; pointer-events:none; opacity:.55;
  background:
    radial-gradient(45rem 30rem at 12% -8%, rgba(139,124,246,.30), transparent 60%),
    radial-gradient(40rem 28rem at 92% 8%, rgba(251,176,32,.16), transparent 60%),
    radial-gradient(50rem 40rem at 70% 100%, rgba(34,184,207,.16), transparent 60%);
  filter:saturate(1.1);
}
.amb-root > *:not(.aurora){position:relative; z-index:1;}

.nav{
  display:flex; align-items:center; justify-content:space-between;
  padding:22px clamp(20px,5vw,64px); max-width:1180px; margin:0 auto;
}
.brand{display:flex; flex-direction:column; gap:1px;}
.brand-mark{font-family:'Fraunces',serif; font-weight:700; font-size:23px; letter-spacing:-.01em;}
.brand-sub{font-size:11px; color:var(--dim); letter-spacing:.14em; text-transform:uppercase;}
.nav-stat{text-align:right; display:flex; flex-direction:column;}
.nav-stat-num{font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:500;}
.nav-stat-lab{font-size:10px; color:var(--dim); letter-spacing:.12em; text-transform:uppercase;}

.hero{
  max-width:1180px; margin:0 auto; padding:clamp(18px,4vw,46px) clamp(20px,5vw,64px) 40px;
  display:grid; grid-template-columns:1.02fr 1.15fr; gap:clamp(28px,4vw,56px); align-items:center;
}
.eyebrow{
  display:inline-flex; align-items:center; gap:7px; font-size:11.5px; color:var(--violet);
  letter-spacing:.16em; text-transform:uppercase; margin-bottom:18px;
  border:1px solid var(--line); padding:6px 12px; border-radius:999px; background:rgba(167,139,250,.06);
}
.hero h1{
  font-family:'Fraunces',serif; font-weight:600; font-size:clamp(38px,5.4vw,64px);
  line-height:1.02; letter-spacing:-.02em; margin:0 0 20px;
}
.lede{font-size:clamp(14.5px,1.4vw,16.5px); line-height:1.6; color:var(--mut); max-width:38ch; margin:0 0 22px;}
.hero-tags{display:flex; flex-wrap:wrap; gap:8px;}
.hero-tags span{
  font-size:12px; color:var(--mut); border:1px solid var(--line);
  padding:5px 11px; border-radius:8px; background:var(--ink2);
}
.hero-copy{animation:rise .8s cubic-bezier(.2,.7,.2,1) both;}

/* LAB */
.lab{
  background:linear-gradient(180deg,rgba(34,29,82,.9),rgba(22,19,48,.92));
  border:1px solid var(--line); border-radius:20px; padding:22px;
  box-shadow:0 30px 80px -40px rgba(0,0,0,.8), inset 0 1px 0 rgba(255,255,255,.04);
  animation:rise .8s .12s cubic-bezier(.2,.7,.2,1) both;
}
.lab-head{display:flex; align-items:center; justify-content:space-between; margin-bottom:16px;}
.lab-title{display:inline-flex; align-items:center; gap:8px; font-size:13px; font-weight:600; color:var(--violet); letter-spacing:.04em;}
.lab-count{font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--dim);}
.story{position:relative; background:var(--ink2); border:1px solid var(--line); border-radius:14px; padding:18px 18px 16px; margin-bottom:14px;}
.qmark{position:absolute; top:12px; right:14px; color:var(--line);}
.story .pre{font-size:13.5px; line-height:1.6; color:var(--mut); margin:0 0 10px;}
.story .sent{font-family:'Fraunces',serif; font-size:19px; line-height:1.45; margin:0 0 10px; color:var(--text);}
.story .sent mark{
  background:linear-gradient(transparent 58%, rgba(251,176,32,.34) 0);
  color:#fde9b8; padding:0 2px; border-radius:2px; font-style:italic;
}
.story .end{font-size:13.5px; line-height:1.6; color:var(--mut); margin:0; padding-top:10px; border-top:1px dashed var(--line);}
.sense{display:flex; flex-direction:column; gap:4px; margin-bottom:24px;}
.sense-lab{font-size:13px; color:var(--mut);}
.sense-lab b{color:var(--text);}
.sense-gloss{font-size:14.5px; color:var(--text); font-weight:500; line-height:1.4;}

/* spectrum */
.spectrum{position:relative; height:74px; margin:6px 4px 18px;}
.spectrum-track{
  position:absolute; top:34px; left:0; right:0; height:8px; border-radius:99px;
  background:linear-gradient(90deg,#6366f1,#22b8cf,#34d399,#fbbf24,#fb923c);
  opacity:.85;
}
.spectrum-ticks{position:absolute; top:48px; left:0; right:0; display:flex; justify-content:space-between;}
.spectrum-ticks span{font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--dim);}
.marker{position:absolute; top:0; transform:translateX(-50%); opacity:0; transition:left .9s cubic-bezier(.2,.8,.2,1), opacity .5s;}
.marker.on{opacity:1;}
.marker .dot{width:18px; height:18px; border-radius:50%; margin:26px auto 0; border:2px solid #100e26;}
.marker .dot.ghost{background:transparent; border:2px dashed var(--mut); width:16px; height:16px; margin-top:27px;}
.marker .tag{
  position:absolute; top:-2px; left:50%; transform:translateX(-50%);
  font-family:'JetBrains Mono',monospace; font-size:10.5px; white-space:nowrap;
  padding:3px 7px; border-radius:6px; border:1px solid var(--line); background:var(--panel2); color:var(--mut);
}
.marker.model .tag.strong{color:var(--text); border-color:rgba(251,176,32,.4);}
.marker.human{top:30px;}
.marker.human .tag{top:auto; bottom:-4px;}

.lab-actions{display:flex; align-items:center; gap:12px; flex-wrap:wrap;}
.btn{
  display:inline-flex; align-items:center; gap:8px; font-size:13.5px; font-weight:600;
  padding:11px 16px; border-radius:11px; cursor:pointer; border:1px solid var(--line);
  transition:transform .15s, background .2s, border-color .2s; font-family:inherit;
}
.btn:active{transform:translateY(1px);}
.btn.primary{
  background:linear-gradient(95deg,#8b7cf6,#22b8cf); color:#100e26; border:none;
  box-shadow:0 10px 30px -12px rgba(139,124,246,.8);
}
.btn.primary:hover{filter:brightness(1.07);}
.btn.ghost{background:transparent; color:var(--mut);}
.btn.ghost:hover{border-color:var(--violet); color:var(--text);}
.verdict{display:inline-flex; align-items:center; gap:8px; font-size:13px; font-weight:600; padding:10px 14px; border-radius:11px;}
.verdict.ok{color:#6ee7b7; background:rgba(52,211,153,.1); border:1px solid rgba(52,211,153,.25);}
.verdict.miss{color:#fbbf24; background:rgba(251,176,32,.1); border:1px solid rgba(251,176,32,.28);}

/* bands */
.band{max-width:1180px; margin:0 auto; padding:clamp(40px,7vw,84px) clamp(20px,5vw,64px);}
.section-head{max-width:62ch; opacity:0; transform:translateY(22px); transition:.7s cubic-bezier(.2,.7,.2,1);}
.section-head.in{opacity:1; transform:none;}
.kicker{font-size:11.5px; color:var(--violet); letter-spacing:.18em; text-transform:uppercase;}
.section-head h2{font-family:'Fraunces',serif; font-weight:600; font-size:clamp(26px,3.4vw,38px); line-height:1.1; letter-spacing:-.02em; margin:12px 0 12px;}
.section-head p{font-size:15px; line-height:1.65; color:var(--mut); margin:0;}

.pipe{
  display:flex; align-items:stretch; gap:10px; margin-top:34px; flex-wrap:wrap;
  opacity:0; transform:translateY(22px); transition:.7s .1s cubic-bezier(.2,.7,.2,1);
}
.pipe.in{opacity:1; transform:none;}
.pipe-node{
  flex:1 1 180px; background:var(--panel); border:1px solid var(--line); border-radius:14px;
  padding:16px 16px 18px; display:flex; flex-direction:column; gap:6px; min-width:0;
}
.pipe-node.accent{border-color:rgba(167,139,250,.45); background:linear-gradient(180deg,rgba(167,139,250,.12),var(--panel));}
.pipe-node.out{border-color:rgba(251,176,32,.35);}
.pipe-lab{display:inline-flex; align-items:center; gap:5px; font-size:10.5px; letter-spacing:.14em; text-transform:uppercase; color:var(--dim);}
.pipe-node strong{font-size:16px; font-weight:600;}
.pipe-node em{font-style:normal; font-size:12px; color:var(--mut);}
.pipe-arrow{color:var(--line); align-self:center; flex:0 0 auto;}

.charts{display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:34px;
  opacity:0; transform:translateY(22px); transition:.7s .1s cubic-bezier(.2,.7,.2,1);}
.charts.in{opacity:1; transform:none;}
.card{background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:18px;}
.card-lab{font-size:12.5px; color:var(--mut); margin-bottom:8px; font-weight:500;}
.chart-note{font-size:12px; color:var(--dim); line-height:1.5; margin-top:10px;}

.stat-row{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-top:22px;
  opacity:0; transform:translateY(22px); transition:.7s .2s cubic-bezier(.2,.7,.2,1);}
.stat-row.in{opacity:1; transform:none;}
.stat{background:var(--ink2); border:1px solid var(--line); border-radius:14px; padding:18px; text-align:center;}
.stat-num{font-family:'JetBrains Mono',monospace; font-size:clamp(24px,3vw,32px); font-weight:500;}
.stat-lab{font-size:11.5px; color:var(--dim); letter-spacing:.06em; margin-top:4px;}

.foot{
  max-width:1180px; margin:0 auto; padding:40px clamp(20px,5vw,64px) 56px;
  border-top:1px solid var(--line); display:flex; justify-content:space-between;
  align-items:flex-end; gap:24px; flex-wrap:wrap;
}
.foot-l p{font-size:13px; color:var(--mut); margin:6px 0 0;}
.foot-tech{display:flex; gap:8px; flex-wrap:wrap;}
.foot-tech span{font-size:12px; color:var(--mut); border:1px solid var(--line); padding:5px 10px; border-radius:8px; font-family:'JetBrains Mono',monospace;}

@keyframes rise{from{opacity:0; transform:translateY(26px);} to{opacity:1; transform:none;}}

@media (max-width:880px){
  .hero{grid-template-columns:1fr;}
  .charts{grid-template-columns:1fr;}
  .stat-row{grid-template-columns:repeat(2,1fr);}
  .pipe-arrow{transform:rotate(90deg);}
}
@media (prefers-reduced-motion:reduce){
  *{animation:none !important; transition:none !important;}
  .section-head,.pipe,.charts,.stat-row{opacity:1; transform:none;}
  .marker{transition:none;}
}
`;
