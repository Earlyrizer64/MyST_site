---
title: Lennard-Jones Potential Explorer
---

# Lennard-Jones Potential Explorer

Use the particle selector to set σ and ε parameters, then drag the slider to move the particles and explore the potential energy curve.

```{raw} html
<style>
#lj-wrap { padding: 1rem 0; font-family: var(--font-sans); }
.lj-row { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
.lj-left { flex: 1; min-width: 260px; }
.lj-right { width: 200px; }
.ctrl-row { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.ctrl-label { font-size: 13px; color: var(--color-text-secondary); width: 20px; }
.ctrl-val { font-size: 13px; font-weight: 500; min-width: 48px; color: var(--color-text-primary); }
.stat-card { background: var(--color-background-secondary); border-radius: var(--border-radius-md); padding: 10px 14px; margin-bottom: 10px; }
.stat-label { font-size: 12px; color: var(--color-text-secondary); margin: 0 0 2px; }
.stat-val { font-size: 20px; font-weight: 500; color: var(--color-text-primary); margin: 0; }
select { width: 100%; margin-bottom: 10px; }
.particle-viz { width: 100%; height: 80px; border-radius: var(--border-radius-md); background: var(--color-background-secondary); margin-bottom: 16px; overflow: hidden; }
</style>

<div id="lj-wrap">
  <h2 class="sr-only">Interactive Lennard-Jones potential explorer with particle type selector and distance slider</h2>

  <div class="lj-row">
    <div class="lj-left">
      <svg class="particle-viz" id="pviz" viewBox="0 0 400 80" preserveAspectRatio="xMidYMid meet">
        <circle id="p1" cx="80" cy="40" r="14" fill="#185FA5" opacity="0.85"/>
        <circle id="p2" cx="200" cy="40" r="14" fill="#185FA5" opacity="0.85"/>
        <line id="pline" x1="94" y1="40" x2="186" y2="40" stroke="#888" stroke-width="1" stroke-dasharray="4 3"/>
        <text id="pdist" x="200" y="18" font-size="11" fill="#888" text-anchor="middle"></text>
      </svg>

      <div class="ctrl-row">
        <span class="ctrl-label">r</span>
        <input type="range" id="rSlider" min="0.8" max="4.0" step="0.01" value="1.5" style="flex:1;">
        <span class="ctrl-val" id="rVal">1.50 σ</span>
      </div>

      <div style="position: relative; width: 100%; height: 280px;">
        <canvas id="ljChart" role="img" aria-label="Lennard-Jones potential curve showing energy vs distance, with a movable point indicating current particle separation">LJ potential energy vs distance.</canvas>
      </div>
    </div>

    <div class="lj-right">
      <p style="font-size:12px; color:var(--color-text-secondary); margin:0 0 6px;">Particle type</p>
      <select id="particleSel">
        <option value="Ar">Argon (Ar)</option>
        <option value="Ne">Neon (Ne)</option>
        <option value="Kr">Krypton (Kr)</option>
        <option value="Xe">Xenon (Xe)</option>
        <option value="CH4">Methane (CH₄)</option>
        <option value="N2">Nitrogen (N₂)</option>
        <option value="custom">Custom</option>
      </select>

      <div id="customInputs" style="display:none; margin-bottom:10px;">
        <div style="display:flex; gap:8px; margin-bottom:6px; align-items:center;">
          <label style="font-size:12px; color:var(--color-text-secondary); width:28px;">σ (Å)</label>
          <input type="number" id="sigmaIn" value="3.4" step="0.1" style="flex:1; font-size:13px;">
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
          <label style="font-size:12px; color:var(--color-text-secondary); width:28px;">ε (K)</label>
          <input type="number" id="epsIn" value="120" step="1" style="flex:1; font-size:13px;">
        </div>
      </div>

      <div class="stat-card">
        <p class="stat-label">Energy V(r)</p>
        <p class="stat-val" id="statE">—</p>
      </div>
      <div class="stat-card">
        <p class="stat-label">r / σ</p>
        <p class="stat-val" id="statR">—</p>
      </div>
      <div class="stat-card">
        <p class="stat-label">σ (Å)</p>
        <p class="stat-val" id="statSig">—</p>
      </div>
      <div class="stat-card">
        <p class="stat-label">ε (K)</p>
        <p class="stat-val" id="statEps">—</p>
      </div>
      <div class="stat-card">
        <p class="stat-label">State</p>
        <p class="stat-val" id="statState" style="font-size:14px;">—</p>
      </div>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const PARTICLES = {
  Ar:  { sigma: 3.40, eps: 120.0 },
  Ne:  { sigma: 2.75, eps: 35.7  },
  Kr:  { sigma: 3.60, eps: 171.0 },
  Xe:  { sigma: 3.96, eps: 221.0 },
  CH4: { sigma: 3.73, eps: 148.0 },
  N2:  { sigma: 3.70, eps: 95.1  },
};

function lj(r_over_sigma) {
  const x = 1 / r_over_sigma;
  return 4 * (Math.pow(x, 12) - Math.pow(x, 6));
}

function getParams() {
  const sel = document.getElementById('particleSel').value;
  if (sel === 'custom') {
    return {
      sigma: parseFloat(document.getElementById('sigmaIn').value) || 3.4,
      eps:   parseFloat(document.getElementById('epsIn').value)   || 120
    };
  }
  return PARTICLES[sel];
}

const rPoints = [];
for (let i = 0.85; i <= 4.0; i += 0.02) rPoints.push(parseFloat(i.toFixed(3)));
const labels = rPoints.map(r => r.toFixed(2));

const isDark = matchMedia('(prefers-color-scheme: dark)').matches;
const gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.07)';
const axisColor = isDark ? 'rgba(255,255,255,0.35)' : 'rgba(0,0,0,0.35)';
const lineColor = '#185FA5';
const pointColor = '#D85A30';

const ctx = document.getElementById('ljChart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels,
    datasets: [
      {
        label: 'V(r) / ε',
        data: rPoints.map(r => {
          const v = lj(r); return (v < -1.5 || v > 3) ? null : parseFloat(v.toFixed(4));
        }),
        borderColor: lineColor,
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        spanGaps: false,
      },
      {
        label: 'Current r',
        data: [],
        pointRadius: 7,
        pointBackgroundColor: pointColor,
        pointBorderColor: pointColor,
        showLine: false,
        borderWidth: 0,
      }
    ]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false },
    },
    scales: {
      x: {
        title: { display: true, text: 'r / σ', color: axisColor, font: { size: 12 } },
        ticks: { color: axisColor, maxTicksLimit: 8, callback: (v, i) => rPoints[i] % 0.5 < 0.02 ? rPoints[i].toFixed(1) : '' },
        grid: { color: gridColor },
      },
      y: {
        min: -1.5,
        max: 3.0,
        title: { display: true, text: 'V(r) / ε', color: axisColor, font: { size: 12 } },
        ticks: { color: axisColor, maxTicksLimit: 6 },
        grid: { color: gridColor },
      }
    }
  }
});

function getState(vOverEps, rOverSigma) {
  if (vOverEps > 1.5) return '⚠ Repulsive';
  if (rOverSigma < 1.05) return 'Hard repulsion';
  if (Math.abs(vOverEps + 1) < 0.05) return '✦ Equilibrium';
  if (vOverEps < 0) return 'Attractive well';
  return 'Near-zero';
}

function update() {
  const r = parseFloat(document.getElementById('rSlider').value);
  const { sigma, eps } = getParams();
  const v = lj(r);
  const vClamped = Math.max(-1.5, Math.min(3, v));

  const idx = rPoints.findIndex(p => Math.abs(p - r) < 0.015);
  const ptData = new Array(rPoints.length).fill(null);
  if (idx >= 0) ptData[idx] = parseFloat(vClamped.toFixed(4));
  chart.data.datasets[1].data = ptData;
  chart.update('none');

  document.getElementById('rVal').textContent = r.toFixed(2) + ' σ';
  document.getElementById('statE').textContent = (v * eps).toFixed(1) + ' K';
  document.getElementById('statR').textContent = r.toFixed(2);
  document.getElementById('statSig').textContent = sigma.toFixed(2);
  document.getElementById('statEps').textContent = eps.toFixed(1);
  document.getElementById('statState').textContent = getState(v, r);

  const p1x = 80, p2xMin = 110, p2xMax = 340;
  const frac = (r - 0.8) / (4.0 - 0.8);
  const p2x = p2xMin + frac * (p2xMax - p2xMin);
  document.getElementById('p2').setAttribute('cx', p2x.toFixed(1));
  document.getElementById('pline').setAttribute('x2', (p2x - 14).toFixed(1));
  document.getElementById('pdist').setAttribute('x', ((p1x + p2x) / 2).toFixed(1));
  document.getElementById('pdist').textContent = 'r = ' + r.toFixed(2) + ' σ';
}

document.getElementById('rSlider').addEventListener('input', update);
document.getElementById('particleSel').addEventListener('change', () => {
  const sel = document.getElementById('particleSel').value;
  document.getElementById('customInputs').style.display = sel === 'custom' ? 'block' : 'none';
  update();
});
document.getElementById('sigmaIn').addEventListener('input', update);
document.getElementById('epsIn').addEventListener('input', update);

update();
</script>
```
