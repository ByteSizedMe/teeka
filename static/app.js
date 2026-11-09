// Shared by the family pages: navigation, data access, the certificate and the help card.
window.Teeka = (() => {
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch (e) {} throw new Error(m); }
    return r.json();
  }
  const post = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

  const PAGES = [['/dashboard', 'Home'], ['/people', 'People'], ['/library', 'Library'], ['/reports', 'Reports']];
  function nav() {
    const here = location.pathname;
    $('nav').replaceChildren(...PAGES.map(([href, label]) => {
      const a = el('a', here === href ? 'on' : '', label);
      a.href = href;
      if (here === href) a.setAttribute('aria-current', 'page');
      return a;
    }));
  }

  function helpCard(box) {
    box.replaceChildren(
      el('h3', null, 'If a real scam happened'),
      el('p', null, 'Stop paying and call your bank. Then report it straight away:'),
      Object.assign(el('a', 'help-num', 'Call 1930'), { href: 'tel:1930' }),
      Object.assign(el('a', 'help-link', 'cybercrime.gov.in'), { href: 'https://cybercrime.gov.in/', target: '_blank', rel: 'noopener' }),
      el('p', 'muted small', 'National cybercrime helpline and portal, as listed by I4C, Ministry of Home Affairs.'));
  }

  // One series: the share of times each red flag fooled them. The number sits beside the bar.
  function bars(box, rows, emptyText) {
    if (!rows.length) { box.replaceChildren(el('p', 'muted', emptyText)); return; }
    box.replaceChildren(...rows.map((r) => {
      const row = el('div', 'bar-row');
      row.title = r.title;
      const track = el('span', 'bar-track');
      const fill = el('i', 'bar-fill');
      fill.style.width = `${Math.max(r.pct, 2)}%`;
      track.append(fill);
      row.append(el('span', 'bar-label', r.label), track, el('span', 'bar-value', `${r.pct}%`));
      return row;
    }));
  }
  const flagRows = (flags) => flags.slice().sort((a, b) => b.pct - a.pct).map((f) => ({ label: f.en, pct: f.pct, title: `Fooled ${f.missed} of ${f.seen} times` }));

  // A share image with no name, phone number or anything said on a call.
  async function certificate(stats) {
    await document.fonts.load('48px "Tiro Devanagari Hindi"').catch(() => {});
    const W = 1200, H = 630, c = document.createElement('canvas');
    c.width = W; c.height = H;
    const g = c.getContext('2d');
    g.fillStyle = '#fbf6e9'; g.fillRect(0, 0, W, H);
    g.strokeStyle = '#0f5c4d'; g.lineWidth = 6; g.strokeRect(28, 28, W - 56, H - 56);
    g.lineWidth = 2; g.strokeRect(42, 42, W - 84, H - 84);
    g.textAlign = 'center'; g.fillStyle = '#0f5c4d';
    g.font = '76px "Tiro Devanagari Hindi", Georgia, serif'; g.fillText('Teeka टीका', W / 2, 150);
    g.fillStyle = '#55615a'; g.font = '30px Mukta, system-ui, sans-serif'; g.fillText('Scam-call vaccination card', W / 2, 200);
    g.fillStyle = '#22302b'; g.font = '50px "Tiro Devanagari Hindi", Georgia, serif';
    g.fillText(`${stats.doses} practice call${stats.doses === 1 ? '' : 's'} completed`, W / 2, 310);
    g.fillText(`${stats.protected} scam${stats.protected === 1 ? '' : 's'} refused`, W / 2, 380);
    g.save(); g.translate(W / 2, 498); g.rotate(-0.06);
    g.strokeStyle = '#0f5c4d'; g.lineWidth = 5; g.strokeRect(-210, -46, 420, 86);
    g.fillStyle = '#0f5c4d'; g.font = '800 46px Mukta, system-ui, sans-serif'; g.fillText('PRACTISED', 0, 14);
    g.restore();
    const a = el('a'); a.download = 'teeka-certificate.png';
    a.href = c.toDataURL('image/png');
    document.body.append(a); a.click(); a.remove();
    return a.href;
  }

  const shortDate = (iso) => { const [y, m, d] = iso.slice(0, 10).split('-').map(Number); return new Date(y, m - 1, d).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }); };
  return { $, el, api, post, nav, helpCard, bars, flagRows, certificate, shortDate };
})();
