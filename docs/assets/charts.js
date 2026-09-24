// Utilitaires partagés par les pages : thème, formats français, loi normale,
// montage des graphiques Observable Plot et vues tableau.
// Script classique (pas de module) pour que les pages s'ouvrent en file://.
(function () {
  const fr = d3.formatLocale({
    decimal: ",",
    thousands: " ",
    grouping: [3],
    currency: ["", " €"],
    percent: " %",
  });
  const fmt = {
    num: (d, digits = 2) => fr.format(`,.${digits}f`)(d),
    int: (d) => fr.format(",d")(d),
    pct: (d, digits = 1) => fr.format(`.${digits}%`)(d),
    sigma: (d, digits = 2) => `${fr.format(`.${digits}f`)(d)} σ`,
    // Graduations : jusqu'à 3 décimales, sans zéros superflus (0,05 ; 0,1 ; 1).
    tick: (d) => fr.format(",.3~f")(d),
  };

  // --- Thème -------------------------------------------------------------
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function theme() {
    return {
      series: [1, 2, 3, 4, 5].map((i) => css(`--series-${i}`)),
      surface: css("--surface-1"),
      ink: css("--text-primary"),
      ink2: css("--text-secondary"),
      muted: css("--text-muted"),
      grid: css("--grid"),
      axis: css("--axis"),
      deemph: css("--mark-muted"),
    };
  }

  const THEMES = ["auto", "light", "dark"];
  const THEME_LABELS = { auto: "Thème : auto", light: "Thème : clair", dark: "Thème : sombre" };

  function readStoredTheme() {
    try { return localStorage.getItem("theme") || "auto"; } catch { return "auto"; }
  }

  function applyTheme(value) {
    if (value === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = value;
    try { localStorage.setItem("theme", value); } catch { /* stockage indisponible */ }
    const btn = document.querySelector(".theme-toggle");
    if (btn) btn.textContent = THEME_LABELS[value];
    redrawAll();
  }

  function initThemeToggle() {
    const btn = document.querySelector(".theme-toggle");
    let current = readStoredTheme();
    applyTheme(current);
    btn?.addEventListener("click", () => {
      current = THEMES[(THEMES.indexOf(current) + 1) % THEMES.length];
      applyTheme(current);
    });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);
  }

  // --- Loi normale -------------------------------------------------------
  // erf : Abramowitz & Stegun 7.1.26 (erreur < 1,5e-7), suffisant pour l'affichage.
  function erf(x) {
    const s = Math.sign(x);
    const a = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * a);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
    return s * y;
  }
  const pdf = (x, m = 0, s = 1) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  const cdf = (x, m = 0, s = 1) => 0.5 * (1 + erf((x - m) / (s * Math.SQRT2)));
  function quantile(p) {
    let lo = -10, hi = 10;
    for (let i = 0; i < 80; i++) {
      const mid = (lo + hi) / 2;
      if (cdf(mid) < p) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }
  const linspace = (a, b, n) => d3.range(n).map((i) => a + ((b - a) * i) / (n - 1));

  // --- Montage des graphiques -------------------------------------------
  const mounted = [];

  function draw(entry) {
    const width = entry.el.clientWidth;
    if (!width) return;
    entry.width = width;
    entry.el.replaceChildren(entry.render(theme(), width));
  }

  function redrawAll() {
    mounted.forEach(draw);
  }

  /** Monte un graphique : `render(t, width)` renvoie un nœud Plot, redessiné au resize et au changement de thème. */
  function mount(el, render) {
    const entry = { el, render, width: 0 };
    mounted.push(entry);
    draw(entry);
    let timer;
    new ResizeObserver(() => {
      if (el.clientWidth === entry.width) return;
      clearTimeout(timer);
      timer = setTimeout(() => draw(entry), 80);
    }).observe(el);
    return { redraw: () => draw(entry) };
  }

  /** Options communes : fond transparent, encre secondaire, grille en filet discret. */
  function frame(t, width, options = {}) {
    const { marks = [], ...rest } = options;
    return {
      width,
      height: width < 520 ? 260 : 320,
      marginLeft: 52,
      marginRight: 20,
      marginTop: 32,
      marginBottom: 44,
      style: { background: "transparent", color: t.ink2, fontFamily: "inherit", fontSize: "12px", overflow: "visible" },
      ...rest,
      marks: [Plot.gridY({ stroke: t.grid, strokeOpacity: 1 }), ...marks],
    };
  }

  /** Remplit un <details class="table-view"> avec un tableau (contenu inséré en textContent). */
  function table(details, columns, rows) {
    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    const tbl = document.createElement("table");
    const head = tbl.createTHead().insertRow();
    for (const col of columns) {
      const th = document.createElement("th");
      th.textContent = col.label;
      head.appendChild(th);
    }
    const body = tbl.createTBody();
    for (const row of rows) {
      const tr = body.insertRow();
      for (const col of columns) tr.insertCell().textContent = col.value(row);
    }
    wrap.appendChild(tbl);
    details.querySelector(".table-wrap")?.remove();
    details.appendChild(wrap);
  }

  /** Remplace le texte des éléments [data-bind="chemin.vers.valeur"] par la valeur formatée. */
  function bind(root, data, formatters) {
    for (const el of root.querySelectorAll("[data-bind]")) {
      const key = el.dataset.bind;
      const value = key.split(".").reduce((o, k) => o?.[k], data);
      el.textContent = formatters[key] ? formatters[key](value) : String(value);
    }
  }

  window.Charts = { fmt, theme, initThemeToggle, pdf, cdf, quantile, linspace, mount, frame, table, bind };
})();
