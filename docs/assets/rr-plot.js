// Généré par rr-style/build.py depuis tokens.json — ne pas éditer.
// Thème rr/ pour Observable Plot (script classique : fonctionne en file://).
// Requiert d3 et rr-tokens.css. Expose window.rrPlot.
(function () {
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const palette = {"light": ["#9E5468", "#4670A8", "#854F14", "#8368A8", "#1A653A"], "dark": ["#BA5F78", "#5E97CD", "#B98749", "#9E83C5", "#3F936E"]};

  /** Couleurs courantes, lues sur les variables CSS (suit le thème clair/sombre). */
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
      accent: css("--accent"),
    };
  }

  /** Options communes : fond transparent, texte mono secondaire, grille horizontale discrète. */
  function frame(t, width, options = {}) {
    const { marks = [], ...rest } = options;
    return {
      width,
      height: width < 520 ? 260 : 320,
      marginLeft: 52, marginRight: 20, marginTop: 32, marginBottom: 44,
      style: { background: "transparent", color: t.ink2, fontFamily: "var(--font-mono)", fontSize: "11px", overflow: "visible" },
      ...rest,
      marks: [Plot.gridY({ stroke: t.grid, strokeOpacity: 1 }), ...marks],
    };
  }

  const locale = typeof d3 !== "undefined"
    ? d3.formatLocale({ decimal: ",", thousands: "\u202f", grouping: [3], currency: ["", " €"], percent: " %" })
    : null;

  window.rrPlot = { palette, theme, frame, locale };
})();
