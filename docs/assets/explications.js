// Graphiques de la page « Marges d'erreur et variations ».
// Données : docs/data/explications.js (généré par analyses/explications.py).
(function () {
  const { fmt, pdf, cdf, quantile, linspace, mount, frame, table, bind } = Charts;
  const DATA = window.DATA.explications;
  const Z95 = 1.959964;
  const $ = (sel, root = document) => root.querySelector(sel);

  Charts.initThemeToggle();

  bind(document, DATA, {
    "femmes.min": fmt.int,
    "femmes.max": fmt.int,
    "femmes.moyenne": (v) => fmt.int(Math.round(v)),
    "femmes.ecart_type": (v) => fmt.num(v, 0),
    "femmes.part_dans_ecart_type": (v) => fmt.pct(v, 0),
    "inversion.seuil_analytique": (v) => fmt.num(v, 3),
    "inversion.p_seuil_mc": (v) => fmt.pct(v, 2),
  });

  const sigmaAxis = (label = "Position (en écarts-types σ)") => ({ label, labelAnchor: "center", labelArrow: "none", tickFormat: fmt.tick });
  const densityAxis = { label: "Densité", labelArrow: "none", ticks: 4, tickFormat: fmt.tick };
  const baseline = (t) => Plot.ruleY([0], { stroke: t.axis });

  /** Relie un groupe de boutons préréglés à un curseur. */
  function presets(fig, input, onChange) {
    const buttons = [...fig.querySelectorAll(".presets button")];
    const sync = () => {
      for (const b of buttons) b.setAttribute("aria-pressed", String(Math.abs(+b.dataset.value - +input.value) < 1e-9));
      onChange(+input.value);
    };
    for (const b of buttons) b.addEventListener("click", () => { input.value = b.dataset.value; sync(); });
    input.addEventListener("input", sync);
    sync();
  }

  // --- Deux lois normales (valeurs observées à 0 et à d) -------------------
  function twoNormals(t, width, { d, bands = true, tails = false, points = [], extra = [], xmin = -4, xmax = d + 4 }) {
    const [c1, c2] = t.series;
    const xs = linspace(xmin, xmax, 321);
    const curve = (m) => xs.map((x) => ({ x, y: pdf(x, m) }));
    const zone = (a, b, m) => linspace(Math.max(a, xmin), Math.min(b, xmax), 81).map((x) => ({ x, y: pdf(x, m) }));
    const wide = xs.map((x) => ({ x, a: pdf(x, 0), b: pdf(x, d) }));
    const marks = [];
    if (bands) {
      marks.push(Plot.areaY(zone(-Z95, Z95, 0), { x: "x", y: "y", fill: c1, fillOpacity: 0.14 }));
      marks.push(Plot.areaY(zone(d - Z95, d + Z95, d), { x: "x", y: "y", fill: c2, fillOpacity: 0.14 }));
    }
    if (tails) {
      marks.push(Plot.areaY(zone(Z95, xmax, 0), { x: "x", y: "y", fill: c1, fillOpacity: 0.6 }));
      marks.push(Plot.areaY(zone(xmin, d - Z95, d), { x: "x", y: "y", fill: c2, fillOpacity: 0.6 }));
    }
    marks.push(
      baseline(t),
      Plot.line(curve(0), { x: "x", y: "y", stroke: c1, strokeWidth: 2 }),
      Plot.line(curve(d), { x: "x", y: "y", stroke: c2, strokeWidth: 2 }),
    );
    marks.push(...extra);
    if (points.length) {
      const pts = points.map((p) => ({ ...p, y: pdf(p.x, p.m), color: t.series[p.serie] }));
      marks.push(
        Plot.ruleX(pts, { x: "x", y1: 0, y2: "y", stroke: "color", strokeWidth: 1.5 }),
        Plot.dot(pts, { x: "x", y: "y", r: 5, fill: "color", stroke: t.surface, strokeWidth: 2 }),
      );
    }
    marks.push(
      Plot.ruleX(wide, Plot.pointerX({ x: "x", stroke: t.muted })),
      Plot.tip(wide, Plot.pointerX({
        x: "x",
        y: (r) => Math.max(r.a, r.b),
        channels: { "Densité, première valeur": "a", "Densité, seconde valeur": "b" },
        format: { x: (v) => fmt.sigma(v), y: false, "Densité, première valeur": (v) => fmt.num(v, 3), "Densité, seconde valeur": (v) => fmt.num(v, 3) },
      })),
    );
    return Plot.plot(frame(t, width, { x: { ...sigmaAxis(), domain: [xmin, xmax] }, y: { ...densityAxis, domain: [0, 0.42] }, marks }));
  }

  const pInversion = (d) => cdf(-d / Math.SQRT2);
  const margesLabel = (d) => (d < 2 * Z95 - 1e-3 ? "se chevauchent" : d <= 2 * Z95 + 1e-2 ? "jointives" : "séparées");

  // --- Pile ou face ------------------------------------------------------
  {
    const fig = $("#fig-pile-face");
    const { effectifs, lancers, repetitions } = DATA.pile_ou_face;
    const s = Math.sqrt(lancers);
    // Les sommes sont paires : chaque barre couvre un pas de 2.
    const attendu = (k) => repetitions * 2 * pdf(k, 0, s);
    mount($(".plot", fig), (t, width) => Plot.plot(frame(t, width, {
      x: { label: "Somme des 100 lancers (pile = +1, face = −1)", labelAnchor: "center", labelArrow: "none", domain: [-40, 40] },
      y: { label: "Répétitions", labelArrow: "none" },
      marks: [
        Plot.rectY(effectifs, { x1: (d) => d.somme - 0.85, x2: (d) => d.somme + 0.85, y: "n", fill: t.series[0], ry2: 4 }),
        baseline(t),
        Plot.line(linspace(-40, 40, 321), { x: (k) => k, y: attendu, stroke: t.series[1], strokeWidth: 2 }),
        Plot.tip(effectifs, Plot.pointerX({
          x: "somme", y: "n",
          channels: { Répétitions: "n", "Attendu (loi normale)": (d) => attendu(d.somme) },
          format: { x: false, y: false, Répétitions: true, "Attendu (loi normale)": (v) => fmt.num(v, 1) },
          title: null,
        })),
      ],
    })));
    table($("details", fig), [
      { label: "Somme", value: (d) => fmt.int(d.somme) },
      { label: "Répétitions", value: (d) => fmt.int(d.n) },
      { label: "Attendu (loi normale)", value: (d) => fmt.num(attendu(d.somme), 1) },
    ], effectifs);
  }

  // --- Échantillons de femmes -------------------------------------------
  {
    const fig = $("#fig-femmes");
    const { effectifs, moyenne, ecart_type } = DATA.femmes;
    const dedans = (v) => Math.abs(v - moyenne) <= ecart_type;
    const reperes = [
      { x: moyenne - ecart_type, label: "−1 σ" },
      { x: moyenne, label: "moyenne" },
      { x: moyenne + ecart_type, label: "+1 σ" },
    ];
    mount($(".plot", fig), (t, width) => Plot.plot(frame(t, width, {
      x: { label: "Nombre de femmes dans l'échantillon", labelAnchor: "center", labelArrow: "none", tickFormat: fmt.int },
      y: { label: "Échantillons", labelArrow: "none" },
      marks: [
        Plot.rectY(effectifs, { x1: (d) => d.femmes - 0.4, x2: (d) => d.femmes + 0.4, y: "n", fill: (d) => (dedans(d.femmes) ? t.series[0] : t.deemph), ry2: 2 }),
        baseline(t),
        Plot.ruleX(reperes, { x: "x", stroke: t.muted }),
        Plot.text(reperes, { x: "x", frameAnchor: "top", dy: -14, text: "label", fill: t.ink2 }),
        Plot.tip(effectifs, Plot.pointerX({
          x: "femmes", y: "n",
          channels: { Femmes: "femmes", Échantillons: "n" },
          format: { x: false, y: false, Femmes: true, Échantillons: true },
        })),
      ],
    })));
    table($("details", fig), [
      { label: "Femmes", value: (d) => fmt.int(d.femmes) },
      { label: "Échantillons", value: (d) => fmt.int(d.n) },
      { label: "À moins d'un écart-type", value: (d) => (dedans(d.femmes) ? "oui" : "non") },
    ], effectifs);
  }

  // --- Niveau de confiance (interactif) ---------------------------------
  {
    const fig = $("#fig-confiance");
    const input = $("input", fig);
    const xs = linspace(-4, 4, 321);
    let conf = 0.95;
    const chart = mount($(".plot", fig), (t, width) => {
      const z = quantile(0.5 + conf / 2);
      const zone = linspace(-z, z, 161).map((x) => ({ x, y: pdf(x) }));
      const bornes = [{ x: -z, label: `−${fmt.num(z, 2)} σ` }, { x: z, label: `+${fmt.num(z, 2)} σ` }];
      const lignes = xs.map((x) => ({ x, y: pdf(x), part: 2 * cdf(Math.abs(x)) - 1 }));
      return Plot.plot(frame(t, width, {
        x: { ...sigmaAxis(), domain: [-4, 4] },
        y: { ...densityAxis, domain: [0, 0.42] },
        marks: [
          Plot.areaY(zone, { x: "x", y: "y", fill: t.series[0], fillOpacity: 0.16 }),
          baseline(t),
          Plot.line(lignes, { x: "x", y: "y", stroke: t.series[0], strokeWidth: 2 }),
          Plot.ruleX(bornes, { x: "x", stroke: t.muted }),
          Plot.text(bornes, { x: "x", frameAnchor: "top", dy: -14, text: "label", fill: t.ink2 }),
          Plot.text([0], { x: 0, y: 0.12, text: () => fmt.pct(conf, conf > 0.99 ? 1 : 0), fill: t.ink, fontSize: 18, fontWeight: 600 }),
          Plot.ruleX(lignes, Plot.pointerX({ x: "x", stroke: t.muted })),
          Plot.tip(lignes, Plot.pointerX({
            x: "x", y: "y",
            channels: { "Part des tirages à ± cette distance": "part" },
            format: { x: (v) => fmt.sigma(v), y: (v) => fmt.num(v, 3), "Part des tirages à ± cette distance": (v) => fmt.pct(v, 1) },
          })),
        ],
      }));
    });
    presets(fig, input, (v) => {
      conf = v / 100;
      $("[data-out=conf]", fig).textContent = fmt.pct(conf, conf > 0.99 || v % 1 ? 1 : 0);
      $("[data-out=z]", fig).textContent = `± ${fmt.sigma(quantile(0.5 + conf / 2))}`;
      chart.redraw();
    });
    const niveaux = [0.5, 0.6827, 0.8, 0.9, 0.95, 0.99, 0.999];
    table($("details", fig), [
      { label: "Niveau de confiance", value: (c) => fmt.pct(c, 2) },
      { label: "Marge (en σ)", value: (c) => fmt.num(quantile(0.5 + c / 2), 3) },
    ], niveaux);
  }

  // --- Valeur réelle / valeur observée ----------------------------------
  {
    const fig = $("#fig-reelle-estimee");
    mount($(".plot", fig), (t, width) => {
      const h = pdf(2, 0);
      const liaison = [{ x: 0, y: h }, { x: 2, y: h }];
      // Pics des deux distributions et liaison entre les points de même densité.
      return twoNormals(t, width, {
        d: 2, bands: false, xmin: -3, xmax: 5,
        points: [{ x: 2, m: 0, serie: 1 }, { x: 0, m: 2, serie: 0 }],
        extra: [
          Plot.ruleX([{ x: 0, c: t.series[0] }, { x: 2, c: t.series[1] }], { x: "x", y1: 0, y2: pdf(0), stroke: "c", strokeWidth: 1.5 }),
          Plot.line(liaison, { x: "x", y: "y", stroke: t.muted, strokeWidth: 1 }),
          Plot.text([{ x: 1, y: h }], { x: "x", y: "y", dy: -10, text: () => "même densité", fill: t.ink2 }),
        ],
      });
    });
    table($("details", fig), [
      { label: "Position", value: (r) => r.label },
      { label: "Densité, valeur réelle (centrée en 0)", value: (r) => fmt.num(pdf(r.x, 0), 3) },
      { label: "Densité, valeur observée (centrée en 2)", value: (r) => fmt.num(pdf(r.x, 2), 3) },
    ], [{ x: 0, label: "0 σ (valeur réelle)" }, { x: 2, label: "2 σ (valeur observée)" }]);
  }

  // --- Deux valeurs observées (interactif) ------------------------------
  {
    const fig = $("#fig-deux-valeurs");
    const input = $("input", fig);
    let d = 3.92;
    const chart = mount($(".plot", fig), (t, width) => twoNormals(t, width, { d, xmin: -4, xmax: 10 }));
    presets(fig, input, (v) => {
      d = v;
      $("[data-out=d]", fig).textContent = fmt.sigma(d);
      $("[data-out=p]", fig).textContent = fmt.pct(pInversion(d), 1);
      $("[data-out=marges]", fig).textContent = margesLabel(d);
      chart.redraw();
    });
    table($("details", fig), [
      { label: "Distance", value: (v) => fmt.sigma(v, 3) },
      { label: "Probabilité d'inversion", value: (v) => fmt.pct(pInversion(v), 2) },
      { label: "Marges à 95 %", value: margesLabel },
    ], [0, 1, 2, 2.328, 3, 3.92, 4.5, 5, 6]);
  }

  // --- Queues de 2,5 % --------------------------------------------------
  {
    const fig = $("#fig-queues");
    mount($(".plot", fig), (t, width) => twoNormals(t, width, { d: 7, tails: true, xmin: -4, xmax: 11 }));
    table($("details", fig), [
      { label: "Zone", value: (r) => r.zone },
      { label: "Probabilité", value: (r) => r.p },
    ], [
      { zone: "Première valeur au-delà de +1,96 σ", p: "2,5 % (1/40)" },
      { zone: "Seconde valeur en deçà de −1,96 σ", p: "2,5 % (1/40)" },
      { zone: "Les deux à la fois", p: "0,0625 % (1/1600)" },
    ]);
  }

  // --- Marges jointives avec inversion ----------------------------------
  {
    const fig = $("#fig-contact");
    const d = 2 * Z95;
    const reelles = [{ x: 1.75, m: 0, serie: 0, nom: "Première valeur" }, { x: 1.5, m: d, serie: 1, nom: "Seconde valeur" }];
    mount($(".plot", fig), (t, width) => twoNormals(t, width, { d, points: reelles, xmin: -4, xmax: d + 4 }));
    table($("details", fig), [
      { label: "Valeur", value: (r) => r.nom },
      { label: "Observée", value: (r) => fmt.sigma(r.m) },
      { label: "Réelle", value: (r) => fmt.sigma(r.x) },
      { label: "Dans sa marge à 95 %", value: (r) => (Math.abs(r.x - r.m) <= Z95 ? "oui" : "non") },
    ], reelles);
  }

  // --- Probabilité d'inversion selon la distance ------------------------
  {
    const fig = $("#fig-inversion");
    const { courbe, monte_carlo, seuil } = DATA.inversion;
    mount($(".plot", fig), (t, width) => Plot.plot(frame(t, width, {
      x: { label: "Distance entre les deux valeurs observées (en σ)", labelAnchor: "center", labelArrow: "none", tickFormat: fmt.tick },
      y: { label: "Probabilité d'inversion", labelArrow: "none", domain: [0, 0.5], tickFormat: (v) => fmt.pct(v, 0) },
      marks: [
        baseline(t),
        Plot.ruleY([0.05], { stroke: t.muted }),
        Plot.ruleX([seuil], { stroke: t.muted }),
        Plot.text([seuil], { x: seuil, y: 0.45, dx: 6, textAnchor: "start", text: () => `${fmt.num(seuil, 3)} σ = 2 × 1,164 σ`, fill: t.ink2 }),
        Plot.text([0.05], { x: 5, y: 0.05, dy: -8, textAnchor: "end", text: () => "5 %", fill: t.ink2 }),
        Plot.line(courbe, { x: "d", y: "p", stroke: t.series[0], strokeWidth: 2 }),
        Plot.dot(monte_carlo, { x: "d", y: "p", r: 4.5, fill: t.series[1], stroke: t.surface, strokeWidth: 2 }),
        Plot.ruleX(courbe, Plot.pointerX({ x: "d", stroke: t.muted })),
        Plot.tip(courbe, Plot.pointerX({
          x: "d", y: "p",
          format: { x: (v) => fmt.sigma(v), y: (v) => fmt.pct(v, 2) },
        })),
      ],
    })));
    const mc = new Map(monte_carlo.map((r) => [r.d, r.p]));
    table($("details", fig), [
      { label: "Distance", value: (r) => fmt.sigma(r.d) },
      { label: "Calcul exact", value: (r) => fmt.pct(r.p, 2) },
      { label: "Monte-Carlo", value: (r) => (mc.has(r.d) ? fmt.pct(mc.get(r.d), 2) : "—") },
    ], courbe.filter((_, i) => i % 5 === 0));
  }

  // --- Distance idéale --------------------------------------------------
  {
    const fig = $("#fig-distance-ideale");
    const d = DATA.inversion.seuil;
    mount($(".plot", fig), (t, width) => twoNormals(t, width, { d, xmin: -4, xmax: d + 4 }));
    table($("details", fig), [
      { label: "Valeur", value: (r) => r.nom },
      { label: "Position", value: (r) => fmt.sigma(r.m, 3) },
      { label: "Marge à 95 %", value: (r) => `${fmt.num(r.m - Z95, 2)} à ${fmt.num(r.m + Z95, 2)} σ` },
    ], [{ nom: "Première valeur", m: 0 }, { nom: "Seconde valeur", m: d }]);
  }

  // --- Correction selon n et p -----------------------------------------
  function multiLines(figId, rows) {
    const fig = $(figId);
    const series = [...new Set(rows.map((r) => r.serie))];
    const wide = d3.groups(rows, (r) => r.coeff).map(([coeff, rs]) => ({ coeff, ...Object.fromEntries(rs.map((r) => [r.serie, r.p])) }));
    const pctFormats = Object.fromEntries(series.map((s) => [s, (v) => fmt.pct(v, 2)]));
    mount($(".plot", fig), (t, width) => Plot.plot(frame(t, width, {
      x: { label: "Coefficient (en σ, de chaque côté)", labelAnchor: "center", labelArrow: "none", tickFormat: fmt.tick },
      y: { label: "2 × probabilité d'inversion", labelArrow: "none", domain: [0, 0.3], tickFormat: (v) => fmt.pct(v, 0) },
      color: { domain: series, range: t.series },
      marks: [
        baseline(t),
        Plot.ruleX([1.164], { stroke: t.muted }),
        Plot.text([1.164], { x: 1.164, y: 0.28, dx: 6, textAnchor: "start", text: () => "1,164 σ", fill: t.ink2 }),
        Plot.line(rows, { x: "coeff", y: "p", z: "serie", stroke: "serie", strokeWidth: 2 }),
        Plot.ruleX(wide, Plot.pointerX({ x: "coeff", stroke: t.muted })),
        Plot.tip(wide, Plot.pointerX({
          x: "coeff", y: (r) => r[series[0]],
          channels: Object.fromEntries(series.map((s) => [s, s])),
          format: { x: (v) => `coefficient ${fmt.num(v, 2)}`, y: false, ...pctFormats },
        })),
      ],
    })));
    table($("details", fig), [
      { label: "Coefficient", value: (r) => fmt.num(r.coeff, 2) },
      ...series.map((s) => ({ label: s, value: (r) => fmt.pct(r[s], 2) })),
    ], wide.filter((_, i) => i % 5 === 0));
  }
  multiLines("#fig-selon-n", DATA.correction.selon_n);
  multiLines("#fig-selon-p", DATA.correction.selon_p);

  // --- Formules ---------------------------------------------------------
  addEventListener("load", () => {
    window.renderMathInElement?.(document.body, {
      delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    });
  });
})();
