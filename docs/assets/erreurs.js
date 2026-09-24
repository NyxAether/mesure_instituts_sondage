// Graphiques de la page « Erreur empirique des sondages ».
// Données : docs/data/erreurs.js (généré par analyses/erreurs.py), un jeu par périmètre
// (tous pays / France) ; le filtre de la barre de navigation bascule tous les graphiques.
(function () {
  const { fmt, mount, frame, table, bind } = Charts;
  // La page peut désigner un autre jeu de données (variante « sondages individuels »).
  const ALL = window.DATA[document.body.dataset.donnees || "erreurs"];
  const $ = (sel, root = document) => root.querySelector(sel);

  let D = ALL.tous;
  /** Fonctions appelées à chaque changement de périmètre. */
  const onScope = [];

  Charts.initThemeToggle();

  const formats = {
    "source.lignes": fmt.int,
    "source.sondages": fmt.int,
    "nuage.nb_lignes": fmt.int,
    "nuage.nb_sondages": fmt.int,
    "nuage.part_hors_marge": (v) => fmt.pct(v, 0),
    "surfaces.ratio_min": (v) => `${fmt.num(v, 1)} ×`,
    "surfaces.ratio_max": (v) => `${fmt.num(v, 1)} ×`,
    "equivalents.nb_sondages": fmt.int,
    "equivalents.nb_proches": fmt.int,
    "equivalents.median_reel": fmt.int,
    "equivalents.medianes.optimal_kl": fmt.int,
    "mimetisme.nb_elections": fmt.int,
    "mimetisme.part_consensus": (v) => fmt.pct(v, 0),
    "mimetisme.part_resserrement": (v) => fmt.pct(v, 0),
    "mimetisme.consensus_median": (v) => fmt.num(v, 2),
    "mimetisme.consensus_hasard_median": (v) => fmt.num(v, 2),
    "mimetisme.resserrement_median": (v) => fmt.num(v, 2),
  };

  // Valeur absente (ex. surface impossible à lisser) : tiret plutôt que « NaN ».
  const sansVide = Object.fromEntries(Object.entries(formats).map(([k, f]) => [k, (v) => (v == null ? "—" : f(v))]));

  const pts = (v, digits = 1) => `${fmt.num(v * 100, digits)} pt`;
  const logN = { type: "log", label: "Taille d'échantillon (n)", labelAnchor: "center", labelArrow: "none", tickFormat: fmt.int, ticks: 5 };
  const baseline = (t) => Plot.ruleY([0], { stroke: t.axis });

  /** Boutons radio (.presets) : un seul actif, `onChange(valeur)` à chaque choix. */
  function radio(group, onChange) {
    const buttons = [...group.querySelectorAll("button")];
    const select = (value) => {
      for (const b of buttons) b.setAttribute("aria-pressed", String(b.dataset.value === value));
      onChange(value);
    };
    for (const b of buttons) b.addEventListener("click", () => select(b.dataset.value));
    select((buttons.find((b) => b.getAttribute("aria-pressed") === "true") ?? buttons[0]).dataset.value);
  }

  /** Message à la place d'un graphique vide (tranches toutes sous le seuil d'effectif). */
  function vide(texte) {
    const p = document.createElement("p");
    p.className = "chart-empty";
    p.textContent = texte;
    return p;
  }

  // --- Nuage des écarts --------------------------------------------------
  {
    const fig = $("#fig-nuage");
    const chart = mount($(".plot", fig), (t, width) => {
      const { points, marge } = D.nuage;
      return Plot.plot(frame(t, width, {
        height: width < 520 ? 300 : 380,
        x: logN,
        y: { label: "Écart (points de %)", labelArrow: "none", domain: [-0.15, 0.15], tickFormat: (v) => fmt.num(v * 100, 0) },
        marks: [
          Plot.areaY(marge, { x: "n", y1: (d) => -d.m, y2: "m", fill: t.series[0], fillOpacity: 0.16 }),
          Plot.line(marge, { x: "n", y: "m", stroke: t.series[0], strokeWidth: 1.5 }),
          Plot.line(marge, { x: "n", y: (d) => -d.m, stroke: t.series[0], strokeWidth: 1.5 }),
          baseline(t),
          Plot.dot(points, { x: "n", y: "residu", r: 2.5, fill: t.series[1], fillOpacity: 0.55 }),
          Plot.tip(points, Plot.pointer({
            x: "n", y: "residu",
            channels: { pays: "pays", année: "annee", sondage: "poll", résultat: "vote" },
            format: { x: (v) => `${fmt.int(v)} sondés`, y: (v) => pts(v), sondage: (v) => fmt.pct(v), résultat: (v) => fmt.pct(v), année: (v) => String(v) },
          })),
        ],
      }));
    });
    onScope.push(() => {
      chart.redraw();
      table($("details", fig), [
        { label: "Pays", value: (r) => r.pays },
        { label: "Année", value: (r) => String(r.annee) },
        { label: "n", value: (r) => fmt.int(r.n) },
        { label: "Sondage", value: (r) => fmt.pct(r.poll) },
        { label: "Résultat", value: (r) => fmt.pct(r.vote) },
        { label: "Écart", value: (r) => pts(r.residu) },
      ], [...D.nuage.points].sort((a, b) => a.pays.localeCompare(b.pays, "fr") || a.annee - b.annee));
    });
  }

  // --- Erreur moyenne par tranche de taille -------------------------------
  {
    const fig = $("#fig-taille");
    const chart = mount($(".plot", fig), (t, width) => {
      const rows = D.par_taille;
      const long = rows.flatMap((r) => [
        { ...r, serie: "Erreur observée", e: r.obs },
        { ...r, serie: "Erreur attendue", e: r.th },
      ]);
      const [nMin, nMax] = d3.extent(rows, (r) => r.n);
      return Plot.plot(frame(t, width, {
        x: { ...logN, domain: [nMin / 1.18, nMax * 1.18] },
        y: { label: "Erreur moyenne (points de %)", labelArrow: "none", domain: [0, d3.max(long, (r) => r.e) * 1.12], tickFormat: (v) => fmt.num(v * 100, 1) },
        color: { domain: ["Erreur observée", "Erreur attendue"], range: [t.series[0], t.series[1]] },
        marks: [
          baseline(t),
          Plot.line(long, { x: "n", y: "e", z: "serie", stroke: "serie", strokeWidth: 2 }),
          Plot.dot(long, { x: "n", y: "e", fill: "serie", r: 4, stroke: t.surface, strokeWidth: 2 }),
          Plot.ruleX(rows, Plot.pointerX({ x: "n", stroke: t.muted })),
          Plot.tip(rows, Plot.pointerX({
            x: "n", y: "obs",
            channels: { observée: "obs", attendue: "th", "taille équivalente": "n_equiv", lignes: "lignes" },
            format: {
              x: (v) => `n médian ${fmt.int(v)}`, y: false,
              observée: (v) => pts(v, 2), attendue: (v) => pts(v, 2),
              "taille équivalente": (v) => fmt.int(Math.round(v)), lignes: fmt.int,
            },
          })),
        ],
      }));
    });
    onScope.push(() => {
      chart.redraw();
      table($("details", fig), [
        { label: "Tranche de n", value: (r) => `${fmt.int(r.n_min)}–${fmt.int(r.n_max)}` },
        { label: "Lignes", value: (r) => fmt.int(r.lignes) },
        { label: "Erreur observée", value: (r) => pts(r.obs, 2) },
        { label: "Erreur attendue", value: (r) => pts(r.th, 2) },
        { label: "Hors marge 95 %", value: (r) => fmt.pct(r.hors_marge, 0) },
        { label: "Taille équivalente", value: (r) => fmt.int(Math.round(r.n_equiv)) },
      ], D.par_taille);
    });
  }

  // --- Surface 3D (p, n) ---------------------------------------------------
  // Plotly plutôt qu'Observable Plot : la lecture de deux surfaces qui se croisent demande le relief.
  {
    const fig = $("#fig-carte");
    const mono = getComputedStyle(document.documentElement).getPropertyValue("--font-mono").trim();
    const nTicks = [500, 1000, 2000, 5000, 10000];
    // Axe vertical borné pour que les rares écarts extrêmes n'écrasent pas les surfaces.
    const Z_OBS_MAX = 8;
    const horsCadre = $("[data-out=hors-cadre]", fig);
    let vue = "obs";
    let camera = { eye: { x: 1.5, y: -1.5, z: 0.7 }, center: { x: 0, y: 0, z: -0.12 } };
    let gd = null;
    let S;

    /** Grandeurs dérivées du périmètre courant (en points de %). */
    function derive() {
      const src = D.surfaces;
      if (!src) return null;
      const enPts = (m) => m.map((ligne) => ligne.map((v) => v * 100));
      const obs = enPts(src.obs);
      const th = enPts(src.th);
      const points = D.nuage.points.filter((d) => d.n >= 500 && d.n <= 10000 && d.vote <= 0.6);
      return {
        src, obs, th, points,
        P: src.p.map((v) => v * 100),
        ratio: src.obs.map((ligne, i) => ligne.map((v, j) => v / src.th[i][j])),
        zMax: d3.max([...obs.flat(), ...th.flat()]),
        horsCadre: points.filter((d) => Math.abs(d.residu) * 100 > Z_OBS_MAX).length,
      };
    }
    S = derive();

    function traces(t) {
      const low = d3.interpolateRgb(t.surface, t.series[1])(0.15);
      const rampe = [[0, low], [1, t.series[1]]];
      const barre = (titre) => ({ title: { text: titre, side: "right" }, thickness: 10, len: 0.6, outlinewidth: 0, tickfont: { color: t.ink2 } });
      const survol = "p %{x:.0f} %<br>n %{y:,.0f}<br>";
      const { P, src } = S;
      if (vue === "ratio") {
        return [{
          type: "surface", x: P, y: src.n, z: S.ratio, colorscale: rampe, cmin: 1, colorbar: barre("× attendue"),
          hovertemplate: `${survol}rapport %{z:.2f} ×<extra></extra>`,
        }];
      }
      const out = [{
        type: "surface", x: P, y: src.n, z: S.obs, colorscale: rampe, cmin: 0, cmax: S.zMax, colorbar: barre("pt"),
        hovertemplate: `${survol}observée %{z:.2f} pt<extra></extra>`,
      }];
      if (vue === "comp") {
        out.push({
          type: "surface", x: P, y: src.n, z: S.th, colorscale: [[0, t.series[0]], [1, t.series[0]]], showscale: false, opacity: 0.55,
          hovertemplate: `${survol}attendue %{z:.2f} pt<extra></extra>`,
        });
      } else {
        out.push({
          type: "scatter3d", mode: "markers", x: S.points.map((d) => d.vote * 100), y: S.points.map((d) => d.n), z: S.points.map((d) => Math.abs(d.residu) * 100),
          marker: { size: 2, color: t.ink, opacity: 0.5 },
          customdata: S.points.map((d) => [d.pays, d.annee]),
          hovertemplate: `%{customdata[0]} %{customdata[1]}<br>${survol}écart %{z:.1f} pt<extra></extra>`,
        });
      }
      return out;
    }

    function layout(t, width) {
      const axe = (titre, extra = {}) => ({
        title: { text: titre }, color: t.ink2, gridcolor: t.grid, zerolinecolor: t.axis, linecolor: t.axis,
        showbackground: false, showspikes: false, ...extra,
      });
      return {
        width, height: width < 520 ? 360 : 460,
        margin: { l: 0, r: 0, t: 0, b: 0 },
        paper_bgcolor: "rgba(0,0,0,0)",
        font: { family: mono, size: 11, color: t.ink2 },
        separators: ", ",
        showlegend: false,
        hoverlabel: { bgcolor: t.surface, bordercolor: t.axis, font: { family: mono, color: t.ink } },
        scene: {
          camera, dragmode: "turntable", aspectmode: "manual", aspectratio: { x: 1.15, y: 1.15, z: 0.75 },
          xaxis: axe("proportion p (%)", { ticksuffix: " %" }),
          yaxis: axe("taille n", { type: "log", tickvals: nTicks, ticktext: nTicks.map(fmt.int) }),
          zaxis: axe(vue === "ratio" ? "observée / attendue" : "erreur (pt)", vue === "obs" ? { range: [0, Z_OBS_MAX] } : { rangemode: "tozero" }),
        },
      };
    }

    const chart = mount($(".plot", fig), (t, width) => {
      if (gd) Plotly.purge(gd);
      if (!S) { gd = null; return vide("Trop peu de sondages pour lisser une surface sur ce périmètre."); }
      const el = document.createElement("div");
      gd = el;
      // Plotly mesure son conteneur : le tracé attend que le nœud soit inséré par mount().
      requestAnimationFrame(() => {
        if (gd !== el) return;
        Plotly.newPlot(el, traces(t), layout(t, width), { displaylogo: false, scrollZoom: false, modeBarButtonsToRemove: ["toImage", "resetCameraLastSave3d"] });
        el.on("plotly_relayout", (e) => { if (e["scene.camera"]) camera = e["scene.camera"]; });
      });
      return el;
    });

    const legende = [...fig.querySelectorAll(".legend li")];
    radio($(".presets", fig), (value) => {
      vue = value;
      for (const li of legende) li.hidden = !li.dataset.vues.split(" ").includes(vue);
      chart.redraw();
    });

    onScope.push(() => {
      S = derive();
      if (!S) {
        horsCadre.textContent = "surface indisponible";
        chart.redraw();
        table($("details", fig), [{ label: "Données", value: (r) => r }], []);
        return;
      }
      horsCadre.textContent = S.horsCadre === 0 ? "aucun écart ne dépasse 8 pt"
        : `${fmt.int(S.horsCadre)} écart${S.horsCadre > 1 ? "s" : ""} de plus de 8 pt ${S.horsCadre > 1 ? "sont" : "est"} hors cadre`;
      chart.redraw();
      const { src } = S;
      const rows = src.n.flatMap((n, i) => src.p.map((p, j) => ({ p, n, obs: src.obs[i][j], th: src.th[i][j] })))
        .filter((_, k) => k % 4 === 0);
      table($("details", fig), [
        { label: "p", value: (r) => fmt.pct(r.p, 0) },
        { label: "n", value: (r) => fmt.int(Math.round(r.n)) },
        { label: "Erreur observée", value: (r) => pts(r.obs, 2) },
        { label: "Erreur attendue", value: (r) => pts(r.th, 2) },
        { label: "Rapport", value: (r) => `${fmt.num(r.obs / r.th, 2)} ×` },
      ], rows);
    });
  }

  // --- Taille équivalente : médiane glissante ------------------------------
  const { mesures } = ALL.tous.equivalents;
  const cles = Object.keys(mesures);
  {
    const fig = $("#fig-glissante");
    const chart = mount($(".plot", fig), (t, width) => {
      const { glissante } = D.equivalents;
      const long = cles.flatMap((m) => glissante.map((r) => ({ x: r.poll_sample, y: r[m], serie: mesures[m] })));
      const [yMin, yMax] = d3.extent(long, (r) => r.y);
      return Plot.plot(frame(t, width, {
        height: width < 520 ? 300 : 360,
        x: { label: "Taille réelle du sondage", labelAnchor: "center", labelArrow: "none", tickFormat: fmt.int, ticks: 6 },
        y: { type: "log", label: "Taille équivalente", labelArrow: "none", domain: [Math.min(20, yMin * 0.8), Math.max(8000, yMax * 1.25)], tickFormat: fmt.int },
        color: { domain: cles.map((m) => mesures[m]), range: t.series },
        marks: [
          Plot.line(long, { x: "x", y: "y", z: "serie", stroke: "serie", strokeWidth: 2 }),
          Plot.ruleX(glissante, Plot.pointerX({ x: "poll_sample", stroke: t.muted })),
          Plot.tip(glissante, Plot.pointerX({
            x: "poll_sample", y: "optimal_kl",
            channels: Object.fromEntries(cles.map((m) => [mesures[m], m])),
            format: { x: (v) => `taille réelle ≈ ${fmt.int(Math.round(v))}`, y: false, ...Object.fromEntries(cles.map((m) => [mesures[m], (v) => fmt.int(Math.round(v))])) },
          })),
        ],
      }));
    });
    onScope.push(() => {
      chart.redraw();
      table($("details", fig), [
        { label: "Taille réelle", value: (r) => fmt.int(Math.round(r.poll_sample)) },
        ...cles.map((m) => ({ label: mesures[m], value: (r) => fmt.int(Math.round(r[m])) })),
      ], D.equivalents.glissante);
    });
  }

  // --- Boîtes à moustaches par facteur -------------------------------------
  {
    const fig = $("#fig-boites");
    const facteurs = {
      poll_sample: { label: "Taille réelle du sondage", tranches: true },
      year: { label: "Année de l'élection", tranches: true },
      daysbeforeED: { label: "Jours avant l'élection", tranches: true },
      nb_candidates: { label: "Nombre de partis", tranches: false },
    };
    const choix = { fenetre: "14", facteur: "poll_sample", mesure: "optimal_kl" };
    const contexte = $("[data-out=contexte]", fig);
    const lignes = () => D.equivalents.boites.filter((b) => String(b.fenetre) === choix.fenetre && b.facteur === choix.facteur && b.mesure === choix.mesure);
    const texteContexte = () => {
      const n = fmt.int(D.equivalents.effectifs_fenetres[choix.fenetre]);
      const qui = `Sondages réalisés au plus ${choix.fenetre} jours avant le scrutin (${n})`;
      return facteurs[choix.facteur].tranches ? `${qui}, en tranches d'effectifs comparables.` : `${qui}.`;
    };

    const chart = mount($(".plot", fig), (t, width) => {
      const rows = lignes();
      if (!rows.length) return vide(`Aucune tranche n'atteint ${D.equivalents.effectif_min} sondages pour ce choix.`);
      return Plot.plot(frame(t, width, {
        marginLeft: 56,
        x: { domain: rows.map((r) => r.tranche), label: facteurs[choix.facteur].label, labelAnchor: "center", labelArrow: "none", tickRotate: rows.length > 6 && width < 720 ? -35 : 0, padding: 0.35 },
        y: { label: `Taille équivalente — ${mesures[choix.mesure]}`, labelArrow: "none", tickFormat: fmt.int, nice: true },
        marks: [
          baseline(t),
          Plot.ruleX(rows, { x: "tranche", y1: "bas", y2: "haut", stroke: t.muted }),
          Plot.barY(rows, { x: "tranche", y1: "q1", y2: "q3", fill: t.series[1], fillOpacity: 0.35, stroke: t.series[1], rx: 2 }),
          Plot.tickY(rows, { x: "tranche", y: "med", stroke: t.series[1], strokeWidth: 2.5 }),
          Plot.tip(rows, Plot.pointerX({
            x: "tranche", y: "med",
            channels: { sondages: "effectif", "1er quartile": "q1", "3e quartile": "q3" },
            format: { x: true, y: (v) => `médiane ${fmt.int(Math.round(v))}`, sondages: fmt.int, "1er quartile": (v) => fmt.int(Math.round(v)), "3e quartile": (v) => fmt.int(Math.round(v)) },
          })),
        ],
      }));
    });

    const refresh = () => {
      contexte.textContent = texteContexte();
      chart.redraw();
      table($("details", fig), [
        { label: facteurs[choix.facteur].label, value: (r) => r.tranche },
        { label: "Sondages", value: (r) => fmt.int(r.effectif) },
        { label: "Moustache basse", value: (r) => fmt.int(Math.round(r.bas)) },
        { label: "1er quartile", value: (r) => fmt.int(Math.round(r.q1)) },
        { label: "Médiane", value: (r) => fmt.int(Math.round(r.med)) },
        { label: "3e quartile", value: (r) => fmt.int(Math.round(r.q3)) },
        { label: "Moustache haute", value: (r) => fmt.int(Math.round(r.haut)) },
      ], lignes());
    };
    for (const group of fig.querySelectorAll(".presets")) {
      radio(group, (value) => { choix[group.dataset.group] = value; refresh(); });
    }
    onScope.push(refresh);
  }

  // --- Consensus d'erreur et resserrement ----------------------------------
  {
    const fig = $("#fig-mimetisme");
    const SEUIL = 0.05;
    const categories = ["Compatible avec le hasard", "Resserrement anormal seul", "Consensus anormal", "Consensus et resserrement anormaux"];
    const categorie = (e) => {
      const c = e.rang_consensus < SEUIL, r = e.rang_resserrement < SEUIL;
      return c && r ? categories[3] : c ? categories[2] : r ? categories[1] : categories[0];
    };
    const libelle = (e) => `${e.pays} ${e.annee} · ${e.election === "Presidential" ? "présidentielle" : "législatives"}${e.tour === 2 ? " (2d tour)" : ""}`;
    const rang = (v) => (v < 0.001 ? "< 0,1 %" : fmt.pct(v, 1));

    const chart = mount($(".plot", fig), (t, width) => {
      const M = D.mimetisme;
      if (!M.elections.length) return vide("Aucune élection ne compte assez de sondages pour ce périmètre.");
      // Points compatibles avec le hasard dessinés en premier, sous les autres.
      const rows = M.elections.map((e) => ({ ...e, cat: categorie(e), nom: libelle(e) }))
        .sort((a, b) => categories.indexOf(a.cat) - categories.indexOf(b.cat));
      const [rMin, rMax] = d3.extent(rows, (e) => e.resserrement);
      return Plot.plot(frame(t, width, {
        height: width < 520 ? 320 : 400,
        x: { label: "Consensus d'erreur (0 : erreurs équilibrées, 1 : toutes du même côté)", labelAnchor: "center", labelArrow: "none", domain: [0, 1], tickFormat: fmt.tick },
        y: { type: "log", label: "Resserrement (1 : dispersion du hasard)", labelArrow: "none", domain: [Math.min(0.1, rMin * 0.8), Math.max(10, rMax * 1.2)], tickFormat: fmt.tick },
        color: { domain: categories, range: [t.deemph, t.series[1], t.series[0], t.series[2]] },
        marks: [
          Plot.ruleY([1], { stroke: t.axis }),
          Plot.ruleX([M.consensus_hasard_median], { stroke: t.muted, strokeDasharray: "4 3" }),
          Plot.text([M.consensus_hasard_median], { x: (d) => d, frameAnchor: "top", dy: -14, dx: 4, textAnchor: "start", text: () => "hasard (médiane)", fill: t.ink2 }),
          Plot.dot(rows, { x: "consensus", y: "resserrement", fill: "cat", r: 4.5, stroke: t.surface, strokeWidth: 1.5 }),
          Plot.tip(rows, Plot.pointer({
            x: "consensus", y: "resserrement",
            channels: { élection: "nom", sondages: "sondages", "hasard (consensus)": "consensus_hasard", "rang consensus": "rang_consensus", "rang resserrement": "rang_resserrement" },
            format: {
              élection: true, x: (v) => fmt.num(v, 2), y: (v) => fmt.num(v, 2), sondages: fmt.int,
              "hasard (consensus)": (v) => fmt.num(v, 2), "rang consensus": rang, "rang resserrement": rang,
            },
          })),
        ],
      }));
    });
    onScope.push(() => {
      chart.redraw();
      table($("details", fig), [
        { label: "Élection", value: libelle },
        { label: "Sondages", value: (e) => fmt.int(e.sondages) },
        { label: "Erreur moyenne", value: (e) => pts(e.erreur_moyenne, 2) },
        { label: "Consensus", value: (e) => fmt.num(e.consensus, 2) },
        { label: "Hasard", value: (e) => fmt.num(e.consensus_hasard, 2) },
        { label: "Rang consensus", value: (e) => rang(e.rang_consensus) },
        { label: "Resserrement", value: (e) => fmt.num(e.resserrement, 2) },
        { label: "Rang resserrement", value: (e) => rang(e.rang_resserrement) },
      ], [...D.mimetisme.elections].sort((a, b) => a.pays.localeCompare(b.pays, "fr") || a.annee - b.annee || a.tour - b.tour));
    });
  }

  // --- Filtre de périmètre -------------------------------------------------
  const avertissement = $("#avertissement-france");
  radio($(".scope"), (value) => {
    D = ALL[value];
    document.documentElement.dataset.scope = value;
    avertissement.hidden = value !== "france";
    bind(document, D, sansVide);
    for (const f of onScope) f();
  });

  // --- Formules ---------------------------------------------------------
  addEventListener("load", () => {
    window.renderMathInElement?.(document.body, {
      delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    });
  });
})();
