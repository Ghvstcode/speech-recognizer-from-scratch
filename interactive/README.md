# Interactive pages: the builder's guide

These are the interactive companions to the chapters of *Building a Speech Recognizer from Scratch*. Each page is one static HTML file that loads two local files, `shared.css` and `shared.js`, and nothing else. No CDN, no web fonts, no framework, no build step. A page must work when opened from disk (`file://`) and when served by GitHub Pages.

`sampling.html` is the reference page. When this guide and that file disagree, the file wins; then fix the guide.

| Page | Chapter | Status |
|---|---|---|
| `index.html` | all | done |
| `sampling.html` | 1 | done |
| `windowing.html` | 2-3 | done |
| `dft.html` | 3 | done |
| `mel.html` | 4 | done |
| `dtw.html` | 5 | done |
| `ctc.html` | 8 | done |
| `beam.html` | 10 | done |
| `listen.html` | 11 | done (the one exception to "no build step", see below) |

When your page exists, change its card in `index.html` from `data-status="coming"` to `data-status="ready"`. The card already links to your file name.

## The shape of a page

- **Top bar.** Book title (links to `index.html`), a link to the chapter, the theme toggle.
- **Left column, about 420px.** A mono overline, a large light title, a lede of two or three sentences, then the controls in groups.
- **Right, the stage.** Dot grid. A stepper on top ("Step 2 of 5"), then the figures, each in its own card.
- **Under 800px** it becomes one column in this order: intro, stage, controls. You get that for free from the three sections below; do not reorder them.

## Skeleton to copy

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Mel Filterbank · Building a Speech Recognizer from Scratch</title>
<meta name="description" content="One sentence about the page.">
<link rel="stylesheet" href="shared.css">
<script src="shared.js"></script>  <!-- in <head>, not deferred: it sets the theme before first paint -->
<style>
  /* Page-specific sizes only. If you are styling a button here, stop and use a shared class. */
  #plot { --h: 300px; }
  @media (max-width: 799px) { #plot { --h: 230px; } }
</style>
</head>
<body>

<header class="sr-topbar">
  <a class="sr-brand" href="index.html">Building a Speech Recognizer <span>from Scratch</span></a>
  <nav class="sr-topnav" aria-label="Book">
    <a class="sr-hide-narrow" href="index.html">All interactives</a>
    <a href="../chapters/04-hearing-like-a-human.md">Chapter 4</a>
    <button class="sr-theme-toggle" type="button" data-sr-theme></button>
  </nav>
</header>

<main class="sr-shell">

  <section class="sr-intro">
    <span class="sr-overline">Chapter 4 · Interactive</span>
    <h1 class="sr-title">The Mel Filterbank</h1>
    <p class="sr-lede">Two or three plain sentences. What is on the page and why we care.</p>
  </section>

  <section class="sr-stage" aria-label="Figure">
    <div class="sr-stage-inner">
      <div id="steps"></div>

      <figure class="sr-figure">
        <div class="sr-figure-head">
          <span class="sr-overline">What this plot shows</span>
          <span class="sr-pill" id="plot-pill">40 filters</span>
        </div>
        <div class="sr-figure-body">
          <canvas class="sr-canvas" id="plot" role="img" aria-label="Described in words, updated from JS"></canvas>
        </div>
        <div class="sr-figure-foot">
          <ul class="sr-legend">
            <li><span class="sr-swatch" style="--c: var(--green)"></span>the selected filter</li>
            <li><span class="sr-swatch sr-swatch--thin" style="--c: var(--ink-48)"></span>the others</li>
          </ul>
        </div>
      </figure>

      <div class="sr-stats">
        <div class="sr-stat" id="stat-width">
          <span class="sr-overline">Filter width</span>
          <span class="sr-stat-value">212 Hz</span>
          <span class="sr-stat-sub">from 890 to 1,102 Hz</span>
        </div>
      </div>
    </div>
  </section>

  <section class="sr-controls" aria-label="Controls">
    <div class="sr-control-group">
      <span class="sr-overline">The filterbank</span>
      <div class="sr-field">
        <div class="sr-field-head">
          <label for="n-mels">Number of filters</label>
          <output class="sr-readout" for="n-mels"></output>
        </div>
        <input class="sr-range" type="range" id="n-mels" min="5" max="80" step="1" value="40">
        <p class="sr-hint">Our features use 40.</p>
      </div>
    </div>
  </section>

</main>

<footer class="sr-footer">
  <span>Part of <a href="../chapters/04-hearing-like-a-human.md"><em>Building a Speech Recognizer from Scratch</em></a>, chapter 4.</span>
  <span>The same maths as <a href="../code/src/dsp/mel.rs"><code>mel.rs</code></a>.</span>
</footer>

<script>
(function () {
  'use strict';
  var state = { nMels: 40 };

  var plot = SR.canvas('#plot', function (ctx, w, h, t) {
    var box = SR.plotBox(w, h);
    var x = SR.scale.linear([0, 8000], [box.x, box.x + box.w]);
    var y = SR.scale.linear([0, 1], [box.y + box.h, box.y]);
    SR.axes(ctx, { box: box, x: x, y: y, xTicks: SR.ticks(0, 8000, 8), xFormat: SR.fmt.hzShort,
                   yTicks: [0, 0.5, 1], xLabel: 'frequency (Hz)', yLabel: 'weight' });
    // ...draw with t.ink, t.green, never with a hard-coded colour...
  });

  function update() {
    // 1. recompute from state  2. write readouts  3. ask each canvas to redraw
    plot.redraw();
  }

  var nMels = SR.bindSlider('#n-mels', {
    format: function (v) { return v + ' filters'; },
    onInput: function (v) { state.nMels = v; update(); }
  });

  SR.stepper('#steps', {
    steps: [
      { title: 'A ruler for pitch', body: 'One short paragraph.',
        enter: function (animate) { state.nMels = 10; nMels.set(10); update(); } },
      { title: 'Forty filters', body: 'One short paragraph.',
        enter: function (animate) { state.nMels = 40; nMels.set(40); update(); } }
    ]
  });
})();
</script>
</body>
</html>
```

The pattern in the script is the whole architecture: one `state` object, one `update()` that recomputes, writes readouts and calls `redraw()`, controls that write to `state` and call `update()`, and steps that write to `state`, push it into the controls with `.set()` and call `update()`.

## Tokens

Use the CSS variables. Never type a hex colour in a page.

| Variable | Light | Dark | Use |
|---|---|---|---|
| `--paper` | `#F5F3EF` | `#191615` | page background |
| `--panel` | `#FBFAF7` | `#201D1B` | cards and figures |
| `--ink` | `#191615` | `#F5F3EF` | text, primary strokes |
| `--ink-70` | 70% | 72% | body text beside a figure |
| `--ink-60` | 60% | 62% | small text that must stay readable: overlines, hints |
| `--ink-48` | 48% | 48% | axis ticks, secondary strokes |
| `--ink-30` `--ink-15` `--ink-10` `--ink-7` `--ink-4` | | | axes, card borders, dot grid, hairlines, washes |
| `--green` | `#047857` | `#34D399` | the thing the figure is about; correct |
| `--amber` | `#B45309` | `#FBBF24` | second series; "watch this" |
| `--red` | `#B42318` | `#F87171` | wrong answers and errors only |
| `--green-wash` `--amber-wash` `--red-wash` | | | 10-12% fills behind an accent |
| `--heat-0` `--heat-1` `--heat-2` | paper, `#9FB8A9`, `#04543E` | `#191615`, `#1F6B55`, `#A7F3D0` | heatmap ramp, low to high |
| `--sans` `--mono` | | | Geist/Inter/system; Geist Mono/ui-monospace |

In canvas code the same values arrive as the fourth argument of your draw function: `t.ink`, `t.ink48`, `t.green`, `t.amberWash`, `t.mono`, and so on (`SR.tokens()` anywhere else).

## CSS components

**Shell**

| Class | What it is |
|---|---|
| `sr-topbar`, `sr-brand`, `sr-topnav`, `sr-theme-toggle` | The top bar. Put `data-sr-theme` on the toggle button and `shared.js` does the rest. |
| `sr-shell` | The grid. Its three children are `sr-intro`, `sr-stage`, `sr-controls`. |
| `sr-stage`, `sr-stage-inner` | The dot-grid side and its column of figures (14px gap). Add `sr-stage--sticky` to the stage only if everything in it fits in a short window. |
| `sr-footer` | The last line of the page. |
| `sr-hide-narrow` | Hidden under 560px. For the "All interactives" link. |
| `sr-visually-hidden` | For screen readers only. |

**Type**

| Class | What it is |
|---|---|
| `sr-overline` | Mono, 10.5px, uppercase, spaced. Group labels, figure titles, stat names. |
| `sr-title` | The page title: large, weight 300. One per page, on the `h1`. |
| `sr-lede` | The paragraph under the title. |
| `sr-h2`, `sr-body`, `sr-small`, `sr-hint` | Section title, body text, 13px text, 12.5px help text under a control. |
| `sr-mono` | Mono with tabular numbers. |
| `sr-green`, `sr-amber`, `sr-red` | Colour a word to match its line in the figure. Use rarely. |

**Controls**

| Class | Markup |
|---|---|
| `sr-control-group` | A group of fields with an `sr-overline` on top. Groups are separated by a hairline. |
| `sr-field`, `sr-field-head`, `sr-readout` | Label on the left, live value in mono on the right, the input underneath. See the skeleton. |
| `sr-range` | `<input type="range" class="sr-range">`. Always bind it with `SR.bindSlider` (that is what fills the track). |
| `sr-btn` | Button. Variants: `sr-btn--primary` (filled, one per group at most), `sr-btn--sm`, `sr-btn--pill`. Row: `sr-btn-row`. |
| `sr-chip`, `sr-chip-row` | Small mono preset buttons under a slider. `aria-pressed="true"` marks the active one. |
| `sr-seg` | Segmented control made of radio inputs: `<div class="sr-seg" role="radiogroup" aria-labelledby="..."><label><input type="radio" name="mode" value="a" checked><span>A</span></label>...</div>`. Add `sr-seg--block` to stretch it. Keep labels to one or two words. |
| `sr-check`, `sr-check-row` | `<label class="sr-check"><input type="checkbox" id="x" checked>Label</label>` |
| `sr-stepper` | Built by `SR.stepper`; you only supply an empty `<div id="steps">`. Set `--sr-step-lines` on it if your longest step text needs more than three lines. |

**Surfaces**

| Class | What it is |
|---|---|
| `sr-figure`, `sr-figure-head`, `sr-figure-body`, `sr-figure-foot` | A card holding one canvas or SVG. Head: overline on the left, optional pill on the right. Foot: legend. |
| `sr-canvas` | A canvas that fills its card. Height comes from `--h` (default 280px); set it per canvas in your page `<style>`. |
| `sr-card` | A plain solid card (1px border, 8px radius, panel fill). |
| `sr-group`, `sr-group-label` | Dashed rounded box with a mono label sitting on its border: "these belong together". `sr-group--panel` fills it. |
| `sr-pill` | Small mono label. Variants `sr-pill--green`, `--amber`, `--red`, `--dashed`. |
| `sr-legend`, `sr-swatch` | Legend row. Swatch colour via `style="--c: var(--green)"`. Variants: `sr-swatch--thin`, `--dashed`, `--dot`, `--box`, `--ramp` (the heatmap ramp). |
| `sr-stats`, `sr-stat`, `sr-stat-value`, `sr-stat-sub` | A row of live numbers. `data-tone="green|amber|red|muted"` on the `sr-stat` colours the value. |
| `sr-values` | One line of raw numbers in mono, cut off with an ellipsis. |
| `sr-kv` | A `<dl>` glossary: term on the left, explanation on the right. |
| `sr-note` | An aside in the text column. |
| `sr-card-grid`, `sr-index-card`, `sr-thumb` | Only used by `index.html`. |

## JS helpers (`window.SR`)

**Theme and tokens**

| Helper | Does |
|---|---|
| `SR.theme.get()` / `.set('dark' \| 'light' \| null)` / `.toggle()` / `.onChange(fn)` | The theme. You rarely call these: the toggle button is wired automatically and canvases made with `SR.canvas` redraw themselves. |
| `SR.tokens()` | The tokens as strings for canvas drawing: `{ paper, panel, ink, ink70, ink60, ink48, ink30, ink15, ink10, ink7, ink4, green, amber, red, greenWash, amberWash, redWash, sans, mono, dark }`. Cached per theme. |

**Canvas**

| Helper | Does |
|---|---|
| `SR.canvas(target, draw)` | Hi-DPI setup, resize handling, theme redraw. `draw(ctx, w, h, t)` works in CSS pixels on a cleared canvas. Returns `{ redraw(), flush(), width, height, point(evt), canvas, ctx }`. `redraw()` is cheap to call many times; it draws once per frame. |
| `SR.crisp(v)` | Rounds to the middle of a pixel. Use it for every 1px horizontal or vertical line. |
| `SR.plotBox(w, h, {l, r, t, b})` | The plot rectangle inside the canvas. Defaults leave room for tick labels: left 44, right 12, top 18, bottom 30. |
| `SR.scale.linear(domain, range)`, `SR.scale.log(domain, range)` | `x(value)` to pixels, `x.invert(px)` back. |
| `SR.ticks(min, max, count)`, `SR.logTicks(min, max)` | Round tick values; 1-2-5 per decade for log axes. |
| `SR.axes(ctx, { box, x, y, xTicks, yTicks, xFormat, yFormat, xLabel, yLabel, grid, zero })` | House-style axes: hairline grid, mono tick labels, uppercase axis labels (y label top left, x label bottom right). Put the unit in the label. |
| `SR.line(ctx, n, i => [px, py], { color, width, dash, alpha })` | A polyline. Data lines are 1.5 to 2 wide. |
| `SR.dot(ctx, x, y, r, color)` | A dot with a panel-coloured ring so it reads on top of lines. |
| `SR.pill(ctx, text, x, y, { align, color, min, max })` | A mono label in a pill, drawn on the canvas. `min`/`max` keep it inside the plot. |
| `SR.spacedText(ctx, text, x, y)` | `fillText` with overline letter-spacing. |
| `SR.font.mono(px)`, `SR.font.sans(px, weight)` | Font strings for `ctx.font`. |
| `SR.roundRect(ctx, x, y, w, h, r)` | Adds a rounded rectangle to the current path. |
| `SR.heat(v)`, `SR.heatCss(v)` | `v` in 0..1 to `[r, g, b]` or a CSS colour on the single-hue ramp of the current theme. |
| `SR.heatmap(ctx, data, rows, cols, box, { min, max, flipY })` | Paints a row-major grid (like `Mat`) as crisp cells. For spectrograms, cost grids, lattices. |

**DOM and controls**

| Helper | Does |
|---|---|
| `SR.el(tag, attrs, children)`, `SR.svg(tag, attrs, children)` | Create HTML / SVG elements. `attrs` may hold `text`, `html` and `onclick`-style handlers. |
| `SR.bindSlider(target, { format, onInput, map, value })` | Slider, readout and track fill in one. Returns `{ value, set(v), refresh() }`. `set` does not call `onInput`. |
| `SR.logMap(min, max, { steps, sig, snap })` | Pass as `map` for a logarithmic slider. `snap: [16000]` makes round values easy to hit. |
| `SR.bindRadios(name, onChange)` | For an `sr-seg`. Returns `{ value, set(v) }`. |
| `SR.bindCheck(target, onChange)` | For an `sr-check`. Returns `{ value, set(v) }`. |
| `SR.stepper(target, { steps: [{ title, body, enter(animate) }], onChange, label })` | The walkthrough. Returns `{ go(i), next(), prev(), index, length }`. |
| `SR.tween(ms, onUpdate, onDone)` | Calls `onUpdate(t)` with eased `t` from 0 to 1. Jumps to 1 under reduced motion. Returns a cancel function. |
| `SR.reducedMotion()` | True if the reader asked for less motion. |
| `SR.fmt.int`, `.fixed`, `.sig`, `.hz`, `.hzShort` | `16,000`, `0.50`, `3.05×10⁻⁵`, `16,000 Hz`, `16k`. Use these so every page formats numbers the same way. |
| `SR.clamp`, `SR.lerp`, `SR.parseColor` | Small things. |

**Audio**

| Helper | Does |
|---|---|
| `SR.audio.button(target, () => ({ samples, sampleRate }), { gain })` | Turns a button into play/stop. This is the one to use: it satisfies "sound only on a click, with a visible stop". |
| `SR.audio.play(samples, sampleRate, { gain, onEnd })`, `SR.audio.stop()` | Lower level. Any sample rate works; we resample with our own sinc filter. One sound at a time. |

**Maths (`SR.dsp`)**: mirrors `code/src/dsp/` and `wav.rs`. Use these instead of writing your own, and if the Rust changes, change them here.

| Helper | Does |
|---|---|
| `hann(n)`, `hamming(n)`, `hannSymmetric(n)`, `hammingSymmetric(n)`, `rectangular(n)` | Windows. The plain names are the periodic versions, as in `window.rs`. |
| `dft(re, im?)` | The O(n²) definition. Returns `{ re, im }`. |
| `fft(samples, nFft?)`, `ifft({ re, im })`, `fftInPlace(re, im, inverse)` | Radix-2 FFT. Length must be a power of two; `fft` zero-pads. |
| `powerSpectrum(samples, nFft?)` | `|X[k]|²` for `k = 0 ..= nFft/2`. |
| `nextPow2(n)` | |
| `hzToMel(hz)`, `melToHz(mel)` | HTK: `mel = 2595 · log10(1 + hz / 700)`. |
| `melFilterbank({ nMels, nFft, sampleRate, fmin, fmax })` | Triangles with peak 1. Returns `{ filters, centersHz, bins }`. |
| `applyFilterbank(fb, power)` | One energy per filter. |
| `quantize(x, bits)`, `dequantize(q, bits)` | PCM rounding exactly as `Wav::write` and `Wav::read` do it. |
| `sincInterp(x, pos, halfWidth)`, `resample(x, fromRate, toRate)` | Windowed-sinc interpolation and resampling. |

`mel.rs`, `fft.rs` and `resample.rs` were not written yet when this file was. If the Rust ends up differing (for example in how the filterbank treats its edges or normalizes its triangles), the Rust is right and `shared.js` must follow.

## Do

- **Read colours from `t`** inside every draw call. A canvas that caches a colour will be wrong after the reader switches theme.
- **Draw in CSS pixels.** `SR.canvas` has already scaled the context. Never touch `canvas.width` or `devicePixelRatio` yourself.
- **Size canvases in CSS** with `--h`, and give them a smaller `--h` under 800px. Compute everything inside `draw` from the `w` and `h` you are given, so a resize is just another draw.
- **Thin out as the canvas narrows.** Fewer ticks (`w < 440`), dots only when they are at least about 4px apart, a pill saying what was left out. Check at 380px.
- **Clip data to the plot box** (`ctx.save(); ctx.rect(...); ctx.clip(); ... ctx.restore();`) so lines do not run over the tick labels.
- **Use `SR.crisp`** for 1px lines; leave curves unsnapped.
- **One accent at a time.** Green is the subject of the figure. Amber appears when something needs watching (on the sampling page the curve turns from green to amber when it aliases). Red only means wrong. Everything else is ink at some opacity.
- **Make each step set the whole state**, not a delta from the previous step. Readers jump around with the numbered dots.
- **Set state at once when `animate` is false** (page load). If you tween between steps, do it with `SR.tween`, and cancel the tween when the reader touches a control.
- **Give every control a visible label and a live readout.** Sliders: `<label for>` plus `<output for>`. Segmented controls: `role="radiogroup"` with `aria-labelledby`.
- **Describe each canvas in words.** `role="img"` and an `aria-label` that you rewrite in `update()` with the numbers that matter ("A 5,000 Hz tone sampled 8,000 times a second appears as 3,000 Hz").
- **Write like the book.** First person plural, plain words, a concrete number over an abstraction, no exclamation marks, no emoji. Step text: three or four sentences. Define a term in bold the first time it appears.
- **Use the book's numbers** as defaults: 16,000 Hz, 25 ms frames, 10 ms hop, 40 mel filters, fmin 20 Hz, blank is class 0.
- **Test** at 380, 800 and 1400px wide, in both themes, with the keyboard only, with the console open, and from `file://`.

## Don't

- Don't load anything from the network, and don't use `import`/`type="module"` (Chrome blocks module scripts on `file://`).
- Don't add a second stylesheet's worth of CSS to your page. If you need a new component, add it to `shared.css`, document it here, and everyone gets it.
- Don't hard-code colours, fonts or pixel ratios.
- Don't rely on `requestAnimationFrame` for correctness. Background tabs do not get frames; state must be right even if nothing has been drawn yet.
- Don't use hover for anything a reader needs. There is no hover on a phone.
- Don't autoplay sound, and don't start an animation that never ends. If something loops, give it a pause button and stop it under `prefers-reduced-motion`.
- Don't use rainbow colour maps. `SR.heat` only.
- Don't use drop shadows, gradients (other than the heat ramp) or rounded corners larger than 10px.
- Don't let the page scroll sideways at 380px. Long mono strings go in `sr-values`, which cuts them off.
- Don't put a figure's meaning only in colour. Dashes, hollow dots and labels carry it too (see the frequency axis on the sampling page).
- Don't invent results. If a page shows an accuracy or a timing, it came from a run of the Rust code.

## Things that surprised me

- Opening `file://` in some embedded browser previews shows a static snapshot with no CSS or JS. That is the preview, not the page. `python3 -m http.server` from the repo root and `http://127.0.0.1:8000/interactive/` is the quick way to check; then check a real `file://` open in Chrome or Safari as well.
- `[hidden]` loses to any `display:` rule, so `shared.css` forces it. You can hide legend items with the `hidden` attribute.
- A log slider with rounded values needs snapping, or the reader can never land on exactly 16,000. `SR.logMap` has `snap` for that.
- When the swap from "Play" to "Stop" changes a button's width the whole row twitches. `SR.audio.button` pins the width; if you build your own toggle, do the same.
- Text under a stepper changes length from step to step and makes the figure jump. `--sr-step-lines` reserves the room.

## Embedded in a chapter

The book's website (built by `site/`) shows a page inside its chapter, in an `<iframe>`, as `page.html?embed=1`. You do not have to do anything for this to work, as long as the page uses the standard shell:

- `shared.js` adds the class `sr-embed` to `<html>` and sets `SR.embed` to `true`.
- `shared.css` then hides `.sr-topbar`, `.sr-intro` and `.sr-footer` and stacks the stage above the controls. The chapter around the frame does the introducing, so anything the reader needs in order to use the figure belongs in the stage or the controls, not in the intro.
- The page reports its height to the chapter whenever it changes, so the frame is exactly as tall as the page and never scrolls. Do not size anything from the height of the window (`100vh`).
- The chapter passes on the reader's theme when they switch it.


## The one page with a build step

`listen.html` runs the book's recognizer in the browser. It loads two files that are not in this directory: `listen.wasm`, which is the crate in `web/` compiled for `wasm32-unknown-unknown`, and `recognizer.bin`, the trained weights from `models/`. The website build (`site/`) compiles the first and copies both next to the page. Opened straight from this directory the page says the recognizer is missing, and that is expected. A browser will also not let a page fetch a `.wasm` file or open the microphone from `file://`, so try it through a local web server, from `_site/`.
