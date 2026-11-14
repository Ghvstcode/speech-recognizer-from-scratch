/* ==========================================================================
   shared.js
   Helpers for the interactive pages of
   "Building a Speech Recognizer from Scratch".

   A classic script (no modules, no build step) so it works from file://.
   Load it in <head>:   <script src="shared.js"></script>
   It exposes one global, `SR`. Read interactive/README.md for the tour.

   The maths in SR.dsp mirrors the Rust in code/src/dsp/ so that what a page
   draws is what the reader's program computes. If you change one, change both.
   ========================================================================== */

(function () {
  'use strict';

  var SR = {};
  var root = document.documentElement;

  /* ------------------------------------------------------------------------
     Theme
     The reader's choice lives in localStorage and on <html data-theme>.
     No choice means "follow the system". Runs before first paint because
     this script is loaded in <head>.
     ------------------------------------------------------------------------ */

  var THEME_KEY = 'sr-theme';
  var darkQuery = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

  function storedTheme() {
    try {
      var t = window.localStorage.getItem(THEME_KEY);
      return t === 'light' || t === 'dark' ? t : null;
    } catch (e) { return null; }
  }

  function currentTheme() {
    var t = root.getAttribute('data-theme');
    if (t === 'light' || t === 'dark') return t;
    return darkQuery && darkQuery.matches ? 'dark' : 'light';
  }

  function announceTheme() {
    tokenCache = null;
    syncThemeButtons();
    document.dispatchEvent(new CustomEvent('sr:themechange', { detail: { theme: currentTheme() } }));
  }

  function setTheme(theme) {
    if (theme === 'light' || theme === 'dark') {
      root.setAttribute('data-theme', theme);
      try { window.localStorage.setItem(THEME_KEY, theme); } catch (e) { /* private mode: fine */ }
    } else {
      root.removeAttribute('data-theme');
      try { window.localStorage.removeItem(THEME_KEY); } catch (e) { /* fine */ }
    }
    announceTheme();
  }

  var SUN = '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.25" stroke-linecap="round" aria-hidden="true"><circle cx="8" cy="8" r="2.75"/><path d="M8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.15 1.15M11.45 11.45l1.15 1.15M3.4 12.6l1.15-1.15M11.45 4.55l1.15-1.15"/></svg>';
  var MOON = '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M13.2 9.6A5.6 5.6 0 0 1 6.4 2.8a5.6 5.6 0 1 0 6.8 6.8Z"/></svg>';

  function syncThemeButtons() {
    var dark = currentTheme() === 'dark';
    var buttons = document.querySelectorAll('[data-sr-theme]');
    for (var i = 0; i < buttons.length; i++) {
      var b = buttons[i];
      // Show where the click takes you: the sun when it is dark, the moon when it is light.
      b.innerHTML = dark ? SUN : MOON;
      b.setAttribute('aria-label', dark ? 'Switch to the light theme' : 'Switch to the dark theme');
      b.setAttribute('title', dark ? 'Light theme' : 'Dark theme');
    }
  }

  (function initTheme() {
    var t = storedTheme();
    if (t) root.setAttribute('data-theme', t);
    if (darkQuery) {
      var onSystemChange = function () { if (!storedTheme()) announceTheme(); };
      if (darkQuery.addEventListener) darkQuery.addEventListener('change', onSystemChange);
      else if (darkQuery.addListener) darkQuery.addListener(onSystemChange);
    }
  })();

  SR.theme = {
    /** 'light' or 'dark': what is on screen right now. */
    get: currentTheme,
    /** 'light', 'dark', or null to follow the system again. */
    set: setTheme,
    toggle: function () { setTheme(currentTheme() === 'dark' ? 'light' : 'dark'); },
    /** Call fn(theme) whenever the theme changes. */
    onChange: function (fn) {
      document.addEventListener('sr:themechange', function (e) { fn(e.detail.theme); });
    }
  };

  /* ------------------------------------------------------------------------
     Tokens
     Canvases cannot use var(--ink), so we read the computed values once per
     theme and hand them to every draw call.
     ------------------------------------------------------------------------ */

  var tokenCache = null;
  var TOKEN_NAMES = {
    paper: '--paper', panel: '--panel', ink: '--ink',
    ink70: '--ink-70', ink60: '--ink-60', ink48: '--ink-48', ink30: '--ink-30',
    ink15: '--ink-15', ink10: '--ink-10', ink7: '--ink-7', ink4: '--ink-4',
    green: '--green', amber: '--amber', red: '--red',
    greenWash: '--green-wash', amberWash: '--amber-wash', redWash: '--red-wash',
    heat0: '--heat-0', heat1: '--heat-1', heat2: '--heat-2',
    sans: '--sans', mono: '--mono'
  };

  /** The design tokens as plain strings, e.g. SR.tokens().green. Cached per theme. */
  SR.tokens = function () {
    if (tokenCache) return tokenCache;
    var cs = window.getComputedStyle(root);
    var t = {};
    for (var key in TOKEN_NAMES) t[key] = cs.getPropertyValue(TOKEN_NAMES[key]).trim();
    t.dark = currentTheme() === 'dark';
    t.heatStops = [parseColor(t.heat0), parseColor(t.heat1), parseColor(t.heat2)];
    tokenCache = t;
    return t;
  };

  /** '#rgb', '#rrggbb' or 'rgb(a)(r, g, b...)' to [r, g, b]. */
  function parseColor(str) {
    str = String(str).trim();
    if (str[0] === '#') {
      var h = str.slice(1);
      if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
      var n = parseInt(h.slice(0, 6), 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    var m = str.match(/[\d.]+/g) || [0, 0, 0];
    return [+m[0], +m[1], +m[2]];
  }
  SR.parseColor = parseColor;

  /* ------------------------------------------------------------------------
     Small utilities
     ------------------------------------------------------------------------ */

  SR.clamp = function (v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; };
  SR.lerp = function (a, b, t) { return a + (b - a) * t; };

  /** True when the reader asked the system for less motion. */
  SR.reducedMotion = function () {
    return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  };

  /**
   * Animate t from 0 to 1 over `ms`, calling onUpdate(easedT) each frame.
   * Jumps straight to 1 under prefers-reduced-motion or in a hidden tab.
   * Returns a function that cancels the animation.
   */
  SR.tween = function (ms, onUpdate, onDone) {
    if (SR.reducedMotion() || document.hidden || ms <= 0) {
      onUpdate(1);
      if (onDone) onDone();
      return function () {};
    }
    var start = null, raf = 0, finished = false;
    function finish() {
      if (finished) return;
      finished = true;
      window.cancelAnimationFrame(raf);
      window.clearTimeout(guard);
      onUpdate(1);
      if (onDone) onDone();
    }
    function frame(now) {
      if (finished) return;
      if (start === null) start = now;
      var t = Math.min(1, (now - start) / ms);
      if (t >= 1) { finish(); return; }
      onUpdate(1 - Math.pow(1 - t, 3)); // ease-out cubic
      raf = window.requestAnimationFrame(frame);
    }
    raf = window.requestAnimationFrame(frame);
    // If the browser stops handing out frames (a background tab), still land on the end state.
    var guard = window.setTimeout(finish, ms + 300);
    return function () { finished = true; window.cancelAnimationFrame(raf); window.clearTimeout(guard); };
  };

  /* Number formatting. One place, so every page prints "16,000 Hz" the same way. */
  SR.fmt = {
    /** 16000 -> "16,000" */
    int: function (v) {
      var s = String(Math.abs(Math.round(v)));
      var out = '';
      while (s.length > 3) { out = ',' + s.slice(-3) + out; s = s.slice(0, -3); }
      return (v < 0 && Math.round(v) !== 0 ? '−' : '') + s + out;
    },
    /** 3.14159, 2 -> "3.14" (with a real minus sign) */
    fixed: function (v, digits) {
      var s = Math.abs(v).toFixed(digits == null ? 2 : digits);
      return (v < 0 && +s !== 0 ? '−' : '') + s;
    },
    /** A number with about `sig` significant digits and no exponent for everyday sizes. */
    sig: function (v, sig) {
      sig = sig || 3;
      if (v === 0) return '0';
      var a = Math.abs(v);
      var sign = v < 0 ? '−' : '';
      if (a >= 1000) return sign + SR.fmt.int(a);
      if (a < 0.001) {
        var e = Math.floor(Math.log10(a));
        return sign + (a / Math.pow(10, e)).toFixed(Math.max(0, sig - 1)) + '×10' + superscript(e);
      }
      var digits = Math.max(0, sig - 1 - Math.floor(Math.log10(a)));
      return sign + a.toFixed(digits);
    },
    /** 16000 -> "16,000 Hz" */
    hz: function (v) { return (v < 100 && v !== Math.round(v) ? SR.fmt.fixed(v, 1) : SR.fmt.int(v)) + ' Hz'; },
    /** 16000 -> "16k", 500 -> "500": short tick labels for frequency axes. */
    hzShort: function (v) { return v >= 1000 ? (+(v / 1000).toFixed(2)) + 'k' : String(+v.toFixed(1)); }
  };

  function superscript(n) {
    var map = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
    return String(n).split('').map(function (c) { return map[c] || c; }).join('');
  }

  /* ------------------------------------------------------------------------
     DOM helpers
     ------------------------------------------------------------------------ */

  function setAttrs(node, attrs) {
    for (var k in attrs) {
      var v = attrs[k];
      if (v == null || v === false) continue;
      if (k === 'text') node.textContent = v;
      else if (k === 'html') node.innerHTML = v;
      else if (k.slice(0, 2) === 'on' && typeof v === 'function') node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? '' : v);
    }
  }

  function append(node, children) {
    if (children == null) return;
    if (!Array.isArray(children)) children = [children];
    for (var i = 0; i < children.length; i++) {
      var c = children[i];
      if (c == null) continue;
      node.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
    }
  }

  /** SR.el('button', { class: 'sr-btn', text: 'Go', onclick: fn }) */
  SR.el = function (tag, attrs, children) {
    var node = document.createElement(tag);
    if (attrs) setAttrs(node, attrs);
    append(node, children);
    return node;
  };

  /** SR.svg('circle', { cx: 10, cy: 10, r: 3, fill: 'var(--green)' }) */
  SR.svg = function (tag, attrs, children) {
    var node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    if (attrs) setAttrs(node, attrs);
    append(node, children);
    return node;
  };

  /** Shorthand for querySelector that accepts an element or a selector. */
  function resolve(target) {
    return typeof target === 'string' ? document.querySelector(target) : target;
  }

  /* ------------------------------------------------------------------------
     Canvas
     ------------------------------------------------------------------------ */

  /**
   * Make a canvas sharp on any screen and keep it that way.
   *
   *   var fig = SR.canvas('#wave', function (ctx, w, h, t) { ...draw in CSS pixels... });
   *   fig.redraw();   // after any state change; calls are coalesced per frame
   *
   * The canvas takes its size from CSS (class "sr-canvas", height from --h).
   * `draw` runs again on resize, on devicePixelRatio change and on theme change.
   * `t` is SR.tokens(), so colours always follow the theme.
   */
  SR.canvas = function (target, draw) {
    var canvas = resolve(target);
    var ctx = canvas.getContext('2d');
    var w = 0, h = 0, raf = 0, timer = 0;

    function measure() {
      var rect = canvas.getBoundingClientRect();
      var dpr = window.devicePixelRatio || 1;
      w = Math.max(1, rect.width);
      h = Math.max(1, rect.height);
      var pw = Math.max(1, Math.round(w * dpr));
      var ph = Math.max(1, Math.round(h * dpr));
      if (canvas.width !== pw || canvas.height !== ph) {
        canvas.width = pw;
        canvas.height = ph;
      }
      ctx.setTransform(pw / w, 0, 0, ph / h, 0, 0);
    }

    function render() {
      raf = 0; timer = 0;
      measure();
      ctx.clearRect(0, 0, w, h);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      draw(ctx, w, h, SR.tokens());
    }

    function redraw() {
      if (raf) return;
      // A hidden tab never gets an animation frame, so use a timer there.
      if (document.hidden) { raf = -1; timer = window.setTimeout(render, 0); return; }
      raf = window.requestAnimationFrame(render);
    }

    function cancel() {
      if (raf > 0) window.cancelAnimationFrame(raf);
      if (timer) window.clearTimeout(timer);
      raf = 0; timer = 0;
    }

    if (window.ResizeObserver) new ResizeObserver(redraw).observe(canvas);
    window.addEventListener('resize', redraw);
    document.addEventListener('sr:themechange', redraw);
    redraw();

    return {
      canvas: canvas,
      ctx: ctx,
      redraw: redraw,
      /** Draw now instead of on the next frame (rarely needed). */
      flush: function () { cancel(); render(); },
      get width() { return w; },
      get height() { return h; },
      /** Pointer event to canvas coordinates in CSS pixels. */
      point: function (evt) {
        var r = canvas.getBoundingClientRect();
        return { x: evt.clientX - r.left, y: evt.clientY - r.top };
      }
    };
  };

  /** Snap a coordinate to the middle of a device pixel so 1px lines stay 1px. */
  SR.crisp = function (v) { return Math.round(v) + 0.5; };

  /* ------------------------------------------------------------------------
     Scales and ticks
     ------------------------------------------------------------------------ */

  /**
   * var x = SR.scale.linear([0, 10], [40, 600]);  x(5) -> 320;  x.invert(320) -> 5
   * var f = SR.scale.log([50, 30000], [40, 600]);
   */
  SR.scale = {
    linear: function (domain, range) {
      var d0 = domain[0], d1 = domain[1], r0 = range[0], r1 = range[1];
      var k = (r1 - r0) / ((d1 - d0) || 1);
      var s = function (v) { return r0 + (v - d0) * k; };
      s.invert = function (p) { return d0 + (p - r0) / (k || 1); };
      s.domain = [d0, d1];
      s.range = [r0, r1];
      return s;
    },
    log: function (domain, range) {
      var l0 = Math.log(domain[0]), l1 = Math.log(domain[1]), r0 = range[0], r1 = range[1];
      var k = (r1 - r0) / ((l1 - l0) || 1);
      var s = function (v) { return r0 + (Math.log(v) - l0) * k; };
      s.invert = function (p) { return Math.exp(l0 + (p - r0) / (k || 1)); };
      s.domain = [domain[0], domain[1]];
      s.range = [r0, r1];
      return s;
    }
  };

  /** About `count` round numbers covering [min, max]: SR.ticks(0, 5, 5) -> [0, 1, 2, 3, 4, 5] */
  SR.ticks = function (min, max, count) {
    count = count || 5;
    var span = max - min;
    if (!(span > 0)) return [min];
    var raw = span / count;
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var norm = raw / mag;
    var step = (norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10) * mag;
    var first = Math.ceil(min / step - 1e-9) * step;
    var out = [];
    for (var v = first; v <= max + step * 1e-9; v += step) out.push(+v.toPrecision(12));
    return out;
  };

  /** 1-2-5 ticks for a log axis: SR.logTicks(50, 30000) -> [50, 100, 200, 500, 1000, ...] */
  SR.logTicks = function (min, max) {
    var out = [];
    for (var e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e++) {
      [1, 2, 5].forEach(function (m) {
        var v = m * Math.pow(10, e);
        if (v >= min * 0.999 && v <= max * 1.001) out.push(v);
      });
    }
    return out;
  };

  /* ------------------------------------------------------------------------
     Drawing in the house style
     ------------------------------------------------------------------------ */

  /**
   * The rectangle a plot lives in, leaving room for tick labels.
   * SR.plotBox(w, h) or SR.plotBox(w, h, { l: 44, r: 12, t: 18, b: 30 })
   */
  SR.plotBox = function (w, h, m) {
    m = m || {};
    var l = m.l == null ? 44 : m.l, r = m.r == null ? 12 : m.r;
    var t = m.t == null ? 18 : m.t, b = m.b == null ? 30 : m.b;
    return { x: l, y: t, w: Math.max(10, w - l - r), h: Math.max(10, h - t - b) };
  };

  SR.font = {
    /** ctx.font = SR.font.mono(10) */
    mono: function (px) { return (px || 10) + 'px ' + SR.tokens().mono; },
    sans: function (px, weight) { return (weight || 400) + ' ' + (px || 12) + 'px ' + SR.tokens().sans; }
  };

  /**
   * Axes in the house style: hairline grid, mono tick labels at ink 48%,
   * uppercase mono axis labels. Units go in the labels.
   *
   *   SR.axes(ctx, {
   *     box: box, x: xScale, y: yScale,
   *     xTicks: [0, 1, 2], yTicks: [-1, 0, 1],      // values, or omit for none
   *     xFormat: fn, yFormat: fn,                   // value -> string
   *     xLabel: 'time (ms)', yLabel: 'amplitude',   // drawn bottom right / top left
   *     grid: true,                                 // hairlines at the ticks
   *     zero: true                                  // a stronger line at y = 0
   *   });
   */
  SR.axes = function (ctx, o) {
    var t = SR.tokens(), b = o.box, c = SR.crisp;
    var xf = o.xFormat || String, yf = o.yFormat || String;
    var grid = o.grid !== false;
    var i, v, p;
    ctx.save();
    ctx.lineWidth = 1;
    ctx.setLineDash([]);

    if (o.yTicks && o.y) {
      ctx.font = SR.font.mono(10);
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (i = 0; i < o.yTicks.length; i++) {
        v = o.yTicks[i];
        p = c(o.y(v));
        if (grid) {
          ctx.strokeStyle = t.ink7;
          ctx.beginPath(); ctx.moveTo(b.x, p); ctx.lineTo(b.x + b.w, p); ctx.stroke();
        }
        ctx.fillStyle = t.ink48;
        ctx.fillText(yf(v), b.x - 8, p);
      }
    }

    if (o.xTicks && o.x) {
      ctx.font = SR.font.mono(10);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      for (i = 0; i < o.xTicks.length; i++) {
        v = o.xTicks[i];
        p = c(o.x(v));
        if (grid) {
          ctx.strokeStyle = t.ink7;
          ctx.beginPath(); ctx.moveTo(p, b.y); ctx.lineTo(p, b.y + b.h); ctx.stroke();
        }
        ctx.strokeStyle = t.ink30;
        ctx.beginPath(); ctx.moveTo(p, c(b.y + b.h)); ctx.lineTo(p, c(b.y + b.h) + 4); ctx.stroke();
        ctx.fillStyle = t.ink48;
        ctx.fillText(xf(v), p, b.y + b.h + 8);
      }
    }

    // The baseline, and optionally y = 0.
    ctx.strokeStyle = t.ink30;
    ctx.beginPath();
    ctx.moveTo(b.x, c(b.y + b.h)); ctx.lineTo(b.x + b.w, c(b.y + b.h));
    ctx.stroke();
    if (o.zero && o.y) {
      p = c(o.y(0));
      if (p > b.y && p < b.y + b.h - 1) {
        ctx.strokeStyle = t.ink15;
        ctx.beginPath(); ctx.moveTo(b.x, p); ctx.lineTo(b.x + b.w, p); ctx.stroke();
      }
    }

    ctx.fillStyle = t.ink48;
    ctx.font = SR.font.mono(10);
    if (o.yLabel) {
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      SR.spacedText(ctx, o.yLabel.toUpperCase(), b.x, b.y - 7);
    }
    if (o.xLabel) {
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      SR.spacedText(ctx, o.xLabel.toUpperCase(), b.x + b.w, b.y + b.h + 8 + (o.xTicks ? 13 : 0));
    }
    ctx.restore();
  };

  /** fillText with the overline letter-spacing, where the browser supports it. */
  SR.spacedText = function (ctx, text, x, y) {
    var had = 'letterSpacing' in ctx;
    if (had) ctx.letterSpacing = '0.8px';
    ctx.fillText(text, x, y);
    if (had) ctx.letterSpacing = '0px';
  };

  /**
   * Stroke a curve y = f(i) through n points.
   *   SR.line(ctx, n, function (i) { return [px, py]; }, { color: t.green, width: 1.75 });
   * Options: color, width (default 1.5), dash (e.g. [3, 3]), alpha.
   */
  SR.line = function (ctx, n, pointAt, o) {
    o = o || {};
    ctx.save();
    ctx.beginPath();
    for (var i = 0; i < n; i++) {
      var p = pointAt(i);
      if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
    }
    ctx.strokeStyle = o.color || SR.tokens().ink;
    ctx.lineWidth = o.width || 1.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.setLineDash(o.dash || []);
    if (o.alpha != null) ctx.globalAlpha = o.alpha;
    ctx.stroke();
    ctx.restore();
  };

  /** A filled dot with a paper ring, so it reads on top of any line. */
  SR.dot = function (ctx, x, y, r, color) {
    var t = SR.tokens();
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, (r || 3) + 1.25, 0, Math.PI * 2);
    ctx.fillStyle = t.panel;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, r || 3, 0, Math.PI * 2);
    ctx.fillStyle = color || t.ink;
    ctx.fill();
    ctx.restore();
  };

  /**
   * A small mono label in a pill, the canvas twin of .sr-pill.
   *   SR.pill(ctx, 'NYQUIST 8,000 HZ', x, y, { align: 'center', color: t.amber, min: box.x, max: box.x + box.w });
   * (x, y) is the anchor: horizontally per `align`, vertically the middle.
   * `min`/`max` keep the pill inside the plot. Returns its rectangle.
   */
  SR.pill = function (ctx, text, x, y, o) {
    o = o || {};
    var t = SR.tokens();
    ctx.save();
    ctx.font = SR.font.mono(10);
    var hadSpacing = 'letterSpacing' in ctx;
    if (hadSpacing) ctx.letterSpacing = '0.4px';
    var tw = ctx.measureText(text).width;
    var w = tw + 14, h = 18;
    var left = o.align === 'left' ? x : o.align === 'right' ? x - w : x - w / 2;
    if (o.min != null) left = Math.max(o.min, left);
    if (o.max != null) left = Math.min(o.max - w, left);
    left = Math.round(left);
    var top = Math.round(y - h / 2);
    ctx.beginPath();
    roundRect(ctx, left + 0.5, top + 0.5, w, h, h / 2);
    ctx.fillStyle = o.fill || t.panel;
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    ctx.strokeStyle = o.stroke || (o.color ? o.color : t.ink15);
    ctx.stroke();
    ctx.fillStyle = o.color || t.ink70;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, left + 7, top + h / 2 + 1);
    if (hadSpacing) ctx.letterSpacing = '0px';
    ctx.restore();
    return { x: left, y: top, w: w, h: h };
  };

  function roundRect(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
  /** Adds a rounded rectangle to the current path. */
  SR.roundRect = roundRect;

  /* Heatmaps: one hue, paper to deep green (and its dark-theme twin). */

  /** t in [0, 1] -> [r, g, b] on the heat ramp of the current theme. */
  SR.heat = function (v) {
    var s = SR.tokens().heatStops;
    v = v < 0 ? 0 : v > 1 ? 1 : v;
    var a = v < 0.5 ? s[0] : s[1], b = v < 0.5 ? s[1] : s[2];
    var u = v < 0.5 ? v * 2 : (v - 0.5) * 2;
    return [
      Math.round(a[0] + (b[0] - a[0]) * u),
      Math.round(a[1] + (b[1] - a[1]) * u),
      Math.round(a[2] + (b[2] - a[2]) * u)
    ];
  };

  /** t in [0, 1] -> 'rgb(r, g, b)' */
  SR.heatCss = function (v) {
    var c = SR.heat(v);
    return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
  };

  /**
   * Paint a rows x cols grid of numbers into a rectangle of the canvas.
   *   SR.heatmap(ctx, data, rows, cols, box, { min: -10, max: 0, flipY: true });
   * `data` is row-major (data[r * cols + c]), like Mat in the Rust code.
   * flipY puts row 0 at the bottom (handy when rows are frequency bins).
   * Cells are drawn as crisp rectangles; fine up to a few hundred per side.
   */
  SR.heatmap = function (ctx, data, rows, cols, box, o) {
    o = o || {};
    var min = o.min, max = o.max, i;
    if (min == null || max == null) {
      var lo = Infinity, hi = -Infinity;
      for (i = 0; i < rows * cols; i++) { if (data[i] < lo) lo = data[i]; if (data[i] > hi) hi = data[i]; }
      if (min == null) min = lo;
      if (max == null) max = hi;
    }
    var span = (max - min) || 1;
    var cw = box.w / cols, ch = box.h / rows;
    for (var r = 0; r < rows; r++) {
      var y0 = Math.round(box.y + (o.flipY ? rows - 1 - r : r) * ch);
      var y1 = Math.round(box.y + ((o.flipY ? rows - 1 - r : r) + 1) * ch);
      for (var c = 0; c < cols; c++) {
        var x0 = Math.round(box.x + c * cw), x1 = Math.round(box.x + (c + 1) * cw);
        ctx.fillStyle = SR.heatCss((data[r * cols + c] - min) / span);
        ctx.fillRect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
      }
    }
  };

  /* ------------------------------------------------------------------------
     Controls
     ------------------------------------------------------------------------ */

  /**
   * A mapping between slider positions and values on a log scale, for
   * frequencies and the like.
   *   SR.logMap(100, 10000, { steps: 400, sig: 3, snap: [440, 1000] })
   * `sig` rounds to that many significant digits; `snap` lists values the
   * slider should land on exactly when it gets close.
   */
  SR.logMap = function (min, max, o) {
    o = o || {};
    var steps = o.steps || 400, sig = o.sig || 3, snap = o.snap || [];
    var ratio = Math.log(max / min);
    var perStep = ratio / steps;
    return {
      steps: steps,
      toValue: function (pos) {
        var v = min * Math.exp(ratio * pos / steps);
        for (var i = 0; i < snap.length; i++) {
          if (Math.abs(Math.log(v / snap[i])) < perStep * 1.01) return snap[i];
        }
        var mag = Math.pow(10, Math.floor(Math.log10(v)) - (sig - 1));
        return SR.clamp(Math.round(v / mag) * mag, min, max);
      },
      toPos: function (v) { return SR.clamp(Math.round(Math.log(v / min) / ratio * steps), 0, steps); }
    };
  };

  /**
   * Tie a range input to its readout and to your state.
   *
   *   var freq = SR.bindSlider('#freq', {
   *     format: SR.fmt.hz,                 // value -> text for the readout
   *     onInput: function (v) { state.f = v; draw(); },
   *     map: SR.logMap(100, 10000)         // optional: non-linear slider
   *   });
   *   freq.set(440);          // move the slider without calling onInput
   *   freq.value              // the current value
   *
   * The readout is the <output for="id"> of the input, or opts.readout.
   * Keeps the filled part of the track and aria-valuetext up to date.
   */
  SR.bindSlider = function (target, o) {
    o = o || {};
    var input = resolve(target);
    var readout = o.readout ? resolve(o.readout) : (input.id ? document.querySelector('output[for="' + input.id + '"]') : null);
    var map = o.map || null;
    var format = o.format || String;
    if (map) { input.min = 0; input.max = map.steps; input.step = 1; }
    var value;

    function paint() {
      var lo = +input.min || 0, hi = input.max === '' ? 100 : +input.max;
      var pct = hi > lo ? (input.value - lo) / (hi - lo) * 100 : 0;
      input.style.setProperty('--sr-pct', pct.toFixed(2) + '%');
      var text = format(value);
      if (readout) readout.textContent = text;
      input.setAttribute('aria-valuetext', text);
    }

    function read() { value = map ? map.toValue(+input.value) : +input.value; }

    input.addEventListener('input', function () {
      read();
      paint();
      if (o.onInput) o.onInput(value);
    });

    var api = {
      input: input,
      get value() { return value; },
      /** Set the value (any value in range, not only slider stops) without firing onInput. */
      set: function (v) {
        value = v;
        input.value = map ? map.toPos(v) : v;
        paint();
      },
      /** Re-run the formatter, e.g. when the label depends on other state. */
      refresh: paint
    };

    if (o.value != null) api.set(o.value); else { read(); paint(); }
    return api;
  };

  /**
   * A segmented control or any group of radio inputs sharing a name.
   *   var mode = SR.bindRadios('signal', function (v) { state.mode = v; draw(); });
   *   mode.set('chirp');  mode.value
   */
  SR.bindRadios = function (name, onChange) {
    var inputs = Array.prototype.slice.call(document.querySelectorAll('input[type="radio"][name="' + name + '"]'));
    inputs.forEach(function (input) {
      input.addEventListener('change', function () { if (input.checked && onChange) onChange(input.value); });
    });
    return {
      inputs: inputs,
      get value() {
        for (var i = 0; i < inputs.length; i++) if (inputs[i].checked) return inputs[i].value;
        return null;
      },
      set: function (v) { inputs.forEach(function (input) { input.checked = input.value === String(v); }); }
    };
  };

  /**
   * A checkbox.
   *   var show = SR.bindCheck('#show-dots', function (on) { state.dots = on; draw(); });
   *   show.set(false);  show.value
   */
  SR.bindCheck = function (target, onChange) {
    var input = resolve(target);
    input.addEventListener('change', function () { if (onChange) onChange(input.checked); });
    return {
      input: input,
      get value() { return input.checked; },
      set: function (v) { input.checked = !!v; }
    };
  };

  /* ------------------------------------------------------------------------
     Stepper: a short guided walkthrough above the figure
     ------------------------------------------------------------------------ */

  /**
   *   var steps = SR.stepper('#steps', {
   *     steps: [
   *       { title: 'A wave', body: 'Text. <b>HTML is allowed.</b>', enter: function (animate) { ...set controls... } },
   *       ...
   *     ],
   *     onChange: function (index, step) {}     // optional, after enter()
   *   });
   *   steps.go(2); steps.next(); steps.prev(); steps.index
   *
   * Each step's enter(animate) should put the page into the state that tells
   * that part of the story. `animate` is false on page load and when the
   * reader re-enters the step they are on: set the state at once then.
   * The reader can still move every control afterwards.
   * Markup it builds: .sr-stepper > .sr-stepper-head, .sr-stepper-title, .sr-stepper-body
   */
  SR.stepper = function (target, o) {
    var host = resolve(target);
    var steps = o.steps;
    var index = -1;

    host.classList.add('sr-stepper');
    host.setAttribute('role', 'group');
    host.setAttribute('aria-label', o.label || 'Walkthrough');

    var count = SR.el('span', { class: 'sr-overline' });
    var dots = SR.el('ol', { class: 'sr-stepper-dots' });
    var dotButtons = steps.map(function (step, i) {
      var b = SR.el('button', {
        class: 'sr-stepper-dot', type: 'button', text: String(i + 1),
        'aria-label': 'Step ' + (i + 1) + ': ' + step.title,
        title: step.title,
        onclick: function () { go(i); }
      });
      dots.appendChild(SR.el('li', null, b));
      return b;
    });
    var prev = SR.el('button', { class: 'sr-btn sr-btn--sm', type: 'button', text: 'Back', onclick: function () { go(index - 1); } });
    var next = SR.el('button', { class: 'sr-btn sr-btn--sm sr-btn--primary', type: 'button', onclick: function () { go(index + 1 < steps.length ? index + 1 : 0); } });
    // The dots are plain buttons, so Tab reaches them; arrow keys also move between steps.
    host.addEventListener('keydown', function (e) {
      if (!e.target.classList || !e.target.classList.contains('sr-stepper-dot')) return;
      var to = e.key === 'ArrowRight' ? index + 1 : e.key === 'ArrowLeft' ? index - 1 : null;
      if (to === null || to < 0 || to >= steps.length) return;
      e.preventDefault();
      go(to);
      dotButtons[to].focus();
    });
    var title = SR.el('h2', { class: 'sr-stepper-title' });
    var body = SR.el('p', { class: 'sr-stepper-body' });
    var text = SR.el('div', { 'aria-live': 'polite' }, [title, body]);

    host.textContent = '';
    append(host, [
      SR.el('div', { class: 'sr-stepper-head' }, [
        count,
        SR.el('div', { class: 'sr-stepper-nav' }, [dots, SR.el('div', { class: 'sr-stepper-buttons' }, [prev, next])])
      ]),
      text
    ]);

    function go(i, quiet, initial) {
      i = SR.clamp(i, 0, steps.length - 1);
      var changed = i !== index;
      index = i;
      var step = steps[i];
      count.textContent = 'Step ' + (i + 1) + ' of ' + steps.length;
      title.textContent = step.title;
      body.innerHTML = step.body;
      dotButtons.forEach(function (b, j) {
        if (j === i) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current');
      });
      prev.disabled = i === 0;
      next.textContent = i === steps.length - 1 ? 'Start over' : 'Next';
      if (step.enter) step.enter(changed && !initial);
      if (!quiet && o.onChange) o.onChange(i, step);
    }

    go(o.start || 0, true, true);

    return {
      go: go,
      next: function () { go(index + 1); },
      prev: function () { go(index - 1); },
      get index() { return index; },
      get length() { return steps.length; }
    };
  };

  /* ------------------------------------------------------------------------
     Audio
     Sound starts only from a click, and whatever is playing can be stopped
     from the same button.
     ------------------------------------------------------------------------ */

  var audioCtx = null;
  var playing = null; // { source, gain, onEnd }

  function getAudioContext() {
    if (!audioCtx) {
      var AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      audioCtx = new AC();
    }
    if (audioCtx.state === 'suspended' && audioCtx.resume) audioCtx.resume();
    return audioCtx;
  }

  function stopAudio() {
    if (!playing) return;
    var p = playing;
    playing = null;
    try {
      var now = audioCtx.currentTime;
      p.gain.gain.cancelScheduledValues(now);
      p.gain.gain.setValueAtTime(p.gain.gain.value, now);
      p.gain.gain.linearRampToValueAtTime(0, now + 0.015); // a short fade, so stopping does not click
      p.source.stop(now + 0.02);
    } catch (e) { /* already stopped */ }
    if (p.onEnd) p.onEnd();
  }

  SR.audio = {
    /** False in the rare browser without Web Audio. */
    get supported() { return !!(window.AudioContext || window.webkitAudioContext); },
    get playing() { return !!playing; },

    /**
     * Play mono samples in [-1, 1] recorded at `sampleRate`. Call it from a click handler.
     *   SR.audio.play(float32Array, 16000, { gain: 0.25, onEnd: fn });
     * Any sample rate works: we resample to the sound card's rate ourselves
     * with SR.dsp.resample, so a 2,000 Hz recording sounds like one.
     * Starting a sound stops the previous one.
     */
    play: function (samples, sampleRate, o) {
      o = o || {};
      var ctx = getAudioContext();
      if (!ctx) return false;
      stopAudio();
      var data = sampleRate === ctx.sampleRate ? samples : SR.dsp.resample(samples, sampleRate, ctx.sampleRate);
      var buffer = ctx.createBuffer(1, Math.max(1, data.length), ctx.sampleRate);
      buffer.getChannelData(0).set(data);
      var source = ctx.createBufferSource();
      source.buffer = buffer;
      var gain = ctx.createGain();
      var level = o.gain == null ? 0.25 : o.gain;
      var now = ctx.currentTime, dur = buffer.duration;
      gain.gain.setValueAtTime(0, now);
      gain.gain.linearRampToValueAtTime(level, now + 0.012);
      gain.gain.setValueAtTime(level, now + Math.max(0.012, dur - 0.012));
      gain.gain.linearRampToValueAtTime(0, now + dur);
      source.connect(gain);
      gain.connect(ctx.destination);
      var me = { source: source, gain: gain, onEnd: o.onEnd };
      source.onended = function () {
        if (playing === me) { playing = null; if (me.onEnd) me.onEnd(); }
      };
      playing = me;
      source.start(now);
      return true;
    },

    /** Stop whatever is playing. */
    stop: stopAudio,

    /**
     * Turn a button into a play / stop toggle.
     *   SR.audio.button('#play', function () {
     *     return { samples: float32Array, sampleRate: 16000 };
     *   }, { gain: 0.25 });
     * While its sound plays the button shows a stop square and the word "Stop".
     * The button should contain its label as text; we add the icon.
     */
    button: function (target, make, o) {
      var btn = resolve(target);
      var label = btn.textContent.trim();
      var icon = SR.el('span', { 'aria-hidden': 'true', style: 'display:inline-grid' });
      var text = SR.el('span', { text: label });
      var PLAY = '<svg width="10" height="12" viewBox="0 0 10 12"><path d="M1 1.2v9.6a.6.6 0 0 0 .92.5l7.3-4.8a.6.6 0 0 0 0-1L1.92.7A.6.6 0 0 0 1 1.2Z" fill="currentColor"/></svg>';
      var STOP = '<svg width="10" height="12" viewBox="0 0 10 12"><rect x="0.5" y="1.5" width="9" height="9" rx="1.5" fill="currentColor"/></svg>';
      btn.textContent = '';
      btn.appendChild(icon);
      btn.appendChild(text);
      icon.innerHTML = PLAY;
      if (!SR.audio.supported) { btn.disabled = true; btn.title = 'This browser cannot play audio'; }

      function idle() {
        btn.removeAttribute('data-playing');
        icon.innerHTML = PLAY;
        text.textContent = label;
        btn.setAttribute('aria-label', 'Play: ' + label);
      }
      idle();

      btn.addEventListener('click', function () {
        if (btn.getAttribute('data-playing') === 'true') { stopAudio(); return; }
        var sound = make();
        if (!sound) return;
        // Keep the width, so swapping the label to "Stop" does not shift the row.
        btn.style.minWidth = Math.ceil(btn.getBoundingClientRect().width) + 'px';
        var ok = SR.audio.play(sound.samples, sound.sampleRate, { gain: (o && o.gain), onEnd: idle });
        if (!ok) return;
        btn.setAttribute('data-playing', 'true');
        icon.innerHTML = STOP;
        text.textContent = 'Stop';
        btn.setAttribute('aria-label', 'Stop: ' + label);
      });
      return btn;
    }
  };

  // Leaving the page (or switching tabs) should not leave a tone playing.
  document.addEventListener('visibilitychange', function () { if (document.hidden) stopAudio(); });

  /* ------------------------------------------------------------------------
     Maths shared by several pages. Mirrors code/src/dsp/*.rs.
     ------------------------------------------------------------------------ */

  var dsp = {};

  /* --- window.rs --------------------------------------------------------- */

  // a - (1 - a) cos(2 pi i / denominator), the family Hann and Hamming belong to.
  function raisedCosine(n, denominator, a) {
    var w = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      w[i] = denominator === 0 ? 1 : a - (1 - a) * Math.cos(2 * Math.PI * i / denominator);
    }
    return w;
  }

  /** The periodic Hann window, 0.5 - 0.5 cos(2 pi i / n). Our default, as in the book. */
  dsp.hann = function (n) { return raisedCosine(n, n, 0.5); };
  /** The symmetric Hann window, dividing by n - 1. For filter design. */
  dsp.hannSymmetric = function (n) { return raisedCosine(n, Math.max(0, n - 1), 0.5); };
  /** The periodic Hamming window, 0.54 - 0.46 cos(2 pi i / n). */
  dsp.hamming = function (n) { return raisedCosine(n, n, 0.54); };
  /** The symmetric Hamming window. */
  dsp.hammingSymmetric = function (n) { return raisedCosine(n, Math.max(0, n - 1), 0.54); };
  /** All ones: "no window at all". */
  dsp.rectangular = function (n) { var w = new Float64Array(n); w.fill(1); return w; };

  /* --- fft.rs ------------------------------------------------------------ */

  /**
   * The DFT straight from its definition, O(n^2):
   *   X[k] = sum over n of x[n] * e^(-2 pi i k n / N)
   * Takes real samples (and optionally imaginary parts); returns { re, im }.
   */
  dsp.dft = function (re, im) {
    var n = re.length;
    var outRe = new Float64Array(n), outIm = new Float64Array(n);
    for (var k = 0; k < n; k++) {
      var sr = 0, si = 0;
      for (var t = 0; t < n; t++) {
        var angle = -2 * Math.PI * k * t / n;
        var c = Math.cos(angle), s = Math.sin(angle);
        var xr = re[t], xi = im ? im[t] : 0;
        sr += xr * c - xi * s;
        si += xr * s + xi * c;
      }
      outRe[k] = sr;
      outIm[k] = si;
    }
    return { re: outRe, im: outIm };
  };

  /**
   * Radix-2 FFT, in place, on two arrays of the same power-of-two length.
   * Bit-reversal shuffle first, then log2(n) rounds of butterflies.
   */
  dsp.fftInPlace = function (re, im, inverse) {
    var n = re.length;
    if (n & (n - 1)) throw new Error('fft: length ' + n + ' is not a power of two');
    var i, j, k, tmp;
    for (i = 1, j = 0; i < n; i++) {
      var bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        tmp = re[i]; re[i] = re[j]; re[j] = tmp;
        tmp = im[i]; im[i] = im[j]; im[j] = tmp;
      }
    }
    for (var len = 2; len <= n; len <<= 1) {
      var angle = (inverse ? 2 : -2) * Math.PI / len;
      var wr = Math.cos(angle), wi = Math.sin(angle);
      var half = len >> 1;
      for (i = 0; i < n; i += len) {
        var cr = 1, ci = 0;
        for (k = 0; k < half; k++) {
          var a = i + k, b = a + half;
          var tr = re[b] * cr - im[b] * ci;
          var ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr; im[a] += ti;
          var ncr = cr * wr - ci * wi;
          ci = cr * wi + ci * wr;
          cr = ncr;
        }
      }
    }
    if (inverse) for (i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  };

  /** FFT of real samples, zero-padded to `nFft` (default: the next power of two). Returns { re, im }. */
  dsp.fft = function (samples, nFft) {
    var n = nFft || dsp.nextPow2(samples.length);
    var re = new Float64Array(n), im = new Float64Array(n);
    for (var i = 0; i < Math.min(n, samples.length); i++) re[i] = samples[i];
    dsp.fftInPlace(re, im, false);
    return { re: re, im: im };
  };

  /** Inverse FFT of { re, im }; returns { re, im }. */
  dsp.ifft = function (spec) {
    var re = Float64Array.from(spec.re), im = Float64Array.from(spec.im);
    dsp.fftInPlace(re, im, true);
    return { re: re, im: im };
  };

  /** |X[k]|^2 for k = 0 ..= n/2: the half of the spectrum that carries information. */
  dsp.powerSpectrum = function (samples, nFft) {
    var s = dsp.fft(samples, nFft);
    var bins = (s.re.length >> 1) + 1;
    var out = new Float64Array(bins);
    for (var k = 0; k < bins; k++) out[k] = s.re[k] * s.re[k] + s.im[k] * s.im[k];
    return out;
  };

  dsp.nextPow2 = function (n) { var p = 1; while (p < n) p <<= 1; return p; };

  /* --- mel.rs ------------------------------------------------------------ */

  /** The HTK formula: mel = 2595 * log10(1 + hz / 700). */
  dsp.hzToMel = function (hz) { return 2595 * Math.log10(1 + hz / 700); };
  /** Its inverse: hz = 700 * (10^(mel / 2595) - 1). */
  dsp.melToHz = function (mel) { return 700 * (Math.pow(10, mel / 2595) - 1); };

  /**
   * Triangular mel filterbank.
   *   var fb = SR.dsp.melFilterbank({ nMels: 40, nFft: 512, sampleRate: 16000, fmin: 20, fmax: 8000 });
   *   fb.filters[m][k]   // weight of FFT bin k (0 ..= nFft/2) in filter m
   *   fb.centersHz       // nMels + 2 edge/centre frequencies, evenly spaced in mel
   *
   * nMels + 2 points are spaced evenly on the mel scale between fmin and fmax.
   * Filter m rises from point m to 1 at point m + 1 and falls back to 0 at
   * point m + 2. Each bin is weighted by where its centre frequency
   * (k * sampleRate / nFft) falls on those slopes. Peaks are 1 (no area
   * normalization).
   */
  dsp.melFilterbank = function (o) {
    var nMels = o.nMels, nFft = o.nFft, sampleRate = o.sampleRate;
    var fmin = o.fmin == null ? 0 : o.fmin;
    var fmax = o.fmax == null ? sampleRate / 2 : o.fmax;
    var bins = (nFft >> 1) + 1;
    var melLo = dsp.hzToMel(fmin), melHi = dsp.hzToMel(fmax);
    var points = new Float64Array(nMels + 2);
    for (var i = 0; i < nMels + 2; i++) points[i] = dsp.melToHz(melLo + (melHi - melLo) * i / (nMels + 1));
    var filters = [];
    for (var m = 0; m < nMels; m++) {
      var left = points[m], centre = points[m + 1], right = points[m + 2];
      var row = new Float64Array(bins);
      for (var k = 0; k < bins; k++) {
        var hz = k * sampleRate / nFft;
        if (hz > left && hz < right) {
          row[k] = hz <= centre ? (hz - left) / (centre - left) : (right - hz) / (right - centre);
        }
      }
      filters.push(row);
    }
    return { filters: filters, centersHz: points, bins: bins };
  };

  /** Apply a filterbank to a power spectrum: one energy per mel filter. */
  dsp.applyFilterbank = function (fb, power) {
    var out = new Float64Array(fb.filters.length);
    for (var m = 0; m < fb.filters.length; m++) {
      var row = fb.filters[m], sum = 0;
      for (var k = 0; k < row.length; k++) sum += row[k] * power[k];
      out[m] = sum;
    }
    return out;
  };

  /* --- wav.rs ------------------------------------------------------------ */

  // f32::round in Rust rounds halves away from zero; Math.round rounds them up.
  function roundAway(x) { return x < 0 ? -Math.floor(-x + 0.5) : Math.floor(x + 0.5); }

  /**
   * Turn a sample in [-1, 1] into the integer a PCM file stores, the way
   * Wav::write does for 16 bits: multiply by 2^(bits-1), round, clamp.
   *   SR.dsp.quantize(0.5, 16) -> 16384
   */
  dsp.quantize = function (x, bits) {
    var scale = Math.pow(2, bits - 1);
    return SR.clamp(roundAway(x * scale), -scale, scale - 1);
  };
  /** And back, the way Wav::read does: divide by 2^(bits-1). */
  dsp.dequantize = function (q, bits) { return q / Math.pow(2, bits - 1); };

  /* --- resample.rs ------------------------------------------------------- */

  /**
   * The value "between the samples": windowed-sinc interpolation at a
   * fractional index `pos`, using `halfWidth` samples on each side
   * (symmetric Hann window). This is what an ideal playback device does,
   * give or take the finite window.
   */
  dsp.sincInterp = function (x, pos, halfWidth) {
    halfWidth = halfWidth || 16;
    var i0 = Math.floor(pos);
    var sum = 0;
    var from = Math.max(0, i0 - halfWidth + 1), to = Math.min(x.length - 1, i0 + halfWidth);
    for (var i = from; i <= to; i++) {
      var d = pos - i;
      if (d === 0) { sum += x[i]; continue; }
      var pd = Math.PI * d;
      var w = 0.5 + 0.5 * Math.cos(pd / halfWidth);
      sum += x[i] * Math.sin(pd) / pd * w;
    }
    return sum;
  };

  /**
   * Change the sample rate with a windowed-sinc filter.
   * Going up, it fills in the values between samples; going down, it first
   * removes what the lower rate cannot hold, so nothing aliases.
   */
  dsp.resample = function (x, fromRate, toRate, halfWidth) {
    halfWidth = halfWidth || 16;
    var ratio = toRate / fromRate;
    var n = Math.max(1, Math.round(x.length * ratio));
    var out = new Float32Array(n);
    var cutoff = Math.min(1, ratio);       // as a fraction of the source Nyquist
    var reach = halfWidth / cutoff;        // taps on each side, in source samples
    for (var j = 0; j < n; j++) {
      var pos = j / ratio;
      var from = Math.max(0, Math.ceil(pos - reach)), to = Math.min(x.length - 1, Math.floor(pos + reach));
      var sum = 0;
      for (var i = from; i <= to; i++) {
        var d = (pos - i) * cutoff;
        if (d === 0) { sum += x[i] * cutoff; continue; }
        var pd = Math.PI * d;
        sum += x[i] * cutoff * Math.sin(pd) / pd * (0.5 + 0.5 * Math.cos(pd / halfWidth));
      }
      out[j] = sum;
    }
    return out;
  };

  SR.dsp = dsp;

  /* ------------------------------------------------------------------------
     Start-up
     ------------------------------------------------------------------------ */

  /* ------------------------------------------------------------------------
     Embedding
     The book's website shows a page inside a chapter, in an <iframe>, as
     page.html?embed=1. In that mode shared.css hides the top bar, the intro
     and the footer (the chapter around the frame says all of that), and the
     page tells its parent how tall it is, so the frame never scrolls. The
     parent also passes on the reader's theme choice.
     ------------------------------------------------------------------------ */

  var EMBED = /[?&]embed\b/.test(window.location.search) && window.parent !== window;
  SR.embed = EMBED;
  if (EMBED) root.classList.add('sr-embed');

  function startEmbed() {
    var last = 0;
    var report = function () {
      var h = Math.ceil(document.body.getBoundingClientRect().height);
      if (h > 0 && h !== last) {
        last = h;
        window.parent.postMessage({ sr: 'height', height: h }, '*');
      }
    };
    window.addEventListener('message', function (e) {
      if (e.source === window.parent && e.data && e.data.sr === 'theme') setTheme(e.data.theme);
    });
    if (window.ResizeObserver) new ResizeObserver(report).observe(document.body);
    window.addEventListener('load', report);
    window.addEventListener('resize', report);
    report();
  }

  function ready() {
    if (EMBED) startEmbed();
    syncThemeButtons();
    document.addEventListener('click', function (e) {
      var b = e.target.closest ? e.target.closest('[data-sr-theme]') : null;
      if (b) SR.theme.toggle();
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ready);
  else ready();

  window.SR = SR;
})();
