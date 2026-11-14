/* site.js
   The small amount of behavior the book's pages need: the theme switch, the
   contents drawer on narrow screens, the "you are here" marker in the list of
   sections, copy buttons on listings, and sizing the embedded interactive
   pages. No libraries. */
(function () {
  'use strict';

  var root = document.documentElement;
  var THEME_KEY = 'sr-theme'; // shared with interactive/shared.js
  var darkQuery = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

  var SUN = '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.25" stroke-linecap="round" aria-hidden="true"><circle cx="8" cy="8" r="2.75"/><path d="M8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.15 1.15M11.45 11.45l1.15 1.15M3.4 12.6l1.15-1.15M11.45 4.55l1.15-1.15"/></svg>';
  var MOON = '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M13.2 9.6A5.6 5.6 0 0 1 6.4 2.8a5.6 5.6 0 1 0 6.8 6.8Z"/></svg>';

  /* ---- Theme ------------------------------------------------------------ */

  function currentTheme() {
    var t = root.getAttribute('data-theme');
    if (t === 'light' || t === 'dark') return t;
    return darkQuery && darkQuery.matches ? 'dark' : 'light';
  }

  var frames = Array.prototype.slice.call(document.querySelectorAll('.play iframe'));

  function tellFrames(theme) {
    frames.forEach(function (f) {
      try { f.contentWindow.postMessage({ sr: 'theme', theme: theme }, '*'); } catch (e) { /* not loaded yet */ }
    });
  }

  var toggle = document.querySelector('.theme-toggle');
  function syncToggle() {
    if (!toggle) return;
    var dark = currentTheme() === 'dark';
    toggle.innerHTML = dark ? SUN : MOON;
    toggle.setAttribute('aria-label', dark ? 'Switch to the light theme' : 'Switch to the dark theme');
    toggle.setAttribute('title', dark ? 'Light theme' : 'Dark theme');
  }
  if (toggle) {
    toggle.addEventListener('click', function () {
      var next = currentTheme() === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { window.localStorage.setItem(THEME_KEY, next); } catch (e) { /* private mode: fine */ }
      syncToggle();
      tellFrames(next);
    });
  }
  if (darkQuery && darkQuery.addEventListener) darkQuery.addEventListener('change', syncToggle);
  syncToggle();

  /* ---- Embedded interactive pages ---------------------------------------
     Each one reports its own height (see the embed section of shared.js), so
     the frame is exactly as tall as the page inside it and never scrolls. */

  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.sr !== 'height' || !(d.height > 0)) return;
    for (var i = 0; i < frames.length; i++) {
      if (frames[i].contentWindow === e.source) {
        frames[i].style.height = Math.ceil(d.height) + 2 + 'px';
        break;
      }
    }
  });

  /* ---- Contents drawer (narrow screens) --------------------------------- */

  var sidebar = document.getElementById('sidebar');
  var menu = document.querySelector('.menu-button');
  function setDrawer(open) {
    if (!sidebar || !menu) return;
    sidebar.classList.toggle('open', open);
    menu.setAttribute('aria-expanded', open ? 'true' : 'false');
  }
  if (sidebar && menu) {
    menu.addEventListener('click', function () { setDrawer(!sidebar.classList.contains('open')); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') setDrawer(false); });
    document.addEventListener('click', function (e) {
      if (!sidebar.classList.contains('open')) return;
      var inside = sidebar.contains(e.target) || menu.contains(e.target);
      if (!inside || (e.target.closest && e.target.closest('.sidebar a'))) setDrawer(false);
    });
  }

  // Keep the current chapter in view in the side list.
  var here = sidebar && sidebar.querySelector('.here');
  if (here && sidebar.scrollHeight > sidebar.clientHeight) {
    sidebar.scrollTop = Math.max(0, here.offsetTop - 120);
  }

  /* ---- Which section is on screen --------------------------------------- */

  var sectionLinks = Array.prototype.slice.call(document.querySelectorAll('.toc-sections a'));
  var sections = sectionLinks.map(function (a) {
    return document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1)));
  });
  var ticking = false;
  function markSection() {
    ticking = false;
    var line = 96; // a little below the top bar
    var active = -1;
    for (var i = 0; i < sections.length; i++) {
      if (sections[i] && sections[i].getBoundingClientRect().top <= line) active = i;
    }
    for (var j = 0; j < sectionLinks.length; j++) {
      sectionLinks[j].classList.toggle('active', j === active);
    }
  }
  if (sections.length) {
    window.addEventListener('scroll', function () {
      if (!ticking) { ticking = true; window.requestAnimationFrame(markSection); }
    }, { passive: true });
    markSection();
  }

  /* ---- Copy buttons on Rust listings ------------------------------------ */

  if (navigator.clipboard && navigator.clipboard.writeText) {
    Array.prototype.forEach.call(document.querySelectorAll('.code-rust, .code-toml'), function (block) {
      var pre = block.querySelector('pre');
      if (!pre) return;
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'copy';
      button.textContent = 'Copy';
      button.addEventListener('click', function () {
        navigator.clipboard.writeText(pre.textContent.replace(/\n$/, '')).then(function () {
          button.textContent = 'Copied';
          button.classList.add('done');
          window.setTimeout(function () { button.textContent = 'Copy'; button.classList.remove('done'); }, 1400);
        });
      });
      block.appendChild(button);
    });
  }
})();
