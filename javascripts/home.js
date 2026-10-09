// Homepage behavior: EN/中文 toggle, scroll reveal, active nav link.
// The initial language is chosen by the inline script in <head> before first paint.
(function () {
  var root = document.documentElement;
  var TITLES = {
    'en': 'Xiaohui Yan (晏小辉) — Self-Evolving Agents · Deep Research · Search',
    'zh-CN': '晏小辉 Xiaohui Yan — 自进化智能体 · 深度研究 · 智能搜索'
  };
  var ARIA = { 'en': '切换到中文', 'zh-CN': 'Switch to English' };

  function applyLang(lang, save) {
    root.lang = lang;
    document.title = TITLES[lang];
    var btn = document.getElementById('lang-toggle');
    if (btn) btn.setAttribute('aria-label', ARIA[lang]);
    if (save) { try { localStorage.setItem('lang', lang); } catch (e) {} }
  }

  function initToggle() {
    applyLang(root.lang === 'zh-CN' ? 'zh-CN' : 'en', false);
    document.getElementById('lang-toggle').addEventListener('click', function () {
      applyLang(root.lang === 'zh-CN' ? 'en' : 'zh-CN', true);
    });
  }

  function initReveal() {
    var items = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window)) {
      for (var i = 0; i < items.length; i++) items[i].classList.add('in');
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('in');
          io.unobserve(entry.target);
        }
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });
    items.forEach(function (el) { io.observe(el); });
  }

  function initScrollSpy() {
    if (!('IntersectionObserver' in window)) return;
    var links = {};
    document.querySelectorAll('.nav-links a').forEach(function (a) {
      links[a.getAttribute('href').slice(1)] = a;
    });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var link = links[entry.target.id];
        if (!link || !entry.isIntersecting) return;
        Object.keys(links).forEach(function (id) {
          links[id].classList.remove('active');
          links[id].removeAttribute('aria-current');
        });
        link.classList.add('active');
        link.setAttribute('aria-current', 'true');
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    Object.keys(links).forEach(function (id) {
      var section = document.getElementById(id);
      if (section) io.observe(section);
    });
  }

  initToggle();
  initReveal();
  initScrollSpy();
})();
