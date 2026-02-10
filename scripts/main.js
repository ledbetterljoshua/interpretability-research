/* ============================================================
   MAIN — Scroll navigation, detail panels, initialization
   ============================================================ */

import { overviewDetails, conceptDetails } from './data.js';
import { initScrollReveals } from './scroll-animations.js';


/* ---- Initialize ---- */

document.addEventListener('DOMContentLoaded', () => {
  initScrollReveals();
  initScrollNav();
});


/* ---- Scroll-based Navigation ---- */

function initScrollNav() {
  const sections = document.querySelectorAll('main > section');
  const navLinks = document.querySelectorAll('.nav-btn');

  if (!sections.length || !navLinks.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          navLinks.forEach((link) => {
            link.classList.toggle(
              'active',
              link.getAttribute('href') === `#${id}`
            );
          });
        }
      });
    },
    {
      rootMargin: '-20% 0px -75% 0px',
      threshold: 0,
    }
  );

  sections.forEach((section) => observer.observe(section));
}


/* ---- Detail Panels ---- */

window.showDetail = function showDetail(key) {
  const d = overviewDetails[key];
  if (!d) return;

  const panel = document.getElementById('detail-panel');
  panel.innerHTML = buildPanelHTML(d);
  panel.classList.add('visible');
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
};

window.showConceptDetail = function showConceptDetail(key) {
  const d = conceptDetails[key];
  if (!d) return;

  const panel = document.getElementById('concept-detail-panel');
  panel.innerHTML = buildPanelHTML(d);
  panel.classList.add('visible');
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
};

function buildPanelHTML(d) {
  return `
    <h3>${d.title}</h3>
    <span class="tag ${d.tag} detail-tag">${d.tagText}</span>
    <p>${d.desc}</p>
    <ul>${d.details.map((x) => `<li>${x}</li>`).join('')}</ul>
    <div class="prereqs">
      <div class="prereqs-label">Prerequisites</div>
      <p>${d.prereqs}</p>
    </div>
  `;
}
