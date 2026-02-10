/* ============================================================
   SCROLL ANIMATIONS — IntersectionObserver-based reveals
   ============================================================ */

/**
 * Observe all `.reveal` elements and add `.visible` when they
 * enter the viewport. Supports staggered card reveals.
 */
export function initScrollReveals() {
  const reveals = document.querySelectorAll('.reveal');

  if (!reveals.length) return;

  // Immediately reveal elements already in view (header)
  const immediateCheck = () => {
    reveals.forEach((el) => {
      const rect = el.getBoundingClientRect();
      if (rect.top < window.innerHeight * 0.95) {
        el.classList.add('visible');
      }
    });
  };

  // Run once on load after a tiny delay for layout
  setTimeout(immediateCheck, 100);

  // Observer for elements that scroll into view
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.08,
      rootMargin: '0px 0px -40px 0px',
    }
  );

  reveals.forEach((el) => observer.observe(el));
}


/**
 * Re-run reveals for a specific section (called when section becomes active).
 * Adds `.reveal` and triggers observation for cards/items within the section.
 */
export function revealSection(sectionEl) {
  if (!sectionEl) return;

  // Cards
  const cards = sectionEl.querySelectorAll('.card, .roadmap-step, .timeline-item');
  cards.forEach((card, i) => {
    card.classList.add('reveal');
    card.style.setProperty('--stagger', `${i * 0.06}s`);
    // Small delay then make visible with stagger
    setTimeout(() => {
      card.classList.add('visible');
    }, 60 + i * 60);
  });

  // Section header
  const header = sectionEl.querySelector('.section-header');
  if (header) {
    header.classList.add('reveal');
    setTimeout(() => header.classList.add('visible'), 30);
  }

  // Concept maps
  const maps = sectionEl.querySelectorAll('.concept-map');
  maps.forEach((m) => {
    m.classList.add('reveal');
    setTimeout(() => m.classList.add('visible'), 120);
  });
}
