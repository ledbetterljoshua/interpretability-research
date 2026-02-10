/* ============================================================
   TRANSFORMER VISUALS — Stepper logic & diagram interactions
   ============================================================ */

import { initScrollReveals } from './scroll-animations.js';


/* ---- Initialize ---- */

document.addEventListener('DOMContentLoaded', () => {
  initScrollReveals();
  initSteppers();
  initScrollNav();
});


/* ---- Steppers ---- */

function initSteppers() {
  document.querySelectorAll('.stepper').forEach((stepper) => {
    const tabs = stepper.querySelectorAll('.stepper-tab');
    const steps = stepper.querySelectorAll('.stepper-step');
    const prevBtn = stepper.querySelector('.step-prev');
    const nextBtn = stepper.querySelector('.step-next');
    const indicator = stepper.querySelector('.stepper-indicator');
    const total = steps.length;

    function goToStep(idx) {
      idx = Math.max(0, Math.min(idx, total - 1));
      tabs.forEach((t, i) => t.classList.toggle('active', i === idx));
      steps.forEach((s, i) => s.classList.toggle('active', i === idx));
      if (prevBtn) prevBtn.disabled = idx === 0;
      if (nextBtn) nextBtn.disabled = idx === total - 1;
      if (indicator) indicator.textContent = `${idx + 1} / ${total}`;
      stepper.dataset.currentStep = idx;
    }

    tabs.forEach((tab, i) => {
      tab.addEventListener('click', () => goToStep(i));
    });

    if (prevBtn) {
      prevBtn.addEventListener('click', () => {
        goToStep((parseInt(stepper.dataset.currentStep) || 0) - 1);
      });
    }

    if (nextBtn) {
      nextBtn.addEventListener('click', () => {
        goToStep((parseInt(stepper.dataset.currentStep) || 0) + 1);
      });
    }

    // Initialize to first step
    goToStep(0);
  });
}


/* ---- Scroll-based Navigation ---- */

function initScrollNav() {
  const sections = document.querySelectorAll('.curriculum-section');
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
