import { useState, useEffect, useRef } from 'react';
import './CTAFooter.css';

const STATS = [
  { value: 26000, suffix: '+', label: 'Titles in Database', color: 'violet' },
  { value: 7, suffix: '', label: 'Pipeline Nodes', color: 'cyan' },
  { value: 3, suffix: '', label: 'AI Techniques', color: 'pink' },
  { value: 2, suffix: '', label: 'Refinement Cycles', color: 'gold' },
];

const TECH = ['FastAPI', 'LangGraph', 'Sentence-BERT', 'React', 'Google OAuth', 'JWT RS256'];

/**
 * Animated counter hook — counts up from 0 to target when visible.
 */
function useCountUp(target, duration = 1500) {
  const [count, setCount] = useState(0);
  const ref = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          const start = performance.now();
          const animate = (now) => {
            const progress = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
            setCount(Math.floor(eased * target));
            if (progress < 1) requestAnimationFrame(animate);
          };
          requestAnimationFrame(animate);
          observer.disconnect();
        }
      },
      { threshold: 0.5 }
    );

    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [target, duration]);

  return [count, ref];
}

function StatItem({ value, suffix, label, color }) {
  const [count, ref] = useCountUp(value);
  return (
    <div className="cta-footer__stat" ref={ref}>
      <div className={`cta-footer__stat-value cta-footer__stat-value--${color}`}>
        {count.toLocaleString()}{suffix}
      </div>
      <div className="cta-footer__stat-label">{label}</div>
    </div>
  );
}

export default function CTAFooter() {
  const scrollTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <footer className="cta-footer" id="footer">
      {/* ── Stats ─────────────────────────────────────────────────────── */}
      <div className="cta-footer__stats">
        {STATS.map((s, i) => (
          <StatItem key={i} {...s} />
        ))}
      </div>

      {/* ── CTA ───────────────────────────────────────────────────────── */}
      <div className="cta-footer__cta">
        <h2>
          Ready to Find Your Next{' '}
          <span className="gradient-text">Obsession</span>?
        </h2>
        <p>
          Tell the agent what you're in the mood for. It'll handle the rest.
        </p>
        <button className="btn-primary" onClick={() => scrollTo('recommend')}>
          🎯 Start Discovering
        </button>
      </div>

      {/* ── Bottom bar ────────────────────────────────────────────────── */}
      <div className="cta-footer__bottom">
        <span className="cta-footer__brand">
          <span className="gradient-text">Zesty</span> Recommender
        </span>
        <div className="cta-footer__tech">
          {TECH.map((t) => (
            <span key={t} className="cta-footer__tech-badge">{t}</span>
          ))}
        </div>
      </div>
    </footer>
  );
}
