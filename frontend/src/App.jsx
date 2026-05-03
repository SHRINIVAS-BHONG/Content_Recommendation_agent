import { useState, useEffect } from 'react';
import { checkHealth } from './api';
import Hero from './components/Hero/Hero';
import PipelineStory from './components/PipelineStory/PipelineStory';
import RecommendationEngine from './components/RecommendationEngine/RecommendationEngine';
import MemorySection from './components/MemorySection/MemorySection';
import AuthCard from './components/AuthCard/AuthCard';
import WhyThisResult from './components/WhyThisResult/WhyThisResult';
import CTAFooter from './components/CTAFooter/CTAFooter';
import zestyLogo from './assets/zesty-logo.jpg';
import './App.css';

// Kanji characters used in the fixed background
const KANJI = ['炎', '力', '夢', '星', '魂', '剣', '風', '闇'];

export default function App() {
  const [backendOnline, setBackendOnline] = useState(false);

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerTrace, setDrawerTrace] = useState([]);
  const [drawerRefinements, setDrawerRefinements] = useState(0);

  useEffect(() => {
    checkHealth()
      .then(() => setBackendOnline(true))
      .catch(() => setBackendOnline(false));
  }, []);

  const handleOpenTrace = (trace, refinementCount) => {
    setDrawerTrace(trace);
    setDrawerRefinements(refinementCount);
    setDrawerOpen(true);
  };

  const scrollTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <>
      {/* ── Global fixed background — visible on every scroll position ── */}
      <div className="app-bg" aria-hidden="true">
        {/* Ambient radial glows */}
        <div className="app-bg__glow app-bg__glow--1" />
        <div className="app-bg__glow app-bg__glow--2" />
        <div className="app-bg__glow app-bg__glow--3" />

        {/* Halftone dot grid */}
        <div className="app-bg__halftone" />

        {/* Speed lines */}
        <div className="app-bg__speed-lines" />

        {/* Floating kanji */}
        {KANJI.map((k, i) => (
          <div key={i} className={`app-bg__kanji app-bg__kanji--${i + 1}`}>{k}</div>
        ))}

        {/* Ember particles */}
        {Array.from({ length: 16 }).map((_, i) => (
          <div key={i} className="app-bg__particle" />
        ))}

        {/* Corner arcs */}
        <div className="app-bg__arc app-bg__arc--tl" />
        <div className="app-bg__arc app-bg__arc--br" />
        <div className="app-bg__arc app-bg__arc--tr" />
      </div>

      {/* ── Navigation ─────────────────────────────────────────────────── */}
      <nav className="nav">
        <div className="nav__brand">
          <img src={zestyLogo} alt="Zesty" className="nav__brand-logo" />
          <span className="gradient-text">Zesty</span>
        </div>

        <ul className="nav__links">
          <li><button className="nav__link" onClick={() => scrollTo('hero')}>Home</button></li>
          <li><button className="nav__link" onClick={() => scrollTo('pipeline')}>Pipeline</button></li>
          <li><button className="nav__link" onClick={() => scrollTo('recommend')}>Recommend</button></li>
          <li><button className="nav__link" onClick={() => scrollTo('memory')}>Memory</button></li>
          <li><button className="nav__link" onClick={() => scrollTo('auth')}>Auth</button></li>
        </ul>

        <div className="nav__status">
          <span className={`nav__status-dot ${backendOnline ? 'nav__status-dot--online' : ''}`} />
          {backendOnline ? 'API Online' : 'API Offline'}
        </div>
      </nav>

      {/* ── Sections ───────────────────────────────────────────────────── */}
      <Hero />

      <div className="app-diagonal" />
      <PipelineStory />

      <div className="app-diagonal app-diagonal--reverse" />
      <RecommendationEngine onOpenTrace={handleOpenTrace} />

      <div className="app-divider" />
      <MemorySection />

      <div className="app-diagonal" />
      <AuthCard />

      <div className="app-divider" />
      <CTAFooter />

      {/* ── Reasoning Trace Drawer ──────────────────────────────────────── */}
      <WhyThisResult
        isOpen={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        trace={drawerTrace}
        refinementCount={drawerRefinements}
      />
    </>
  );
}
