import './Hero.css';
import zestyLogo from '../../assets/zesty-logo.jpg';

const SAMPLE_CARDS = [
  {
    title: 'Attack on Titan',
    genres: ['Action', 'Dark Fantasy', 'Drama'],
    score: 9.0,
    image: 'https://cdn.myanimelist.net/images/anime/10/47347.jpg',
  },
  {
    title: 'Steins;Gate',
    genres: ['Sci-Fi', 'Thriller', 'Drama'],
    score: 9.1,
    image: 'https://cdn.myanimelist.net/images/anime/5/73199.jpg',
  },
  {
    title: 'Vinland Saga',
    genres: ['Action', 'Historical', 'Adventure'],
    score: 8.8,
    image: 'https://cdn.myanimelist.net/images/anime/1500/103005.jpg',
  },
];

export default function Hero() {
  const scrollTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section className="hero" id="hero">
      {/* ── Background layers ─────────────────────────────────────────── */}
      {/* Global bg (kanji, particles, speed lines) is in App.jsx fixed layer */}
      <div className="hero__bg">
        <div className="hero__bg-gradient" />
      </div>

      {/* ── Content ───────────────────────────────────────────────────── */}
      <div className="hero__content">
        {/* Left: text */}
        <div className="hero__text">
          <div className="hero__badge">
            <img src={zestyLogo} alt="Zesty" className="hero__badge-logo" />
            AI-Powered • 7-Node Pipeline
          </div>

          <h1 className="hero__title">
            Your{' '}
            <span className="gradient-text">Intelligent</span>
            <br />
            Anime &amp; Manga Oracle
          </h1>

          <p className="hero__subtitle">
            A self-refining recommendation agent that understands your taste, reasons about
            26,000+ titles, and explains every pick. Powered by Sentence-BERT, LangGraph,
            and a quality-evaluation loop.
          </p>

          <div className="hero__cta-group">
            <button className="btn-primary" onClick={() => scrollTo('recommend')}>
              🎯 Get Recommendations
            </button>
            <button className="btn-secondary" onClick={() => scrollTo('pipeline')}>
              ⚙️ See How It Works
            </button>
          </div>
        </div>

        {/* Right: animated card stack */}
        <div className="hero__cards">
          <div className="hero__card-stack">
            {SAMPLE_CARDS.map((card, i) => (
              <div key={i} className={`hero__card hero__card--${i}`}>
                {card.image ? (
                  <img src={card.image} alt={card.title} style={{ width: '100%', height: '100%', objectFit: 'cover', position: 'absolute', inset: 0, zIndex: 0 }} />
                ) : (
                  <div className="hero__card-bg" />
                )}
                <span className="hero__card-score">★ {card.score}</span>
                <div className="hero__card-inner">
                  <h3 className="hero__card-title">{card.title}</h3>
                  <div className="hero__card-genre">
                    {card.genres.map((g) => (
                      <span key={g} className="hero__card-pill">{g}</span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
