import './MemorySection.css';

const FEATURES = [
  {
    icon: '🧠',
    title: 'Preference Learning',
    desc: 'Uses Exponential Moving Average (EMA) to learn your taste from ratings — boosting genres you love, suppressing ones you avoid.',
  },
  {
    icon: '💬',
    title: 'Conversation Context',
    desc: 'Remembers your recent 5 queries and results, so recommendations get smarter with every interaction.',
  },
  {
    icon: '🔒',
    title: 'Privacy-First Storage',
    desc: 'Built-in privacy manager with retention policies, audit logs, and GDPR-compliant data deletion. You stay in control.',
  },
  {
    icon: '⚡',
    title: 'Personalization Weights',
    desc: 'Preferred genres get a 1.5× boost, avoided genres get 0.3× suppression — applied in real-time during recommendation scoring.',
  },
];

const BUBBLES = [
  'Action ★★★★★',
  'Drama ★★★★',
  'Psychological ★★★★★',
  'Romance ★★★',
  'Sci-Fi ★★★★',
  'Horror ★★',
  'Fantasy ★★★★★',
];

export default function MemorySection() {
  return (
    <section className="memory" id="memory">
      <div className="memory__content">
        {/* Left: features */}
        <div className="memory__text">
          <p className="section-label">Personalization</p>
          <h2 className="section-title">
            It <span className="gradient-text">Learns Your Taste</span>
          </h2>
          <p className="section-subtitle">
            Sign in once, and the agent builds a preference profile that evolves
            with every interaction. No manual setup required.
          </p>

          <ul className="memory__features">
            {FEATURES.map((f, i) => (
              <li key={i} className="memory__feature">
                <div className="memory__feature-icon">{f.icon}</div>
                <div>
                  <h4 className="memory__feature-title">{f.title}</h4>
                  <p className="memory__feature-desc">{f.desc}</p>
                </div>
              </li>
            ))}
          </ul>
        </div>

        {/* Right: floating preference bubbles */}
        <div className="memory__visual">
          <div className="memory__bubble-container">
            {BUBBLES.map((b, i) => (
              <div key={i} className={`memory__bubble memory__bubble--${i + 1}`}>
                {b}
              </div>
            ))}
            <div className="memory__brain">🧠</div>
          </div>
        </div>
      </div>
    </section>
  );
}
