import { useState } from 'react';
import './ResultCard.css';

/**
 * A single recommendation result card with expandable synopsis.
 * @param {{ item: object, index: number, onWhyClick: () => void }} props
 */
export default function ResultCard({ item, index, onWhyClick }) {
  const [expanded, setExpanded] = useState(false);
  const [imgError, setImgError] = useState(false);

  const hasImage = item.image && item.image !== '' && !item.image.includes('null') && !imgError;

  // Synopsis is already truncated to 250 chars by the backend.
  // We show a short preview (120 chars) and let the user expand to see the full 250.
  const PREVIEW_LENGTH = 120;
  const synopsis = item.synopsis || 'No synopsis available.';
  const isLong = synopsis.length > PREVIEW_LENGTH;
  const displayedSynopsis = expanded || !isLong
    ? synopsis
    : synopsis.slice(0, PREVIEW_LENGTH).trimEnd() + '…';

  return (
    <div
      className="result-card"
      style={{ animationDelay: `${index * 0.1}s` }}
    >
      {/* ── Image ─────────────────────────────────────────────────────── */}
      <div className="result-card__image-wrap">
        {hasImage ? (
          <img
            className="result-card__image"
            src={item.image}
            alt={item.title}
            loading="lazy"
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="result-card__no-image">🎬</div>
        )}
        <div className="result-card__image-gradient" />

        {item.score > 0 && (
          <span className="result-card__score">★ {item.score.toFixed(1)}</span>
        )}
        {item.similarity_score != null && (
          <span className="result-card__similarity">
            {(item.similarity_score * 100).toFixed(0)}% match
          </span>
        )}
      </div>

      {/* ── Body ──────────────────────────────────────────────────────── */}
      <div className="result-card__body">
        <h3 className="result-card__title">{item.title}</h3>

        {item.genres?.length > 0 && (
          <div className="result-card__genres">
            {item.genres.slice(0, 5).map((g) => (
              <span key={g} className="result-card__genre-pill">{g}</span>
            ))}
          </div>
        )}

        {/* Synopsis with expand/collapse */}
        <div className="result-card__synopsis-wrap">
          <p className="result-card__synopsis">{displayedSynopsis}</p>
          {isLong && (
            <button
              className="result-card__expand-btn"
              onClick={() => setExpanded((v) => !v)}
            >
              {expanded ? '▲ Show less' : '▼ Read more'}
            </button>
          )}
        </div>

        {item.match_reason && (
          <p className="result-card__match">💡 {item.match_reason}</p>
        )}

        <button className="result-card__why-btn" onClick={onWhyClick}>
          🔎 Why this pick?
        </button>
      </div>
    </div>
  );
}
