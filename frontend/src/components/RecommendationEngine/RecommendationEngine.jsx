import { useState } from 'react';
import { getRecommendations } from '../../api';
import ThinkingAnimation from './ThinkingAnimation';
import ResultCard from './ResultCard';
import './RecommendationEngine.css';

const EXAMPLES = [
  'dark anime like Death Note',
  'wholesome romance manga',
  'epic fantasy like Berserk',
  'funny slice of life anime',
  'psychological thriller manga',
  'sad anime like Your Lie in April',
];

export default function RecommendationEngine({ onOpenTrace }) {
  const [query, setQuery] = useState('');
  const [activeQuery, setActiveQuery] = useState(''); // the query currently shown
  const [mediaType, setMediaType] = useState('anime');
  const [isLoading, setIsLoading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [results, setResults] = useState(null);
  const [resultsReady, setResultsReady] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const [currentPage, setCurrentPage] = useState(0);
  const [reasoningTrace, setReasoningTrace] = useState([]);
  const [refinementCount, setRefinementCount] = useState(0);
  const [error, setError] = useState('');

  const buildFullQuery = (q) => {
    const lc = q.toLowerCase();
    if (!lc.includes('anime') && !lc.includes('manga')) {
      return `${q} ${mediaType}`;
    }
    return q;
  };

  const handleSearch = async (searchQuery) => {
    const q = (searchQuery || query).trim();
    if (!q) return;

    const fullQuery = buildFullQuery(q);

    setError('');
    setResults(null);
    setResultsReady(false);
    setHasMore(false);
    setCurrentPage(0);
    setActiveQuery(fullQuery);
    setIsLoading(true);
    setIsThinking(true);

    try {
      const data = await getRecommendations(fullQuery, 0);
      setReasoningTrace(data.reasoning_trace || []);
      setRefinementCount(data.refinement_count || 0);
      setResults(data.results || []);
      setHasMore(data.has_more ?? false);
      setCurrentPage(0);
      setResultsReady(true);  // signal animation that data is here
    } catch (err) {
      setError(err.message || 'Something went wrong. Is the backend running?');
      setResultsReady(false);
      setIsThinking(false);
      setIsLoading(false);
    }
  };

  const handleLoadMore = async () => {
    if (!activeQuery || isLoadingMore) return;
    const nextPage = currentPage + 1;
    setIsLoadingMore(true);
    setError('');

    try {
      const data = await getRecommendations(activeQuery, nextPage);
      setResults((prev) => [...(prev || []), ...(data.results || [])]);
      setHasMore(data.has_more ?? false);
      setCurrentPage(nextPage);
    } catch (err) {
      setError(err.message || 'Failed to load more results.');
    } finally {
      setIsLoadingMore(false);
    }
  };

  const handleThinkingComplete = () => {
    setIsThinking(false);
    setIsLoading(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleSearch();
  };

  const handleExampleClick = (example) => {
    setQuery(example);
    handleSearch(example);
  };

  return (
    <section className="rec-engine" id="recommend">
      <div className="rec-engine__header">
        <p className="section-label">Try It Live</p>
        <h2 className="section-title">
          <span className="gradient-text">Ask the Agent</span>
        </h2>
        <p className="section-subtitle" style={{ margin: '0 auto' }}>
          Type a natural language query and watch the AI pipeline think, reason, and recommend.
        </p>
      </div>

      {/* ── Search bar ────────────────────────────────────────────────── */}
      <div className="rec-engine__search">
        <div className="rec-engine__toggle">
          <button
            className={`rec-engine__toggle-btn ${mediaType === 'anime' ? 'active' : ''}`}
            onClick={() => setMediaType('anime')}
          >
            🎬 Anime
          </button>
          <button
            className={`rec-engine__toggle-btn ${mediaType === 'manga' ? 'active' : ''}`}
            onClick={() => setMediaType('manga')}
          >
            📖 Manga
          </button>
        </div>

        <div className="rec-engine__input-wrap">
          <input
            className="rec-engine__input"
            type="text"
            placeholder={`Try "dark ${mediaType} like Death Note" or "wholesome romance"…`}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
          />
          <button
            className="rec-engine__search-btn"
            onClick={() => handleSearch()}
            disabled={isLoading || !query.trim()}
          >
            {isLoading ? '⏳ Thinking…' : '🎯 Recommend'}
          </button>
        </div>
      </div>

      {/* ── Example queries ───────────────────────────────────────────── */}
      {!results && !isLoading && (
        <div className="rec-engine__examples">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              className="rec-engine__example"
              onClick={() => handleExampleClick(ex)}
            >
              {ex}
            </button>
          ))}
        </div>
      )}

      {/* ── Error ─────────────────────────────────────────────────────── */}
      {error && <div className="rec-engine__error">⚠️ {error}</div>}

      {/* ── AI Thinking animation ─────────────────────────────────────── */}
      {isThinking && (
        <ThinkingAnimation
          isActive={isThinking}
          refinementCount={refinementCount}
          resultsReady={resultsReady}
          onComplete={handleThinkingComplete}
        />
      )}

      {/* ── Results ───────────────────────────────────────────────────── */}
      {results && !isThinking && results.length > 0 && (
        <>
          <div className="rec-engine__meta">
            <div className="rec-engine__meta-item">
              📊 Results: <span className="rec-engine__meta-value">{results.length}</span>
            </div>
            <div className="rec-engine__meta-item">
              🔄 Refinements: <span className="rec-engine__meta-value">{refinementCount}</span>
            </div>
            <div className="rec-engine__meta-item">
              🧠 Trace steps: <span className="rec-engine__meta-value">{reasoningTrace.length}</span>
            </div>
          </div>

          <div className="rec-engine__results">
            {results.map((item, i) => (
              <ResultCard
                key={item.title + i}
                item={item}
                index={i}
                onWhyClick={() => onOpenTrace(reasoningTrace, refinementCount)}
              />
            ))}
          </div>

          {/* ── Load More ─────────────────────────────────────────────── */}
          {hasMore && (
            <div className="rec-engine__load-more">
              <button
                className="rec-engine__load-more-btn"
                onClick={handleLoadMore}
                disabled={isLoadingMore}
              >
                {isLoadingMore ? '⏳ Loading…' : '⬇ Load More'}
              </button>
            </div>
          )}
        </>
      )}

      {/* ── No results ─────────────────────────────────────────────────── */}
      {results && !isThinking && results.length === 0 && (
        <div className="rec-engine__error">
          No results found. Try a different query!
        </div>
      )}
    </section>
  );
}
