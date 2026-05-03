import { useEffect } from 'react';
import './WhyThisResult.css';

/**
 * Slide-in drawer that shows the reasoning trace from the agent pipeline.
 * @param {{ isOpen: boolean, onClose: () => void, trace: string[], refinementCount: number }} props
 */
export default function WhyThisResult({ isOpen, onClose, trace = [], refinementCount = 0 }) {
  // Lock body scroll when drawer is open
  useEffect(() => {
    document.body.style.overflow = isOpen ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [isOpen]);

  // Close on Escape key
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    if (isOpen) window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, onClose]);

  /**
   * Highlight [node_name] patterns in trace text for visual emphasis.
   */
  const formatTrace = (text) => {
    // Match patterns like [process_node], [evaluator_node], etc.
    return text.replace(
      /\[([a-z_]+)\]/g,
      '<span class="drawer__trace-node">[$1]</span>'
    );
  };

  return (
    <>
      {/* Overlay */}
      <div
        className={`drawer-overlay ${isOpen ? 'open' : ''}`}
        onClick={onClose}
      />

      {/* Drawer panel */}
      <aside className={`drawer ${isOpen ? 'open' : ''}`}>
        <div className="drawer__header">
          <h2 className="drawer__title">🧠 Why These Picks?</h2>
          <button className="drawer__close" onClick={onClose}>✕</button>
        </div>

        <div className="drawer__body">
          {/* Refinement badge */}
          {refinementCount > 0 && (
            <div className="drawer__refinement">
              🔄 {refinementCount} refinement cycle{refinementCount > 1 ? 's' : ''} applied
            </div>
          )}

          {/* Trace steps */}
          {trace.length > 0 ? (
            <div className="drawer__trace-list">
              {trace.map((step, i) => (
                <div key={i} className="drawer__trace-step">
                  <span className="drawer__trace-num">{i + 1}</span>
                  <span dangerouslySetInnerHTML={{ __html: formatTrace(step) }} />
                </div>
              ))}
            </div>
          ) : (
            <div className="drawer__empty">
              No reasoning trace available yet.<br />
              Try making a recommendation query first!
            </div>
          )}
        </div>
      </aside>
    </>
  );
}
