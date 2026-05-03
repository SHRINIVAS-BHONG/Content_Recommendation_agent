import { useState, useEffect, useRef } from 'react';
import './ThinkingAnimation.css';

const STEPS = [
  { icon: '🔍', text: 'Analyzing your query…' },
  { icon: '🧠', text: 'Expanding semantic intent…' },
  { icon: '⚡', text: 'Ranking 26,000+ titles…' },
  { icon: '📊', text: 'Evaluating result quality…' },
  { icon: '🎯', text: 'Final picks ready!' },
];

/**
 * Displays animated sequential AI thinking steps.
 * Waits for resultsReady before calling onComplete so the animation
 * never races ahead of the actual API response.
 */
export default function ThinkingAnimation({ isActive, refinementCount = 0, resultsReady = false, onComplete }) {
  const [currentStep, setCurrentStep] = useState(-1);
  const [animDone, setAnimDone] = useState(false);

  // Reset when a new search starts
  useEffect(() => {
    if (!isActive) {
      setCurrentStep(-1);
      setAnimDone(false);
      return;
    }

    setAnimDone(false);
    setCurrentStep(0);
    let step = 0;
    const totalSteps = STEPS.length + (refinementCount > 0 ? 1 : 0);

    const interval = setInterval(() => {
      step += 1;
      if (step >= totalSteps) {
        clearInterval(interval);
        setCurrentStep(totalSteps - 1); // stay on last step
        setAnimDone(true);
      } else {
        setCurrentStep(step);
      }
    }, 700);

    return () => clearInterval(interval);
  }, [isActive, refinementCount]);

  // Fire onComplete when BOTH animation is done AND results are ready
  useEffect(() => {
    if (animDone && resultsReady) {
      const t = setTimeout(() => onComplete?.(), 300);
      return () => clearTimeout(t);
    }
  }, [animDone, resultsReady]);

  if (!isActive && currentStep === -1) return null;

  const steps = [...STEPS];
  if (refinementCount > 0) {
    steps.splice(4, 0, { icon: '🔄', text: `Refining results (cycle ${refinementCount})…` });
  }

  const waiting = animDone && !resultsReady;

  return (
    <div className="thinking">
      {steps.map((step, i) => {
        let className = 'thinking__step';
        if (i < currentStep) className += ' done';
        else if (i === currentStep) className += ' active';

        return (
          <div key={i} className={className}>
            {i < currentStep ? (
              <span className="thinking__check">✓</span>
            ) : i === currentStep ? (
              <span className={waiting && i === steps.length - 1 ? 'thinking__spinner' : 'thinking__spinner'} />
            ) : null}
            <span>{step.icon} {step.text}</span>
          </div>
        );
      })}

      {waiting && (
        <div className="thinking__waiting">
          <span className="thinking__spinner" />
          <span>Fetching recommendations…</span>
        </div>
      )}
    </div>
  );
}
