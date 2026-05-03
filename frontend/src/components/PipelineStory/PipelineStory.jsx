import { useEffect, useRef } from 'react';
import './PipelineStory.css';

const PIPELINE_NODES = [
  {
    icon: '🔍',
    title: 'Process Query',
    node: 'process_node',
    desc: 'Parses your natural language query to extract tags, detect anime/manga type, identify reference titles, classify intent, and compute a complexity score.',
  },
  {
    icon: '🧠',
    title: 'Reasoning Engine',
    node: 'simple_reasoning_node / deep_reasoning_node',
    desc: 'Routes to simple or deep reasoning based on complexity. Expands tags via co-occurrence data, translates semantic hints, and looks up reference synopses.',
  },
  {
    icon: '⚡',
    title: 'Recommend',
    node: 'recommend_node',
    desc: 'Loads the pre-trained Sentence-BERT model, computes semantic + Jaccard + popularity scores, applies personalization weights, and ranks 26,000+ titles.',
  },
  {
    icon: '📊',
    title: 'Evaluate Quality',
    node: 'evaluator_node',
    desc: 'Self-reflection step: measures coverage, diversity, and average score. Decides whether results meet quality thresholds or need refinement.',
  },
  {
    icon: '🔄',
    title: 'Refine (Loop)',
    node: 'refine_node',
    desc: 'If quality is low, applies 3 strategies: add uncovered tags, switch search strategy, or relax strictness. Loops back to recommend (max 2 cycles).',
  },
  {
    icon: '✨',
    title: 'Output',
    node: 'output_node',
    desc: 'Normalizes the final results into the response schema with title, image, synopsis, score, genres, similarity score, and match reason.',
  },
];

export default function PipelineStory() {
  const nodesRef = useRef([]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
          }
        });
      },
      { threshold: 0.3 }
    );

    nodesRef.current.forEach((el) => {
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  return (
    <section className="pipeline" id="pipeline">
      <div className="pipeline__header">
        <p className="section-label">How It Works</p>
        <h2 className="section-title">
          A <span className="gradient-text">7-Node AI Pipeline</span>
          <br />That Thinks Before It Recommends
        </h2>
        <p className="section-subtitle" style={{ margin: '0 auto' }}>
          Your query flows through a LangGraph pipeline that parses, reasons, recommends,
          evaluates, and self-refines — producing explainable, high-quality picks.
        </p>
      </div>

      <div className="pipeline__timeline">
        <div className="pipeline__line" />
        {PIPELINE_NODES.map((node, i) => (
          <div
            key={i}
            className="pipeline__node"
            ref={(el) => (nodesRef.current[i] = el)}
          >
            <div className="pipeline__dot" />
            <div className="pipeline__card">
              <span className="pipeline__icon">{node.icon}</span>
              <h3 className="pipeline__card-title">{node.title}</h3>
              <p className="pipeline__card-node">{node.node}</p>
              <p className="pipeline__card-desc">{node.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
