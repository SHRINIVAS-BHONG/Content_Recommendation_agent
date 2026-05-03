/**
 * api.js — Thin fetch wrapper for the Zesty backend.
 *
 * All calls target the FastAPI server running on port 8000.
 * No external HTTP libraries needed — just native fetch.
 */

const API_BASE = 'http://localhost:8000';

/**
 * Get recommendations from the AI agent pipeline.
 * @param {string} query — Natural language query (e.g. "dark anime like Death Note")
 * @param {number} page  — 0-indexed page number for Load More (default 0)
 * @returns {Promise<{results: Array, reasoning_trace: string[], refinement_count: number, page: number, has_more: boolean}>}
 */
export async function getRecommendations(query, page = 0) {
  const res = await fetch(`${API_BASE}/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, page }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

/**
 * Health check — verify backend is reachable.
 * @returns {Promise<{status: string}>}
 */
export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Backend unreachable');
  return res.json();
}
