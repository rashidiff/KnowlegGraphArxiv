'use client';

import { useEffect, useRef, useState } from 'react';
import { apiUrl } from '@/lib/api';

export interface PathPaper {
  id: string;
  title: string;
  year?: number;
}

interface Props {
  papers: PathPaper[];
  onPath: (path: PathPaper[] | null) => void;
  onSelectPaper: (id: string) => void;
}

export default function CitationPathExplorer({ papers, onPath, onSelectPaper }: Props) {
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [path, setPath] = useState<PathPaper[] | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const request = useRef<AbortController | null>(null);

  useEffect(() => () => request.current?.abort(), []);

  const clear = () => {
    request.current?.abort();
    request.current = null;
    setBusy(false);
    setError('');
    setPath(null);
    onPath(null);
  };

  const findPath = async () => {
    clear();
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    try {
      const params = new URLSearchParams({ start_id: start, end_id: end });
      const response = await fetch(apiUrl(`/api/graph/path?${params}`), { signal: controller.signal });
      if (!response.ok) throw new Error('Could not find a path. Please try again.');
      const result = await response.json();
      if (controller.signal.aborted) return;
      setPath(result.path);
      onPath(result.found ? result.path : null);
    } catch (err) {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Path lookup failed.');
    } finally {
      if (request.current === controller) setBusy(false);
    }
  };

  return (
    <div className="px-4 py-2 border-b border-border bg-card text-xs space-y-2">
      <div className="flex items-center gap-2">
        <span className="font-medium shrink-0">Connect papers</span>
        <select aria-label="Start paper" value={start} onChange={e => { clear(); setStart(e.target.value); }} className="min-w-0 flex-1 bg-background border border-border rounded p-1">
          <option value="">From…</option>
          {papers.map(p => <option key={p.id} value={p.id}>{p.title}</option>)}
        </select>
        <select aria-label="End paper" value={end} onChange={e => { clear(); setEnd(e.target.value); }} className="min-w-0 flex-1 bg-background border border-border rounded p-1">
          <option value="">To…</option>
          {papers.map(p => <option key={p.id} value={p.id}>{p.title}</option>)}
        </select>
        <button type="button" disabled={!start || !end || busy} onClick={findPath} className="text-primary disabled:opacity-40 shrink-0">{busy ? 'Finding…' : 'Find path'}</button>
        {path !== null && <button type="button" onClick={clear}>Clear</button>}
      </div>
      <p className="text-foreground/60">Uses indexed citations in either direction; connections are not necessarily a chronological reading order.</p>
      <div aria-live="polite" className="max-h-24 overflow-y-auto">
        {error && <p role="alert" className="text-rose-400">{error}</p>}
        {path?.length === 0 && <p>No connection found within the indexed corpus and database traversal limits.</p>}
        {path && path.length > 0 && <>
          <p className="text-primary mb-1">{path.length - 1} citation {path.length === 2 ? 'connection' : 'connections'}</p>
          <ol className="list-decimal pl-5 space-y-1">
            {path.map(p => <li key={p.id}><button type="button" className="text-left hover:text-primary" onClick={() => onSelectPaper(p.id)}>{p.title}{p.year ? ` (${p.year})` : ''}</button></li>)}
          </ol>
        </>}
      </div>
    </div>
  );
}
