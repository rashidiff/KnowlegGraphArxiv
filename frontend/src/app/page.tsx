'use client';

import React, { useState, useEffect, useRef } from 'react';
import Chat from '@/components/Chat';
import GraphView from '@/components/GraphView';
import EvidencePanel from '@/components/EvidencePanel';
import { BookOpen, AlertCircle, FileUp, Sparkles, RefreshCw, Clock, Trash2, Sun, Moon } from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  isClarifying?: boolean;
}

export default function Home() {
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedPaperId, setSelectedPaperId] = useState<string | null>(null);
  
  // Graph and Evidence state
  const [graphData, setGraphData] = useState<{ nodes: any[]; links: any[] }>({ nodes: [], links: [] });
  const [graphContext, setGraphContext] = useState<any>({});
  const [retrievedPapers, setRetrievedPapers] = useState<any[]>([]);

  // Clarification state
  const [clarifyingQuestion, setClarifyingQuestion] = useState<string | null>(null);
  const [clarificationAnswers, setClarificationAnswers] = useState<any[]>([]);
  const [originalQuery, setOriginalQuery] = useState<string>('');

  // PDF upload reference
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadStatus, setUploadStatus] = useState<{ type: 'success' | 'error' | 'loading' | 'warning' | null; message: string }>({ type: null, message: '' });

  // arXiv sync state
  const [syncRunning, setSyncRunning] = useState(false);
  const [corpusStatus, setCorpusStatus] = useState<{ paper_count?: number; last_updated?: string; last_update_added?: number } | null>(null);
  const syncPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load corpus status on mount (but NOT the graph — graph only builds after a query)
  useEffect(() => {
    fetchCorpusStatus();
  }, []);

  const fetchInitialGraph = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/graph/explore');
      if (res.ok) {
        const data = await res.json();
        setGraphData(data);
      }
    } catch (e) {
      console.error("Error fetching initial graph:", e);
    }
  };

  const fetchCorpusStatus = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/corpus/status');
      if (res.ok) {
        const data = await res.json();
        setCorpusStatus(data);
        if (data.refresh_running) {
          setSyncRunning(true);
        } else if (syncRunning) {
          // Refresh just finished
          setSyncRunning(false);
          if (syncPollRef.current) clearInterval(syncPollRef.current);
          fetchInitialGraph();
        }
      }
    } catch (e) {
      // backend not running yet — ignore
    }
  };

  const handleResetCorpus = async () => {
    if (!window.confirm('Clear all cached papers? The graph will rebuild dynamically from your queries.')) return;
    try {
      const res = await fetch('http://localhost:8000/api/corpus/reset', { method: 'DELETE' });
      if (res.ok) {
        setGraphData({ nodes: [], links: [] });
        setGraphContext({});
        setRetrievedPapers([]);
        setCorpusStatus(null);
        setMessages((prev) => [...prev, {
          role: 'assistant',
          content: 'Corpus cleared. The system will now build its knowledge graph dynamically from your arXiv queries.',
        }]);
      }
    } catch (e) {
      console.error('Reset error:', e);
    }
  };

  const handleSyncArxiv = async () => {
    if (syncRunning) return;
    setSyncRunning(true);
    try {
      const res = await fetch('http://localhost:8000/api/refresh', { method: 'POST' });
      if (!res.ok) throw new Error('Refresh failed');
      // Poll for completion every 10 seconds
      syncPollRef.current = setInterval(fetchCorpusStatus, 10000);
    } catch (e) {
      console.error("Sync error:", e);
      setSyncRunning(false);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = input.trim();
    setInput('');
    setOriginalQuery(userMessage); // Track original query for clarification flow
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);
    setClarifyingQuestion(null);

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage }),
      });

      if (!response.ok) {
        throw new Error("API call failed");
      }

      const data = await response.json();
      handleAgentResponse(data);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: "Sorry, I encountered an error executing the agent reasoning flow." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleAnswerClarify = async (answer: string) => {
    if (!clarifyingQuestion) return;

    // Append user's clarification answer in the UI chat
    setMessages((prev) => [...prev, { role: 'user', content: answer }]);
    setLoading(true);

    const question = clarifyingQuestion;
    setClarifyingQuestion(null);

    try {
      const response = await fetch('http://localhost:8000/api/chat/clarify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: originalQuery,
          question: question,
          answer: answer,
          clarification_answers: clarificationAnswers,
        }),
      });

      if (!response.ok) {
        throw new Error("Clarification API call failed");
      }

      const data = await response.json();
      handleAgentResponse(data);
    } catch (error) {
      console.error("Error sending clarification:", error);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: "Sorry, I encountered an error executing the clarification flow." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleAgentResponse = (data: any) => {
    // 1. Check if clarification is needed
    if (data.clarification_needed) {
      setClarifyingQuestion(data.clarification_question);
      setClarificationAnswers(data.clarification_answers || []);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Before I search the graph, I need to align on your intent:\n\n**${data.clarification_question}**`,
          isClarifying: true,
        },
      ]);
      return;
    }

    // 2. Set response context
    setMessages((prev) => [...prev, { role: 'assistant', content: data.final_response }]);
    setRetrievedPapers(data.retrieved_papers || []);
    setGraphContext(data.graph_context || {});

    // 3. Update graph visualization with local subgraph neighborhood
    if (data.graph_context?.subgraph) {
      const sg = data.graph_context.subgraph;
      console.log('[Graph] Received subgraph:', sg.nodes?.length, 'nodes,', sg.links?.length, 'links');
      setGraphData(sg);
    } else {
      console.warn('[Graph] No subgraph in response. graph_context keys:', Object.keys(data.graph_context || {}));
    }

    // Select the first retrieved paper automatically in inspector
    if (data.retrieved_papers && data.retrieved_papers.length > 0) {
      setSelectedPaperId(data.retrieved_papers[0].id);
    }
  };

  // PDF File Upload Handler
  const handlePdfUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadStatus({ type: 'loading', message: 'Parsing PDF and extracting metadata...' });
    const formData = new FormData();
    formData.append('file', file);
    formData.append('topic', 'LLM Agents');

    try {
      const res = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Upload failed");
      }

      const data = await res.json();

      if (data.warning === 'off_topic') {
        setUploadStatus({
          type: 'warning',
          message: `Off-topic paper detected (score: ${data.relevance_score}). Not indexed.`,
        });
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `### Off-Topic Paper Detected\n\nPaper **"${data.metadata?.title}"** has a relevance score of **${data.relevance_score}** — it doesn't appear to be about LLM agents or autonomous systems.\n\nDetected topics: ${data.detected_topics?.join(', ') || 'none'}\n\nThe paper was **not indexed**. If you want to add it anyway, re-upload with force enabled.`,
          },
        ]);
        return;
      }

      setUploadStatus({
        type: 'success',
        message: `"${data.metadata.title}" indexed (relevance: ${data.relevance_score}).`,
      });

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `### Document Indexed Successfully\n\nI have parsed **${data.metadata.title}** [${data.paper_id}] and registered it in the database.\n\n* **Authors**: ${data.metadata.authors?.join(', ')}\n* **Topics**: ${data.detected_topics?.join(', ')}\n* **Relevance score**: ${data.relevance_score}\n* **Concepts**: ${data.metadata.concepts?.join(', ')}\n\nThis node has been added to the graph. You can now search for it or click its node to inspect it!`,
        },
      ]);

      fetchInitialGraph();
      fetchCorpusStatus();
      setSelectedPaperId(data.paper_id);
    } catch (err: any) {
      setUploadStatus({
        type: 'error',
        message: 'Failed to upload/parse PDF paper.',
      });
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className={isDarkMode ? 'dark' : ''}>
      <div className="h-screen w-screen bg-background text-foreground flex flex-col overflow-hidden">
        {/* Hidden file input for PDF uploading */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handlePdfUpload}
          className="hidden"
        />

        {/* Top dashboard nav */}
        <header className="flex items-center justify-between px-6 py-3 border-b border-border bg-sidebar shrink-0">
          <div className="flex items-center gap-3">
            <BookOpen className="text-primary w-5 h-5" />
            <h1 className="text-base font-bold tracking-tight bg-gradient-to-r from-foreground via-foreground/80 to-primary bg-clip-text text-transparent">
              Agentic Knowledge Graph Navigator
            </h1>
          </div>
        
          {/* Right header actions */}
        <div className="flex items-center gap-3">
          {/* Upload status indicator */}
          {uploadStatus.type && (
            <div className={`px-3 py-1.5 rounded-lg text-xs font-medium border flex items-center gap-2 ${
              uploadStatus.type === 'loading' ? 'bg-[#1E293B] border-[#334155] text-[#94A3B8]' :
              uploadStatus.type === 'success' ? 'bg-emerald-950/30 border-emerald-500/30 text-emerald-400' :
              uploadStatus.type === 'warning' ? 'bg-amber-950/30 border-amber-500/30 text-amber-400' :
              'bg-rose-950/30 border-rose-500/30 text-rose-400'
            }`}>
              {uploadStatus.type === 'loading' && <div className="w-3.5 h-3.5 border-2 border-t-transparent border-[#94A3B8] rounded-full animate-spin" />}
              {uploadStatus.type === 'success' && <Sparkles size={14} />}
              {uploadStatus.type === 'warning' && <AlertCircle size={14} />}
              {uploadStatus.type === 'error' && <AlertCircle size={14} />}
              <span className="max-w-[260px] truncate">{uploadStatus.message}</span>
            </div>
          )}

          {/* Theme Toggle Button */}
          <button
            onClick={() => setIsDarkMode(!isDarkMode)}
            title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
            className="flex items-center justify-center p-2 rounded-lg border border-border bg-card text-foreground hover:bg-background transition"
          >
            {isDarkMode ? <Sun size={13} className="text-amber-400" /> : <Moon size={13} className="text-blue-600" />}
          </button>

          {/* Corpus status pill */}
          {corpusStatus?.paper_count != null && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 bg-card border border-border rounded-lg text-[10px] text-[#64748B]">
              <span className="font-semibold text-foreground">{corpusStatus.paper_count}</span>
              <span>papers</span>
              {corpusStatus.last_updated && (
                <>
                  <Clock size={11} className="ml-1" />
                  <span>{new Date(corpusStatus.last_updated).toLocaleDateString()}</span>
                </>
              )}
            </div>
          )}

          {/* Reset corpus button */}
          <button
            onClick={handleResetCorpus}
            title="Clear all cached papers (graph rebuilds from queries)"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-rose-500/30 text-rose-400 hover:bg-rose-950/30 transition"
          >
            <Trash2 size={13} />
            Reset
          </button>

          {/* Sync arXiv button */}
          <button
            onClick={handleSyncArxiv}
            disabled={syncRunning}
            title="Fetch recent agent papers from arXiv (last 4 months)"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition ${
              syncRunning
                ? 'bg-primary/10 border-primary/30 text-primary/60 cursor-not-allowed'
                : 'bg-primary/10 border-primary/40 text-primary hover:bg-primary/20 hover:border-primary/60'
            }`}
          >
            <RefreshCw size={13} className={syncRunning ? 'animate-spin' : ''} />
            {syncRunning ? 'Syncing arXiv...' : 'Sync arXiv'}
          </button>
        </div>
      </header>

      {/* Workspace Panel splits */}
      <main className="flex-1 w-full flex overflow-hidden p-4 gap-4">
        {/* Left Side: Chat log */}
        <section className="w-[38%] h-full flex flex-col shrink-0">
          <Chat
            messages={messages}
            input={input}
            setInput={setInput}
            onSubmit={handleSendMessage}
            loading={loading}
            clarifyingQuestion={clarifyingQuestion}
            onAnswerClarify={handleAnswerClarify}
            onSelectPaper={setSelectedPaperId}
            onFileUploadClick={triggerFileInput}
          />
        </section>

        {/* Right Side: Graph + Inspector split */}
        <section className="flex-1 h-full flex flex-col gap-4 overflow-hidden">
          {/* Top 55%: Interactive Force-directed Network */}
          <div className="h-[55%] w-full">
            <GraphView
              graphData={graphData}
              onSelectNode={setSelectedPaperId}
              selectedNodeId={selectedPaperId}
              highlightedPaths={graphContext?.citation_paths || []}
              isDarkMode={isDarkMode}
            />
          </div>
          {/* Bottom 45%: Grounding Evidence inspector details */}
          <div className="flex-1 w-full min-h-0">
            <EvidencePanel
              selectedPaperId={selectedPaperId}
              graphContext={graphContext}
              retrievedPapers={retrievedPapers}
            />
          </div>
        </section>
      </main>
    </div>
  </div>
  );
}
