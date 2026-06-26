'use client';

import React, { useRef, useEffect } from 'react';
import { Send, Sparkles, AlertCircle, HelpCircle, FileUp } from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  isClarifying?: boolean;
  clarifyingQuestion?: string;
  clarifyingAnswers?: any[];
}

interface ChatProps {
  messages: Message[];
  input: string;
  setInput: (val: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  loading: boolean;
  clarifyingQuestion: string | null;
  onAnswerClarify: (answer: string) => void;
  onSelectPaper: (paperId: string) => void;
  onFileUploadClick: () => void;
}

export default function Chat({
  messages,
  input,
  setInput,
  onSubmit,
  loading,
  clarifyingQuestion,
  onAnswerClarify,
  onSelectPaper,
  onFileUploadClick,
}: ChatProps) {
  const chatEndRef = useRef<HTMLDivElement>(null);
  const clarifyInputRef = useRef<HTMLInputElement>(null);

  // Auto scroll to bottom when messages change or loading state changes
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Regular expression to find citation patterns like [W123] or [W_abc] or [W1]
  const parseCitations = (text: string) => {
    if (!text) return '';
    const parts = [];
    const regex = /\[(W\w+|upload_\w+)\]/g;
    let lastIndex = 0;
    let match;

    while ((match = regex.exec(text)) !== null) {
      const matchIndex = match.index;
      const paperId = match[1];

      // Add text preceding the citation
      if (matchIndex > lastIndex) {
        parts.push(text.substring(lastIndex, matchIndex));
      }

      // Add the clickable citation node
      parts.push(
        <button
          key={matchIndex}
          onClick={() => onSelectPaper(paperId)}
          className="px-1.5 py-0.5 mx-0.5 bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20 hover:border-primary/40 rounded text-[10px] font-bold transition inline-flex items-center"
        >
          {paperId}
        </button>
      );

      lastIndex = regex.lastIndex;
    }

    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }

    return parts.length > 0 ? parts : text;
  };

  // Renders markdown paragraphs and bullet points simply
  const renderMessageContent = (content: string) => {
    return content.split('\n').map((line, lineIdx) => {
      // Headers
      if (line.startsWith('## ')) {
        return <h3 key={lineIdx} className="text-sm font-semibold text-primary mt-4 mb-2 first:mt-0">{line.replace('## ', '')}</h3>;
      }
      if (line.startsWith('### ')) {
        return <h4 key={lineIdx} className="text-xs font-semibold text-foreground/90 mt-3 mb-1">{line.replace('### ', '')}</h4>;
      }
      // Bold list items
      if (line.startsWith('- **') || line.startsWith('* **')) {
        const cleanLine = line.replace(/^[-*]\s+/, '');
        return (
          <div key={lineIdx} className="pl-4 py-0.5 text-xs text-foreground/80 leading-relaxed relative before:content-['•'] before:absolute before:left-0 before:text-primary select-text">
            {parseCitations(cleanLine)}
          </div>
        );
      }
      // Standard list items
      if (line.startsWith('- ') || line.startsWith('* ')) {
        const cleanLine = line.replace(/^[-*]\s+/, '');
        return (
          <div key={lineIdx} className="pl-4 py-0.5 text-xs text-foreground/85 leading-relaxed relative before:content-['•'] before:absolute before:left-0 before:text-[#475569] select-text">
            {parseCitations(cleanLine)}
          </div>
        );
      }
      // Empty lines
      if (!line.trim()) {
        return <div key={lineIdx} className="h-2" />;
      }
      // Standard paragraphs
      return (
        <p key={lineIdx} className="text-xs text-foreground/90 leading-relaxed mb-2 select-text">
          {parseCitations(line)}
        </p>
      );
    });
  };

  const handleClarifySubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (clarifyInputRef.current && clarifyInputRef.current.value.trim()) {
      onAnswerClarify(clarifyInputRef.current.value.trim());
      clarifyInputRef.current.value = '';
    }
  };

  return (
    <div className="w-full h-full flex flex-col bg-sidebar border border-border rounded-xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-card border-b border-border">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
          <span className="font-medium text-sm text-foreground">Agentic Research Navigator</span>
        </div>
        <button
          onClick={onFileUploadClick}
          className="flex items-center gap-1 px-2.5 py-1.5 bg-background hover:bg-card border border-border text-xs font-medium text-foreground rounded-lg transition"
          title="Upload PDF paper"
        >
          <FileUp size={14} />
          Upload PDF
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
        {messages.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-center gap-3 p-6">
            <Sparkles size={40} className="text-primary animate-pulse" />
            <h2 className="text-sm font-semibold text-foreground">Welcome to the Research Navigator</h2>
            <p className="text-xs text-foreground/70 max-w-sm leading-relaxed">
              Ask natural-language questions to discover papers, generate reading roadmaps, compare methods, and find research gaps across the citation network.
            </p>
            <div className="grid grid-cols-1 gap-2 w-full max-w-md mt-4">
              <button
                onClick={() => setInput("What are the most important papers on web agents?")}
                className="p-2.5 bg-card border border-border hover:border-primary text-[11px] text-left text-foreground/75 hover:text-foreground rounded-lg transition"
              >
                "What are the most important papers on web agents?"
              </button>
              <button
                onClick={() => setInput("I want to learn about browser agents. Give me a reading path.")}
                className="p-2.5 bg-card border border-border hover:border-primary text-[11px] text-left text-foreground/75 hover:text-foreground rounded-lg transition"
              >
                "I want to learn about browser agents. Give me a reading path."
              </button>
              <button
                onClick={() => setInput("Suggest a new research direction based on the limitations in the corpus.")}
                className="p-2.5 bg-card border border-border hover:border-primary text-[11px] text-left text-foreground/75 hover:text-foreground rounded-lg transition"
              >
                "Suggest a new research direction based on the limitations in the corpus."
              </button>
            </div>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex flex-col gap-1 max-w-[85%] ${
                msg.role === 'user' ? 'self-end items-end' : 'self-start items-start'
              }`}
            >
              {/* Message Bubble */}
              <div
                className={`p-3.5 rounded-xl border text-xs leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-primary text-[#FFFFFF] border-primary/20 rounded-tr-none'
                    : 'bg-card text-foreground border-border rounded-tl-none select-text'
                }`}
              >
                {msg.role === 'user' ? (
                  <p className="select-text whitespace-pre-wrap">{msg.content}</p>
                ) : (
                  renderMessageContent(msg.content)
                )}
              </div>
            </div>
          ))
        )}

        {/* Loading thoughts indicator */}
        {loading && (
          <div className="self-start max-w-[80%] flex items-start gap-2.5">
            <div className="bg-card border border-border p-3 rounded-xl rounded-tl-none flex items-center gap-3">
              <div className="flex gap-1">
                <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" />
                <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce [animation-delay:0.2s]" />
                <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce [animation-delay:0.4s]" />
              </div>
              <span className="text-[11px] text-[#64748B] font-medium agent-thinking">
                Orchestrating agent reasoning graph...
              </span>
            </div>
          </div>
        )}

        {/* Clarifying Question Modal Prompt */}
        {clarifyingQuestion && !loading && (
          <div className="self-start w-full max-w-lg border border-amber-500/20 bg-amber-500/5 p-4 rounded-xl flex flex-col gap-3">
            <div className="flex gap-2 text-amber-500 items-start">
              <HelpCircle size={18} className="shrink-0 mt-0.5" />
              <div className="flex flex-col gap-1">
                <span className="text-xs font-semibold">Clarification Required</span>
                <span className="text-xs text-foreground/80">{clarifyingQuestion}</span>
              </div>
            </div>
            <form onSubmit={handleClarifySubmit} className="flex gap-2">
              <input
                ref={clarifyInputRef}
                type="text"
                placeholder="Type your clarification answer..."
                className="flex-1 bg-card border border-border hover:border-border/80 focus:border-amber-500 px-3 py-2 text-xs text-foreground rounded-lg focus:outline-none transition"
              />
              <button
                type="submit"
                className="px-4 py-2 bg-amber-600 hover:bg-amber-700 text-xs font-semibold text-white rounded-lg transition"
              >
                Submit
              </button>
            </form>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* Standard Input Form */}
      {!clarifyingQuestion && (
        <form onSubmit={onSubmit} className="p-3 bg-card border-t border-border flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
            placeholder={loading ? "Waiting for agents..." : "Ask your research question..."}
            className="flex-1 bg-background border border-border hover:border-border/80 focus:border-primary px-3 py-2 text-xs text-foreground rounded-lg focus:outline-none transition disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="px-4 py-2 bg-primary hover:bg-primary-hover text-[#FFFFFF] rounded-lg transition flex items-center justify-center disabled:opacity-50 disabled:hover:bg-primary shrink-0"
          >
            <Send size={14} />
          </button>
        </form>
      )}
    </div>
  );
}
