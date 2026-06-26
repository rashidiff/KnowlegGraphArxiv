'use client';

import React, { useState, useEffect } from 'react';
import { FileText, Award, BarChart3, Database, Search, ArrowRight, User } from 'lucide-react';

interface PaperDetails {
  id: string;
  title: string;
  abstract: string;
  authors: string[];
  year: number;
  venue: string;
  citation_count: number;
  topics: string[];
  entities: { type: string; value: string }[];
  references: string[];
  citations: string[];
  intro_summary?: string;
  conclusion_summary?: string;
  section_headers?: string[];
}

interface EvidencePanelProps {
  selectedPaperId: string | null;
  graphContext: any;
  retrievedPapers: any[];
}

export default function EvidencePanel({
  selectedPaperId,
  graphContext,
  retrievedPapers,
}: EvidencePanelProps) {
  const [activeTab, setActiveTab] = useState<'inspector' | 'insights' | 'library'>('insights');
  const [paperDetails, setPaperDetails] = useState<PaperDetails | null>(null);
  const [loading, setLoading] = useState(false);
  
  // Library search state
  const [searchQuery, setSearchQuery] = useState('');
  const [libraryPapers, setLibraryPapers] = useState<any[]>([]);

  // Auto-switch to inspector when a paper is selected
  useEffect(() => {
    if (selectedPaperId) {
      setActiveTab('inspector');
      fetchPaperDetails(selectedPaperId);
    }
  }, [selectedPaperId]);

  // Load initial library list from retrieved papers
  useEffect(() => {
    if (retrievedPapers && retrievedPapers.length > 0) {
      setLibraryPapers(retrievedPapers);
    }
  }, [retrievedPapers]);

  const fetchPaperDetails = async (id: string) => {
    setLoading(true);
    try {
      const res = await fetch(`http://localhost:8000/api/papers/${id}`);
      if (res.ok) {
        const data = await res.json();
        setPaperDetails(data);
      }
    } catch (e) {
      console.error("Error fetching paper details:", e);
    } finally {
      setLoading(false);
    }
  };

  const filteredLibrary = libraryPapers.filter((p) => 
    p.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    p.abstract.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="w-full h-full bg-sidebar border border-border rounded-xl overflow-hidden flex flex-col shadow-sm">
      {/* Tabs */}
      <div className="flex border-b border-border bg-card">
        <button
          onClick={() => setActiveTab('insights')}
          className={`flex-1 py-3 text-xs font-medium border-b-2 flex items-center justify-center gap-1.5 transition ${
            activeTab === 'insights'
              ? 'border-primary text-primary bg-sidebar/50'
              : 'border-transparent text-foreground/70 hover:text-foreground'
          }`}
        >
          <BarChart3 size={14} />
          Graph Insights
        </button>
        <button
          onClick={() => setActiveTab('inspector')}
          className={`flex-1 py-3 text-xs font-medium border-b-2 flex items-center justify-center gap-1.5 transition ${
            activeTab === 'inspector'
              ? 'border-primary text-primary bg-sidebar/50'
              : 'border-transparent text-foreground/70 hover:text-foreground'
          }`}
        >
          <FileText size={14} />
          Paper Inspector
        </button>
        <button
          onClick={() => setActiveTab('library')}
          className={`flex-1 py-3 text-xs font-medium border-b-2 flex items-center justify-center gap-1.5 transition ${
            activeTab === 'library'
              ? 'border-primary text-primary bg-sidebar/50'
              : 'border-transparent text-foreground/70 hover:text-foreground'
          }`}
        >
          <Database size={14} />
          Local Library
        </button>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto p-4">
        {/* TAB 1: GRAPH INSIGHTS */}
        {activeTab === 'insights' && (
          <div className="flex flex-col gap-5">
            {/* Louvain Clusters */}
            {graphContext?.clusters && Object.keys(graphContext.clusters).length > 0 ? (
              <div className="flex flex-col gap-2">
                <h3 className="text-xs font-semibold text-foreground/60 uppercase tracking-wider">Identified Topic Clusters</h3>
                <div className="grid grid-cols-1 gap-2">
                  {Object.entries(graphContext.clusters).map(([commId, items]: any) => (
                    <div key={commId} className="bg-card border border-border p-3 rounded-lg flex flex-col gap-1.5">
                      <span className="text-xs font-semibold text-primary">Cluster Group {commId === 'Unassigned' ? 'Unassigned' : `${parseInt(commId) + 1}`}</span>
                      <ul className="text-[11px] text-foreground/80 list-disc list-inside flex flex-col gap-1">
                        {items.map((item: any, idx: number) => (
                          <li key={idx} className="truncate" title={item.title}>{item.title}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            {/* Foundational Papers */}
            {graphContext?.foundational_papers && graphContext.foundational_papers.length > 0 ? (
              <div className="flex flex-col gap-2">
                <h3 className="text-xs font-semibold text-foreground/60 uppercase tracking-wider">Foundational Papers (PageRank Centrality)</h3>
                <div className="flex flex-col gap-2">
                  {graphContext.foundational_papers.map((p: any, idx: number) => (
                    <div key={p.id} className="flex items-start gap-2.5 bg-card border border-border p-2.5 rounded-lg hover:border-border/80 transition cursor-pointer" onClick={() => fetchPaperDetails(p.id)}>
                      <div className="bg-primary/10 text-primary p-1.5 rounded text-xs font-bold flex items-center justify-center w-6 h-6 shrink-0">
                        {idx + 1}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-medium text-foreground truncate" title={p.title}>{p.title}</p>
                        <p className="text-[10px] text-foreground/60">Rank score: {p.score.toFixed(4)}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            {/* Bridge Papers */}
            {graphContext?.bridge_papers && graphContext.bridge_papers.length > 0 ? (
              <div className="flex flex-col gap-2">
                <h3 className="text-xs font-semibold text-foreground/60 uppercase tracking-wider">Bridge Papers (Betweenness Centrality)</h3>
                <div className="flex flex-col gap-2">
                  {graphContext.bridge_papers.map((p: any) => (
                    <div key={p.id} className="flex items-start gap-2.5 bg-card border border-border p-2.5 rounded-lg hover:border-border/80 transition cursor-pointer" onClick={() => fetchPaperDetails(p.id)}>
                      <Award size={16} className="text-pink-500 shrink-0 mt-0.5" />
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-medium text-foreground truncate" title={p.title}>{p.title}</p>
                        <p className="text-[10px] text-foreground/60">Bridge score: {p.score.toFixed(4)}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            {/* Default State */}
            {(!graphContext || (!graphContext.clusters && !graphContext.foundational_papers)) && (
              <div className="text-center py-12 text-xs text-[#475569]">
                Submit a research query to view citation network insights.
              </div>
            )}
          </div>
        )}

        {/* TAB 2: PAPER INSPECTOR */}
        {activeTab === 'inspector' && (
          <div className="h-full">
            {loading ? (
              <div className="flex flex-col items-center justify-center h-48 gap-2">
                <div className="w-6 h-6 border-2 border-t-transparent border-primary rounded-full animate-spin" />
                <span className="text-xs text-foreground/60">Loading metadata...</span>
              </div>
            ) : paperDetails ? (
              <div className="flex flex-col gap-4">
                {/* Meta details */}
                <div className="flex flex-col gap-1.5">
                  <span className="text-[10px] font-semibold text-primary uppercase tracking-wider">{paperDetails.venue} ({paperDetails.year})</span>
                  <h2 className="text-sm font-semibold text-foreground leading-snug">{paperDetails.title}</h2>
                  <div className="flex flex-wrap gap-1.5 mt-1">
                    {paperDetails.topics.map((t) => (
                      <span key={t} className="px-2 py-0.5 bg-primary/10 text-primary text-[9px] font-medium rounded-full">{t}</span>
                    ))}
                    <span className="px-2 py-0.5 bg-background text-foreground/80 text-[9px] font-medium rounded-full border border-border">
                      Citations: {paperDetails.citation_count}
                    </span>
                  </div>
                </div>

                {/* Authors */}
                <div className="flex items-center gap-1.5 text-xs text-foreground/90 bg-card px-3 py-2 rounded-lg border border-border">
                  <User size={14} className="text-foreground/50" />
                  <span className="truncate">{paperDetails.authors.join(', ')}</span>
                </div>

                {/* Abstract */}
                <div className="flex flex-col gap-1">
                  <span className="text-[10px] font-semibold text-foreground/50 uppercase tracking-wider">Abstract</span>
                  <p className="text-[10px] text-foreground/80 leading-relaxed bg-sidebar border border-border/60 p-3 rounded-lg select-text">
                    {paperDetails.abstract}
                  </p>
                </div>

                {/* Optional summaries (Introduction/Conclusion) */}
                {paperDetails.intro_summary && (
                  <div className="flex flex-col gap-1 border-t border-border pt-3">
                    <span className="text-[10px] font-semibold text-foreground/50 uppercase tracking-wider">Introduction Excerpt</span>
                    <p className="text-xs text-foreground/80 leading-relaxed p-3 bg-card/50 rounded-lg">
                      {paperDetails.intro_summary.substring(0, 400)}...
                    </p>
                  </div>
                )}
                
                {paperDetails.conclusion_summary && (
                  <div className="flex flex-col gap-1">
                    <span className="text-[10px] font-semibold text-foreground/50 uppercase tracking-wider">Conclusion Excerpt</span>
                    <p className="text-xs text-foreground/80 leading-relaxed p-3 bg-card/50 rounded-lg">
                      {paperDetails.conclusion_summary.substring(0, 400)}...
                    </p>
                  </div>
                )}

                {/* Extracted Entities */}
                {paperDetails.entities && paperDetails.entities.length > 0 && (
                  <div className="flex flex-col gap-1.5 border-t border-[#1E293B] pt-3">
                    <span className="text-[10px] font-semibold text-[#64748B] uppercase tracking-wider">Extracted KG Entities</span>
                    <div className="flex flex-wrap gap-1.5">
                      {paperDetails.entities.map((ent, idx) => (
                        <span key={idx} className={`px-2 py-0.5 text-[9px] font-medium rounded-full ${
                          ent.type === 'concept' ? 'bg-[#312E81] text-[#A5B4FC] border border-[#4338CA]' :
                          ent.type === 'method' ? 'bg-[#064E3B] text-[#A7F3D0] border border-[#047857]' :
                          'bg-[#701A75] text-[#F5D0FE] border border-[#A21CAF]'
                        }`}>
                          {ent.type}: {ent.value}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Citation Edges */}
                <div className="grid grid-cols-2 gap-3 border-t border-border pt-3">
                  <div className="flex flex-col gap-1">
                    <span className="text-[10px] font-semibold text-foreground/50 uppercase tracking-wider">
                      References&nbsp;
                      <span className="text-primary">{paperDetails.references.length} local</span>
                      {(paperDetails as any).total_references > paperDetails.references.length && (
                        <span className="text-foreground/40"> / {(paperDetails as any).total_references} total</span>
                      )}
                    </span>
                    <div className="flex flex-col gap-0.5 max-h-28 overflow-y-auto">
                      {paperDetails.references.length === 0 ? (
                        <span className="text-[9px] text-foreground/40 italic">No local references yet</span>
                      ) : paperDetails.references.map((refId) => (
                        <div key={refId} className="text-[9px] text-foreground/80 hover:text-primary cursor-pointer truncate py-0.5 border-b border-border/50"
                             onClick={() => { fetchPaperDetails(refId); setActiveTab('inspector'); }}>
                          → {refId}
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="flex flex-col gap-1">
                    <span className="text-[10px] font-semibold text-foreground/50 uppercase tracking-wider">
                      Cited By&nbsp;<span className="text-primary">{paperDetails.citations.length}</span>
                    </span>
                    <div className="flex flex-col gap-0.5 max-h-28 overflow-y-auto">
                      {paperDetails.citations.length === 0 ? (
                        <span className="text-[9px] text-foreground/40 italic">Not cited by local papers</span>
                      ) : paperDetails.citations.map((citId) => (
                        <div key={citId} className="text-[9px] text-foreground/80 hover:text-primary cursor-pointer truncate py-0.5 border-b border-border/50"
                             onClick={() => { fetchPaperDetails(citId); setActiveTab('inspector'); }}>
                          ← {citId}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-24 text-xs text-foreground/40">
                Click on a paper node in the graph or a citation in the chat to view full details.
              </div>
            )}
          </div>
        )}

        {/* TAB 3: LOCAL LIBRARY */}
        {activeTab === 'library' && (
          <div className="flex flex-col gap-3">
            {/* Search Input */}
            <div className="relative">
              <Search className="absolute left-2.5 top-2.5 text-foreground/40" size={14} />
              <input
                type="text"
                placeholder="Search corpus papers..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-card border border-border pl-8 pr-3 py-1.5 text-xs text-foreground rounded-lg focus:outline-none focus:border-primary transition"
              />
            </div>

            {/* Papers list */}
            <div className="flex flex-col gap-2">
              {filteredLibrary.length > 0 ? (
                filteredLibrary.map((p) => (
                  <div
                     key={p.id}
                     onClick={() => fetchPaperDetails(p.id)}
                     className="p-3 bg-card border border-border rounded-lg hover:border-border/80 cursor-pointer transition flex flex-col gap-1"
                  >
                    <div className="flex items-center justify-between text-[9px] text-foreground/50 font-semibold">
                      <span>{p.venue || "Unknown"}</span>
                      <span>{p.year}</span>
                    </div>
                    <span className="text-[11px] font-medium text-foreground leading-tight hover:text-primary">{p.title}</span>
                    <span className="text-[9px] text-foreground/60 line-clamp-2 leading-relaxed mt-0.5">{p.abstract}</span>
                  </div>
                ))
              ) : (
                <div className="text-center py-12 text-xs text-foreground/40">
                  No papers matching search in the retrieved set.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
