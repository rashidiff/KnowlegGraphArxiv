'use client';

import React, { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import { Network, ZoomIn, ZoomOut, Maximize } from 'lucide-react';

// Dynamically load ForceGraph2D as it relies on the browser canvas and window object
const ForceGraph2D = dynamic(
  () => import('react-force-graph-2d').then((mod) => mod.default),
  { ssr: false }
);

interface Node {
  id: string;
  title: string;
  year: number;
  citation_count: number;
  topic: string;
  x?: number;
  y?: number;
}

interface Link {
  source: string | Node;
  target: string | Node;
  type: string;
}

interface GraphData {
  nodes: Node[];
  links: Link[];
}

interface GraphViewProps {
  graphData: GraphData;
  onSelectNode: (nodeId: string) => void;
  selectedNodeId: string | null;
  highlightedPaths?: any[];
  isDarkMode?: boolean;
}

export default function GraphView({
  graphData,
  onSelectNode,
  selectedNodeId,
  highlightedPaths = [],
  isDarkMode = true,
}: GraphViewProps) {
  const fgRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 500 });
  const [highlightedNodes, setHighlightedNodes] = useState<Set<string>>(new Set());
  const [highlightedLinks, setHighlightedLinks] = useState<Set<string>>(new Set());

  // Configure d3 forces for better node spacing (stronger repulsion, longer links)
  useEffect(() => {
    if (!fgRef.current || !graphData.nodes.length) return;
    const fg = fgRef.current;
    const charge = fg.d3Force('charge');
    if (charge) charge.strength(-400);
    const link = fg.d3Force('link');
    if (link) link.distance(120).strength(0.5);
    fg.d3ReheatSimulation();
  }, [graphData]);

  // Handle responsive resize
  useEffect(() => {
    if (!containerRef.current) return;
    const handleResize = () => {
      setDimensions({
        width: containerRef.current?.clientWidth || 600,
        height: containerRef.current?.clientHeight || 500,
      });
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Compute highlighted nodes and links from citation paths
  useEffect(() => {
    const nodes = new Set<string>();
    const links = new Set<string>();

    highlightedPaths.forEach((pathObj) => {
      const path = pathObj.path || [];
      for (let i = 0; i < path.length; i++) {
        nodes.add(path[i].id);
        if (i < path.length - 1) {
          // Both directions to match links source/target IDs
          const id1 = `${path[i].id}->${path[i+1].id}`;
          const id2 = `${path[i+1].id}->${path[i].id}`;
          links.add(id1);
          links.add(id2);
        }
      }
    });

    setHighlightedNodes(nodes);
    setHighlightedLinks(links);
  }, [highlightedPaths]);

  // Topic color mapper
  const getNodeColor = (node: Node) => {
    if (selectedNodeId === node.id) return '#F43F5E'; // Crimson active node
    if (highlightedNodes.has(node.id)) return '#EC4899'; // Deep pink for path nodes
    
    switch (node.topic) {
      case 'Browser Agents':
        return '#6366F1'; // Indigo
      case 'Web Agents':
        return '#10B981'; // Emerald
      case 'Agent Evaluation':
        return '#EF4444'; // Red
      case 'Human-AI Interaction':
        return '#F59E0B'; // Amber
      case 'Tool Use':
        return '#06B6D4'; // Cyan
      case 'LLM Agents':
        return '#8B5CF6'; // Violet
      default:
        return '#64748B'; // Muted Slate
    }
  };

  // Node size based on citation counts
  const getNodeVal = (node: Node) => {
    const citations = node.citation_count || 0;
    // Minimum size 5 so nodes with 0 citations are clearly visible
    return Math.max(5, Math.min(18, 5 + Math.log1p(citations) * 2.5));
  };

  const handleZoomIn = () => {
    if (fgRef.current) {
      const currentZoom = fgRef.current.zoom();
      fgRef.current.zoom(currentZoom * 1.3, 300);
    }
  };

  const handleZoomOut = () => {
    if (fgRef.current) {
      const currentZoom = fgRef.current.zoom();
      fgRef.current.zoom(currentZoom * 0.7, 300);
    }
  };

  const handleRecenter = () => {
    if (fgRef.current) {
      fgRef.current.zoomToFit(400);
    }
  };

  // Safe checks for empty data
  const hasData = graphData && graphData.nodes && graphData.nodes.length > 0;

  return (
    <div ref={containerRef} className="relative w-full h-full bg-sidebar border border-border rounded-xl overflow-hidden flex flex-col">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-3 bg-card border-b border-border">
        <div className="flex items-center gap-2">
          <Network size={18} className="text-primary" />
          <span className="font-medium text-sm text-foreground">Citation & Knowledge Graph</span>
        </div>
        <div className="flex items-center gap-2">
          <button 
            onClick={handleZoomIn}
            className="p-1 hover:bg-background text-foreground/70 hover:text-foreground rounded transition"
            title="Zoom In"
          >
            <ZoomIn size={16} />
          </button>
          <button 
            onClick={handleZoomOut}
            className="p-1 hover:bg-background text-foreground/70 hover:text-foreground rounded transition"
            title="Zoom Out"
          >
            <ZoomOut size={16} />
          </button>
          <button 
            onClick={handleRecenter}
            className="p-1 hover:bg-background text-foreground/70 hover:text-foreground rounded transition"
            title="Fit to screen"
          >
            <Maximize size={16} />
          </button>
        </div>
      </div>

      {/* Graph area */}
      <div className="flex-1 w-full relative">
        {hasData ? (
          <ForceGraph2D
            key={graphData.nodes.map(n => n.id).join(',')}
            ref={fgRef}
            graphData={graphData}
            width={dimensions.width}
            height={dimensions.height - 45} // Deduct header height
            nodeLabel={(node: any) => `${node.title} (${node.year})`}
            nodeColor={(node: any) => getNodeColor(node)}
            nodeVal={(node: any) => getNodeVal(node)}
            onNodeClick={(node: any) => onSelectNode(node.id)}
            linkWidth={(link: any) => {
              const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
              const targetId = typeof link.target === 'object' ? link.target.id : link.target;
              return highlightedLinks.has(`${sourceId}->${targetId}`) ? 3.5 : 1.0;
            }}
            linkColor={(link: any) => {
              const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
              const targetId = typeof link.target === 'object' ? link.target.id : link.target;
              return highlightedLinks.has(`${sourceId}->${targetId}`) ? '#F43F5E' : (isDarkMode ? '#1E293B' : '#E2E8F0');
            }}
            linkDirectionalParticles={(link: any) => {
              const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
              const targetId = typeof link.target === 'object' ? link.target.id : link.target;
              return highlightedLinks.has(`${sourceId}->${targetId}`) ? 3 : 0;
            }}
            linkDirectionalParticleWidth={2}
            linkDirectionalParticleColor={() => (isDarkMode ? '#FFF' : '#2563EB')}
            nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
              if (node.x == null || node.y == null) return;
              const size = getNodeVal(node);
              const color = getNodeColor(node);

              // Draw node circle
              ctx.beginPath();
              ctx.arc(node.x, node.y, size, 0, 2 * Math.PI, false);
              ctx.fillStyle = color;
              ctx.fill();

              // Add stroke for active node or path nodes
              if (selectedNodeId === node.id || highlightedNodes.has(node.id)) {
                ctx.strokeStyle = '#FFFFFF';
                ctx.lineWidth = 1.5;
                ctx.stroke();
              }

              // Draw title label
              if (globalScale > 1.5) {
                const label = node.title.length > 25 ? node.title.substring(0, 22) + '...' : node.title;
                const fontSize = 10 / globalScale;
                ctx.font = `${fontSize}px Outfit, Inter, sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillStyle = isDarkMode ? '#94A3B8' : '#334155';
                ctx.fillText(label, node.x, node.y + size + 2);
              }
            }}
            cooldownTicks={100}
            onEngineStop={() => {
              // Automatically fit to container once layout stabilizes on load
              if (fgRef.current) {
                fgRef.current.zoomToFit(200);
              }
            }}
          />
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-foreground/50 gap-3 p-6 text-center">
            <Network size={44} className="text-foreground/20" />
            <span className="text-sm font-semibold text-foreground/75">Graph builds from your query</span>
            <span className="text-xs text-foreground/60 max-w-[260px] leading-relaxed">
              Ask a research question — the system will search arXiv live, fetch relevant papers,
              and build this knowledge graph dynamically.
            </span>
          </div>
        )}
      </div>

      {/* Legend */}
      {hasData && (
        <div className="absolute bottom-3 left-3 bg-card/95 border border-border px-3 py-2 rounded-lg text-[10px] text-foreground/80 backdrop-blur flex flex-col gap-1.5 pointer-events-none select-none z-10 shadow-sm">
          <div className="font-semibold text-foreground border-b border-border pb-1 mb-1">Topics Map</div>
          <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-[#6366F1]" /> Browser Agents</div>
          <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-[#10B981]" /> Web Agents</div>
          <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-[#EF4444]" /> Agent Evaluation</div>
          <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-[#F59E0B]" /> Human-AI Interaction</div>
          <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-[#06B6D4]" /> Tool Use</div>
          <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-[#8B5CF6]" /> LLM Agents</div>
        </div>
      )}
    </div>
  );
}
