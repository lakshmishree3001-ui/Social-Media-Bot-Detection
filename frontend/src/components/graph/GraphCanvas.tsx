import React, { useEffect, useRef, useState } from 'react';
import cytoscape, { Core } from 'cytoscape';
import { SubgraphResponse, GraphNode } from '../../api/client';
import { useAppStore } from '../../store/useAppStore';
import { IconNodeRing, IconSearchReticle } from '../common/Icons';

interface GraphCanvasProps {
  data: SubgraphResponse | null;
  loading: boolean;
  onSelectNode?: (nodeId: number) => void;
}

export const GraphCanvas: React.FC<GraphCanvasProps> = ({ data, loading, onSelectNode }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const { selectedAccountId, openDossierForAccount } = useAppStore();
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const [zoomLevel, setZoomLevel] = useState<number>(1.0);
  const [visibleCount, setVisibleCount] = useState<{ nodes: number; edges: number }>({ nodes: 0, edges: 0 });

  useEffect(() => {
    if (!containerRef.current || !data) return;

    // Check reduced-motion and session-storage for hero animation
    const prefersReducedMotion = typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const hasAnimated = typeof window !== 'undefined' && sessionStorage.getItem('graphwarden_hero_settled');
    const shouldAnimate = !prefersReducedMotion && !hasAnimated;

    // Build Cytoscape elements with accessible node shapes:
    // human = circle (ellipse), bot = square (rectangle), suspicious = diamond
    const elements: cytoscape.ElementDefinition[] = [];

    // Nodes
    data.nodes.forEach((n) => {
      let nodeColor = '#2C5D4F'; // human (ledger)
      let nodeShape = 'ellipse'; // human circle

      if (n.predicted_class === 'bot' || n.ground_truth === 'bot') {
        nodeColor = '#A93B26'; // bot (stamp)
        nodeShape = 'rectangle'; // bot square
      } else if (n.predicted_class === 'suspicious' || n.ground_truth === 'suspicious') {
        nodeColor = '#C4892B'; // suspicious (signal)
        nodeShape = 'diamond'; // suspicious diamond
      }

      // Slightly larger node radius so eye reads network structure immediately
      const size = Math.max(16, Math.min(42, 14 + Math.log10(Math.max(1, n.followers)) * 5.5));

      elements.push({
        group: 'nodes',
        data: {
          id: n.id,
          label: `@${n.label}`,
          color: nodeColor,
          shape: nodeShape,
          size: size,
          raw: n,
        },
      });
    });

    // Edges - higher visibility at rest
    data.edges.forEach((e) => {
      elements.push({
        group: 'edges',
        data: {
          id: e.id,
          source: e.source,
          target: e.target,
          weight: e.weight,
        },
      });
    });

    // Destroy existing instance if any
    if (cyRef.current) {
      cyRef.current.destroy();
    }

    // Initialize Cytoscape
    const cy = cytoscape({
      container: containerRef.current,
      elements: elements,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            'shape': 'data(shape)' as any,
            'width': 'data(size)',
            'height': 'data(size)',
            'label': '',
            'border-width': 1.5,
            'border-color': '#191713',
            'transition-property': 'background-color, border-width, border-color',
            'transition-duration': 150,
          },
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 3.5,
            'border-color': '#F2EDE3',
          },
        },
        {
          selector: 'edge',
          style: {
            'width': 1.2,
            'line-color': '#666053',
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#666053',
            'arrow-scale': 0.7,
            'opacity': 0.75,
          },
        },
        {
          selector: 'edge:selected',
          style: {
            'width': 2.5,
            'line-color': '#C4892B',
            'target-arrow-color': '#C4892B',
            'opacity': 1.0,
          },
        },
      ],
      layout: {
        name: 'cose',
        animate: shouldAnimate,
        animationDuration: 1400,
        idealEdgeLength: () => 65,
        nodeOverlap: 20,
        refresh: 20,
        fit: true,
        padding: 30,
        randomize: shouldAnimate,
        componentSpacing: 100,
        nodeRepulsion: () => 400000,
        edgeElasticity: () => 100,
      } as any,
      minZoom: 0.1,
      maxZoom: 4.0,
    });

    if (shouldAnimate && typeof window !== 'undefined') {
      sessionStorage.setItem('graphwarden_hero_settled', 'true');
    }

    // Readout updates
    const updateMetrics = () => {
      setZoomLevel(cy.zoom());
      setVisibleCount({
        nodes: cy.nodes().length,
        edges: cy.edges().length,
      });
    };

    cy.on('zoom pan render', updateMetrics);
    updateMetrics();

    // Event listeners
    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      const raw = node.data('raw') as GraphNode;
      const accountId = parseInt(raw.id, 10);
      if (onSelectNode) {
        onSelectNode(accountId);
      } else {
        openDossierForAccount(accountId);
      }
    });

    cy.on('mouseover', 'node', (evt) => {
      const node = evt.target;
      const raw = node.data('raw') as GraphNode;
      setHoveredNode(raw);
    });

    cy.on('mouseout', 'node', () => {
      setHoveredNode(null);
    });

    cyRef.current = cy;

    // Highlight initial selected node
    if (selectedAccountId !== null) {
      const target = cy.$(`#${selectedAccountId}`);
      if (target.length > 0) {
        target.select();
      }
    }

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, [data]);

  // Sync selection when store changes
  useEffect(() => {
    if (!cyRef.current || selectedAccountId === null) return;
    const cy = cyRef.current;
    cy.nodes().unselect();
    const target = cy.$(`#${selectedAccountId}`);
    if (target.length > 0) {
      target.select();
    }
  }, [selectedAccountId]);

  const handleResetView = () => {
    if (cyRef.current) {
      cyRef.current.fit(undefined, 30);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (rect) {
      setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    }
  };

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        backgroundColor: 'var(--ink-veil)',
        overflow: 'hidden',
      }}
      onMouseMove={handleMouseMove}
    >
      {/* Loading Overlay */}
      {loading && (
        <div style={{
          position: 'absolute',
          inset: 0,
          backgroundColor: 'rgba(42, 39, 33, 0.85)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 10,
          color: 'var(--paper)',
          fontFamily: 'var(--font-mono)',
          fontSize: '12px',
          letterSpacing: '0.08em',
        }}>
          <IconNodeRing size={28} color="var(--signal)" />
          <span style={{ marginTop: '12px' }}>CALCULATING GRAPH TOPOLOGY & ATTENTION FLOWS...</span>
        </div>
      )}

      {/* Cytoscape Container */}
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

      {/* Corner Plotter Instrument Readout (Section 4) */}
      <div style={{
        position: 'absolute',
        top: '12px',
        right: '12px',
        zIndex: 5,
        backgroundColor: 'rgba(25, 23, 19, 0.88)',
        border: '1px solid var(--rule)',
        padding: '5px 10px',
        fontFamily: 'var(--font-mono)',
        fontSize: '10px',
        color: 'var(--paper)',
        letterSpacing: '0.06em',
      }}>
        ZOOM: {zoomLevel.toFixed(2)}x · {visibleCount.nodes || (data?.nodes?.length ?? 0)} NODES · {visibleCount.edges || (data?.edges?.length ?? 0)} EDGES
      </div>

      {/* Floating Canvas Controls */}
      <div style={{
        position: 'absolute',
        bottom: '16px',
        left: '16px',
        display: 'flex',
        gap: '8px',
        zIndex: 5,
      }}>
        <button
          onClick={handleResetView}
          style={{
            backgroundColor: 'var(--paper)',
            color: 'var(--ink)',
            border: '1px solid var(--rule)',
            padding: '6px 12px',
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <IconSearchReticle size={14} color="var(--ink)" />
          RESET VIEW
        </button>

        <div style={{
          backgroundColor: 'rgba(25, 23, 19, 0.88)',
          color: 'var(--paper)',
          border: '1px solid var(--rule)',
          padding: '6px 14px',
          fontFamily: 'var(--font-mono)',
          fontSize: '10.5px',
          display: 'flex',
          alignItems: 'center',
          gap: '14px',
        }}>
          <span><strong style={{ color: 'var(--stamp)' }}>■ SQUARE</strong>: BOT</span>
          <span><strong style={{ color: 'var(--ledger)' }}>● CIRCLE</strong>: HUMAN</span>
          <span><strong style={{ color: 'var(--signal)' }}>◆ DIAMOND</strong>: SUSPICIOUS</span>
        </div>
      </div>

      {/* Floating Hover Tooltip */}
      {hoveredNode && (
        <div style={{
          position: 'absolute',
          left: `${mousePos.x + 15}px`,
          top: `${mousePos.y + 15}px`,
          pointerEvents: 'none',
          backgroundColor: 'var(--paper)',
          color: 'var(--ink)',
          border: '1.5px solid var(--ink)',
          padding: '8px 12px',
          zIndex: 20,
          fontFamily: 'var(--font-mono)',
          fontSize: '11px',
          maxWidth: '220px',
        }}>
          <div style={{ fontWeight: 700, fontSize: '12px', marginBottom: '4px' }}>
            @{hoveredNode.label.replace('@', '')}
          </div>
          <div>ID: #{hoveredNode.id}</div>
          <div>CLASS: <strong style={{
            color: hoveredNode.predicted_class === 'bot' ? 'var(--stamp)' : 'var(--ledger)'
          }}>{hoveredNode.predicted_class.toUpperCase()}</strong></div>
          <div>CONFIDENCE: {(hoveredNode.confidence * 100).toFixed(1)}%</div>
          <div>FOLLOWERS: {hoveredNode.followers.toLocaleString()}</div>
          <div style={{ marginTop: '4px', fontSize: '9.5px', color: 'var(--ink-muted)' }}>
            CLICK TO OPEN DOSSIER
          </div>
        </div>
      )}
    </div>
  );
};

