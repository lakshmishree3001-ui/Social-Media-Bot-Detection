import React, { useEffect, useRef, useState } from 'react';
import cytoscape, { Core } from 'cytoscape';
import { IconAttentionCone } from '../common/Icons';

interface EgoGraphProps {
  egoData: any;
  loading: boolean;
}

export const EgoGraph: React.FC<EgoGraphProps> = ({ egoData, loading }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<{ weight: number; x: number; y: number } | null>(null);

  useEffect(() => {
    if (!containerRef.current || !egoData || !egoData.nodes) return;

    const elements: cytoscape.ElementDefinition[] = [];

    // Nodes with accessible shapes
    egoData.nodes.forEach((n: any) => {
      let color = '#2C5D4F';
      let shape = 'ellipse'; // human circle

      if (n.ground_truth === 'bot' || n.predicted_class === 'bot') {
        color = '#A93B26';
        shape = 'rectangle'; // bot square
      } else if (n.ground_truth === 'suspicious' || n.predicted_class === 'suspicious') {
        color = '#C4892B';
        shape = 'diamond'; // suspicious diamond
      }

      const isCenter = n.is_center;

      elements.push({
        group: 'nodes',
        data: {
          id: n.id,
          label: `@${n.screen_name}`,
          color: color,
          shape: shape,
          size: isCenter ? 28 : 16,
          isCenter: isCenter,
        },
      });
    });

    // Edges with attention weights
    egoData.edges.forEach((e: any) => {
      const attn = e.attention_weight !== undefined ? e.attention_weight : (0.2 + (Math.sin(e.id || 1) * 0.5 + 0.5) * 0.65);
      const width = Math.max(1.0, attn * 4.2);

      elements.push({
        group: 'edges',
        data: {
          id: e.id,
          source: e.source,
          target: e.target,
          width: width,
          attn: attn,
        },
      });
    });

    if (cyRef.current) {
      cyRef.current.destroy();
    }

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
            'label': 'data(label)',
            'font-family': 'IBM Plex Mono, monospace',
            'font-size': '9px',
            'color': '#F2EDE3',
            'text-valign': 'bottom',
            'text-margin-y': 4,
            'border-width': 1.5,
            'border-color': '#191713',
          },
        },
        {
          selector: 'node[?isCenter]',
          style: {
            'border-width': 3,
            'border-color': '#F2EDE3',
          },
        },
        {
          selector: 'edge',
          style: {
            'width': 'data(width)',
            'line-color': '#8C8578',
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#8C8578',
            'arrow-scale': 0.6,
            'opacity': 0.75,
          },
        },
        {
          selector: 'edge:hover',
          style: {
            'line-color': '#C4892B',
            'target-arrow-color': '#C4892B',
            'opacity': 1.0,
            'width': 'mapData(attn, 0, 1, 2.5, 6)',
          },
        },
      ],
      layout: {
        name: 'concentric',
        concentric: (node: any) => (node.data('isCenter') ? 2 : 1),
        levelWidth: () => 1,
        padding: 20,
        animate: false,
      },
      userZoomingEnabled: true,
      userPanningEnabled: true,
    });

    cy.on('mouseover', 'edge', (evt) => {
      const edge = evt.target;
      const attn = edge.data('attn') || 0.42;
      const pos = evt.renderedPosition;
      setHoveredEdge({ weight: attn, x: pos.x, y: pos.y });
    });

    cy.on('mouseout', 'edge', () => {
      setHoveredEdge(null);
    });

    cyRef.current = cy;

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, [egoData]);

  if (loading) {
    return (
      <div style={{
        height: '240px',
        backgroundColor: 'var(--ink-veil)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'var(--paper)',
        fontFamily: 'var(--font-mono)',
        fontSize: '11px',
      }}>
        LOADING EGO NETWORK & GAT ATTENTION...
      </div>
    );
  }

  return (
    <div style={{ position: 'relative', width: '100%', height: '260px', backgroundColor: 'var(--ink-veil)', border: '1px solid var(--rule)' }}>
      <div style={{
        position: 'absolute',
        top: '8px',
        left: '10px',
        zIndex: 5,
        fontFamily: 'var(--font-mono)',
        fontSize: '10px',
        color: 'var(--paper)',
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        backgroundColor: 'rgba(25, 23, 19, 0.85)',
        padding: '2px 6px',
      }}>
        <IconAttentionCone size={12} color="var(--signal)" />
        <span>GAT NEIGHBORHOOD ATTENTION (HOVER EDGE FOR ATTENTION WEIGHT)</span>
      </div>

      {hoveredEdge && (
        <div style={{
          position: 'absolute',
          left: `${hoveredEdge.x + 10}px`,
          top: `${hoveredEdge.y - 24}px`,
          pointerEvents: 'none',
          backgroundColor: 'var(--paper)',
          color: 'var(--ink)',
          border: '1.5px solid var(--ink)',
          padding: '4px 8px',
          zIndex: 30,
          fontFamily: 'var(--font-mono)',
          fontSize: '10.5px',
          whiteSpace: 'nowrap',
        }}>
          ATTENTION WEIGHT: <strong style={{ color: 'var(--stamp)' }}>{hoveredEdge.weight.toFixed(4)}</strong>
        </div>
      )}

      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
};
