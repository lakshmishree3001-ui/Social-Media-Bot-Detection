import React from 'react';

interface IconProps {
  size?: number;
  color?: string;
  className?: string;
}

/**
 * Custom SVG Mark 1: Node Ring
 * Topological vertex identifier
 */
export const IconNodeRing: React.FC<IconProps> = ({ size = 16, color = "currentColor", className = "" }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none" className={className} stroke={color} strokeWidth="1.5">
    <circle cx="8" cy="8" r="6" />
    <circle cx="8" cy="8" r="2" fill={color} />
  </svg>
);

/**
 * Custom SVG Mark 2: Edge Ray
 * Interaction directional vector
 */
export const IconEdgeRay: React.FC<IconProps> = ({ size = 16, color = "currentColor", className = "" }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none" className={className} stroke={color} strokeWidth="1.5">
    <line x1="2" y1="14" x2="14" y2="2" />
    <polyline points="8,2 14,2 14,8" />
  </svg>
);

/**
 * Custom SVG Mark 3: Attention Cone
 * Graph Attention weight angle representation
 */
export const IconAttentionCone: React.FC<IconProps> = ({ size = 16, color = "currentColor", className = "" }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none" className={className} stroke={color} strokeWidth="1.5">
    <path d="M2 8 L14 3 L14 13 Z" strokeLinejoin="miter" />
    <line x1="8" y1="5.5" x2="8" y2="10.5" strokeDasharray="1.5,1.5" />
  </svg>
);

/**
 * Custom SVG Mark 4: Dossier Fold
 * Archival case file document container
 */
export const IconDossierFold: React.FC<IconProps> = ({ size = 16, color = "currentColor", className = "" }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none" className={className} stroke={color} strokeWidth="1.5">
    <path d="M2 3 L8 3 L10 5 L14 5 L14 13 L2 13 Z" />
    <line x1="5" y1="8" x2="11" y2="8" />
    <line x1="5" y1="10.5" x2="9" y2="10.5" />
  </svg>
);

/**
 * Custom SVG Mark 5: Stamp Seal
 * Forensic verdict certification stamp
 */
export const IconStampSeal: React.FC<IconProps> = ({ size = 16, color = "currentColor", className = "" }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none" className={className} stroke={color} strokeWidth="1.5">
    <rect x="2.5" y="2.5" width="11" height="11" />
    <circle cx="8" cy="8" r="3.5" />
  </svg>
);

/**
 * Custom SVG Mark 6: Search Reticle
 * Crosshair focus mark for forensic target filtering
 */
export const IconSearchReticle: React.FC<IconProps> = ({ size = 16, color = "currentColor", className = "" }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none" className={className} stroke={color} strokeWidth="1.5">
    <circle cx="8" cy="8" r="5" />
    <line x1="8" y1="1" x2="8" y2="4" />
    <line x1="8" y1="12" x2="8" y2="15" />
    <line x1="1" y1="8" x2="4" y2="8" />
    <line x1="12" y1="8" x2="15" y2="8" />
  </svg>
);

/**
 * Custom SVG Mark 7: Signal Burst
 * Coordinated anomaly burst and temporal spike mark
 */
export const IconSignalBurst: React.FC<IconProps> = ({ size = 16, color = "currentColor", className = "" }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none" className={className} stroke={color} strokeWidth="1.5">
    <line x1="8" y1="2" x2="8" y2="6" />
    <line x1="3" y1="13" x2="6" y2="9" />
    <line x1="13" y1="13" x2="10" y2="9" />
    <circle cx="8" cy="11" r="1.5" fill={color} />
  </svg>
);
