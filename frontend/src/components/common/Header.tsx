import React from 'react';
import { NavLink, Link } from 'react-router-dom';
import { IconNodeRing } from './Icons';
import { useAppStore } from '../../store/useAppStore';
import { DatasetSwitcher } from './DatasetSwitcher';

export const Header: React.FC = () => {
  const { activeModel } = useAppStore();

  return (
    <header style={{
      backgroundColor: 'var(--paper)',
      borderBottom: '1.5px solid var(--ink)',
      padding: '0 24px',
      height: '54px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      position: 'sticky',
      top: 0,
      zIndex: 100,
    }}>
      {/* Brand Identity */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <Link to="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--ink)' }}>
          <IconNodeRing size={18} color="var(--stamp)" />
          <span style={{
            fontFamily: 'var(--font-display)',
            fontSize: '18px',
            fontWeight: 700,
            letterSpacing: '0.04em',
            textTransform: 'uppercase',
          }}>
            Graphwarden
          </span>
        </Link>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '10px',
          color: 'var(--ink-muted)',
          letterSpacing: '0.08em',
          borderLeft: '1px solid var(--rule-light)',
          paddingLeft: '12px',
        }}>
          GNN FORENSIC LABORATORY // v2.0
        </span>
      </div>

      {/* Navigation Tabs */}
      <nav style={{ display: 'flex', gap: '2px', height: '100%' }}>
        {[
          { path: '/app', label: 'ANALYST WORKBENCH' },
          { path: '/app/communities', label: 'COMMUNITIES' },
          { path: '/app/coordination', label: 'COORDINATION' },
          { path: '/app/temporal', label: 'TEMPORAL' },
          { path: '/app/models', label: 'MODELS' },
          { path: '/app/datasets', label: 'DATASETS' },
        ].map((tab) => (
          <NavLink
            key={tab.path}
            to={tab.path}
            end={tab.path === '/app'}
            style={({ isActive }) => ({
              display: 'flex',
              alignItems: 'center',
              padding: '0 14px',
              fontFamily: 'var(--font-mono)',
              fontSize: '11px',
              fontWeight: 500,
              letterSpacing: '0.05em',
              textDecoration: 'none',
              color: isActive ? 'var(--paper)' : 'var(--ink)',
              backgroundColor: isActive ? 'var(--ink)' : 'transparent',
              transition: 'background-color 100ms ease',
              borderLeft: '1px solid var(--rule-light)',
            })}
          >
            {tab.label}
          </NavLink>
        ))}
      </nav>

      {/* Active Model (Demoted to plain text) + Dataset Switcher (Elevated context control) */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '10.5px',
          color: 'var(--ink-muted)',
          letterSpacing: '0.04em',
        }}>
          ENGINE: <strong style={{ color: 'var(--stamp)', fontWeight: 700 }}>{activeModel}</strong>
        </span>
        <DatasetSwitcher />
      </div>
    </header>
  );
};

