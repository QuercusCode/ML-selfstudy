import React from 'react';

function ResourceReader({ resourceUrl, resourceTitle, onClose }) {
  const isExternal = resourceUrl.startsWith('http://') || resourceUrl.startsWith('https://');

  return (
    <div className="reader-overlay">
      <div className="reader-header glass-panel">
        <div className="reader-title">
          <span style={{ fontWeight: '600', marginRight: '16px', color: 'var(--accent-color)' }}>Reading:</span> 
          {resourceTitle}
        </div>
        
        <div className="reader-actions">
          <a 
            href={resourceUrl} 
            target="_blank" 
            rel="noopener noreferrer" 
            className="external-link-btn"
            title="Open in new tab"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
              <polyline points="15 3 21 3 21 9"></polyline>
              <line x1="10" y1="14" x2="21" y2="3"></line>
            </svg>
            Open External
          </a>
          
          <button className="close-reader-btn" onClick={onClose} title="Close Reader">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>
      </div>

      <div className="reader-content" style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', background: '#0b0f19' }}>
        {isExternal ? (
          <div className="glass-panel" style={{
            maxWidth: '540px',
            width: '90%',
            padding: '48px 32px',
            textAlign: 'center',
            borderRadius: '16px',
            border: '1px solid var(--glass-border)',
            background: 'rgba(255, 255, 255, 0.02)',
            backdropFilter: 'blur(16px)',
            boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.4)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center'
          }}>
            <div style={{
              width: '80px',
              height: '80px',
              borderRadius: '50%',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--glass-border)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '24px'
            }}>
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="var(--accent-color)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="2" y1="12" x2="22" y2="12"></line>
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
              </svg>
            </div>
            
            <h3 style={{ fontSize: '1.6rem', marginBottom: '12px', color: '#fff', fontWeight: '600' }}>
              External Resource
            </h3>
            
            <p style={{ color: 'var(--text-secondary)', marginBottom: '32px', lineHeight: '1.6', fontSize: '1.05rem' }}>
              To protect user security and respect licensing rules, websites like <strong>GitHub</strong>, <strong>Nature</strong>, or <strong>YouTube</strong> do not permit their pages to be loaded directly inside frame applications.
            </p>
            
            <a 
              href={resourceUrl} 
              target="_blank" 
              rel="noopener noreferrer" 
              className="action-btn"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '10px',
                padding: '14px 28px',
                background: 'var(--accent-color)',
                color: '#fff',
                borderRadius: '8px',
                textDecoration: 'none',
                fontWeight: '600',
                fontSize: '1rem',
                border: 'none',
                cursor: 'pointer',
                transition: 'var(--transition)',
                boxShadow: '0 4px 14px 0 rgba(79, 70, 229, 0.4)'
              }}
            >
              Open in a New Tab
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                <polyline points="15 3 21 3 21 9"></polyline>
                <line x1="10" y1="14" x2="21" y2="3"></line>
              </svg>
            </a>
          </div>
        ) : (
          <iframe 
            src={resourceUrl} 
            title={resourceTitle}
            className="reader-iframe"
          />
        )}
      </div>
    </div>
  );
}

export default ResourceReader;
