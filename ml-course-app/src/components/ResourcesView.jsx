import React from 'react';
import resourcesData from '../data/resources.json';

function ResourcesView({ openResource }) {
  return (
    <div className="resources-view">
      <div style={{ marginBottom: '32px' }}>
        <h1 style={{ fontSize: '2.5rem', margin: 0 }}>Resources Library</h1>
        <p style={{ color: 'var(--text-secondary)', marginTop: '8px', fontSize: '1.1rem' }}>
          The complete master list of all textbooks, courses, tools, and papers referenced in the curriculum.
        </p>
      </div>

      {resourcesData.map((category, cIdx) => (
        <div key={cIdx} className="resource-category">
          <h2 className="category-title">{category.category}</h2>
          <div className="resources-grid">
            {category.items.map((item, iIdx) => (
              <button 
                key={iIdx} 
                onClick={() => openResource(item.url, item.title)}
                className="resource-card glass-panel"
                style={{ textAlign: 'left', background: 'transparent', border: '1px solid var(--glass-border)' }}
              >
                <div className="resource-type">{item.type}</div>
                <h3 className="resource-title">{item.title}</h3>
                <div className="resource-author">by {item.author}</div>
                <p className="resource-desc">{item.description}</p>
                <div className="resource-link-icon">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                    <polyline points="15 3 21 3 21 9"></polyline>
                    <line x1="10" y1="14" x2="21" y2="3"></line>
                  </svg>
                </div>
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export default ResourcesView;
