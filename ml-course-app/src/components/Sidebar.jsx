import React from 'react';
import ProgressBar from './ProgressBar';

function Sidebar({ blocks, allWeeks, selectedWeekIdx, setSelectedWeekIdx, isWeekCompleted, overallProgress, bookmarks, theme, setTheme, isDrawerOpen, setIsDrawerOpen }) {
  let globalWeekCounter = 0;

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h2 style={{ background: 'linear-gradient(to right, var(--text-primary), var(--accent-color))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', fontSize: '1.5rem', marginBottom: '8px' }}>
          ML/AI Curriculum
        </h2>
        <ProgressBar progress={overallProgress} />
        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px', textAlign: 'right', marginBottom: '16px' }}>
          {Math.round(overallProgress)}% Complete
        </div>

        {/* Visual Theme Customizer */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', background: 'rgba(255,255,255,0.02)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--glass-border)' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: '500' }}>Theme:</span>
          <div style={{ display: 'flex', gap: '8px' }}>
            {[
              { id: 'cyberpunk', color: '#8b5cf6', title: 'Glass Cyberpunk' },
              { id: 'nord', color: '#88c0d0', title: 'Nord Frost' },
              { id: 'monokai', color: '#ffd866', title: 'Monokai Pro' },
              { id: 'slate', color: '#3b82f6', title: 'Slate Minimalist' }
            ].map(t => (
              <button
                key={t.id}
                onClick={() => setTheme(t.id)}
                style={{
                  width: '16px',
                  height: '16px',
                  borderRadius: '50%',
                  background: t.color,
                  border: theme === t.id ? '2px solid #fff' : '1px solid rgba(0,0,0,0.5)',
                  cursor: 'pointer',
                  padding: 0,
                  outline: 'none',
                  boxShadow: theme === t.id ? '0 0 8px ' + t.color : 'none',
                  transition: 'transform 0.2s'
                }}
                title={t.title}
                onMouseEnter={(e) => e.target.style.transform = 'scale(1.25)'}
                onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
              />
            ))}
          </div>
        </div>
        
        <div 
          className={`week-item ${selectedWeekIdx === 'resources' ? 'active' : ''}`}
          onClick={() => setSelectedWeekIdx('resources')}
          style={{ marginTop: '16px', background: selectedWeekIdx === 'resources' ? 'var(--accent-faint)' : 'rgba(255,255,255,0.03)', border: '1px solid var(--glass-border)', padding: '12px 16px', borderRadius: '8px' }}
        >
          <span style={{ fontSize: '0.95rem', fontWeight: '500', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"></path></svg>
            Resources Library
          </span>
        </div>
        
        <div 
          className={`week-item ${selectedWeekIdx === 'lab' ? 'active' : ''}`}
          onClick={() => setSelectedWeekIdx('lab')}
          style={{ marginTop: '8px', background: selectedWeekIdx === 'lab' ? 'var(--accent-faint)' : 'rgba(255,255,255,0.03)', border: '1px solid var(--glass-border)', padding: '12px 16px', borderRadius: '8px' }}
        >
          <span style={{ fontSize: '0.95rem', fontWeight: '500', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
              <polyline points="2 17 12 22 22 17"></polyline>
              <polyline points="2 12 12 17 22 12"></polyline>
            </svg>
            Interactive Lab
          </span>
        </div>

        <div 
          className={`week-item ${selectedWeekIdx === 'scheduler' ? 'active' : ''}`}
          onClick={() => setSelectedWeekIdx('scheduler')}
          style={{ marginTop: '8px', background: selectedWeekIdx === 'scheduler' ? 'var(--accent-faint)' : 'rgba(255,255,255,0.03)', border: '1px solid var(--glass-border)', padding: '12px 16px', borderRadius: '8px' }}
        >
          <span style={{ fontSize: '0.95rem', fontWeight: '500', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
              <line x1="16" y1="2" x2="16" y2="6"></line>
              <line x1="8" y1="2" x2="8" y2="6"></line>
              <line x1="3" y1="10" x2="21" y2="10"></line>
            </svg>
            Study Scheduler
          </span>
        </div>

        <div 
          className={`week-item ${isDrawerOpen ? 'active' : ''}`}
          onClick={() => setIsDrawerOpen(!isDrawerOpen)}
          style={{ marginTop: '8px', background: isDrawerOpen ? 'var(--accent-faint)' : 'rgba(255,255,255,0.03)', border: '1px solid var(--glass-border)', padding: '12px 16px', borderRadius: '8px' }}
        >
          <span style={{ fontSize: '0.95rem', fontWeight: '500', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"></path></svg>
            Math & Code Reference
          </span>
        </div>

        {bookmarks && bookmarks.length > 0 && (
          <div style={{ marginTop: '16px', borderTop: '1px solid var(--glass-border)', paddingTop: '16px' }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 'bold', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              📌 Bookmarks
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {bookmarks.map(wIdx => {
                const week = allWeeks[wIdx];
                if (!week) return null;
                return (
                  <div
                    key={wIdx}
                    className={`week-item ${selectedWeekIdx === wIdx ? 'active' : ''}`}
                    onClick={() => setSelectedWeekIdx(wIdx)}
                    style={{ padding: '8px 12px', fontSize: '0.85rem' }}
                  >
                    <span>{week.title}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
      <div className="sidebar-content">
        {blocks.map((block, bIdx) => (
          <div key={bIdx} className="block-group">
            <div className="block-title">{block.title}</div>
            {block.weeks.map((week, wIdx) => {
              const currentIdx = globalWeekCounter++;
              const isCompleted = isWeekCompleted(currentIdx);
              const isActive = selectedWeekIdx === currentIdx;
              
              return (
                <div 
                  key={currentIdx}
                  className={`week-item ${isActive ? 'active' : ''}`}
                  onClick={() => setSelectedWeekIdx(currentIdx)}
                >
                  <span style={{ fontSize: '0.9rem' }}>{week.title}</span>
                  <div className={`progress-circle ${isCompleted ? 'completed' : ''}`}></div>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

export default Sidebar;
