import React, { useState, useEffect } from 'react';

function SchedulerView({ allWeeks, completedTasks, activityDates }) {
  const [startDate, setStartDate] = useState(() => {
    const saved = localStorage.getItem('ml-scheduler-start');
    return saved || new Date().toISOString().split('T')[0];
  });
  const [pace, setPace] = useState(() => {
    const saved = localStorage.getItem('ml-scheduler-pace');
    return saved ? parseFloat(saved) : 1.0; // weeks per calendar week
  });

  useEffect(() => {
    localStorage.setItem('ml-scheduler-start', startDate);
    localStorage.setItem('ml-scheduler-pace', pace.toString());
  }, [startDate, pace]);

  // Calculate Streak
  const getStreak = () => {
    if (!activityDates || activityDates.length === 0) return 0;
    const uniqueDates = [...new Set(activityDates)].sort((a, b) => new Date(b) - new Date(a));
    
    const today = new Date();
    const todayStr = today.toISOString().split('T')[0];
    
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = yesterday.toISOString().split('T')[0];
    
    // Streak is active only if there's activity today or yesterday
    if (uniqueDates[0] !== todayStr && uniqueDates[0] !== yesterdayStr) {
      return 0;
    }

    let streak = 0;
    let current = new Date(uniqueDates[0]);

    for (let i = 0; i < uniqueDates.length; i++) {
      const dateStr = uniqueDates[i];
      const expectedStr = current.toISOString().split('T')[0];
      
      if (dateStr === expectedStr) {
        streak++;
        current.setDate(current.getDate() - 1);
      } else {
        break;
      }
    }
    return streak;
  };

  const streakVal = getStreak();

  // Helper to calculate target date for a given week index
  const getWeekTargetDate = (wIdx) => {
    const start = new Date(startDate);
    const daysOffset = Math.round((wIdx * 7) / pace);
    start.setDate(start.getDate() + daysOffset);
    return start;
  };

  // Check if a week is completed
  const isWeekCompleted = (wIdx) => {
    const week = allWeeks[wIdx];
    if (!week || !week.tasks) return false;
    let total = 0;
    let completed = 0;
    week.tasks.forEach((task, tIdx) => {
      task.items.forEach((_, iIdx) => {
        total++;
        if (completedTasks[`${wIdx}-${tIdx}-${iIdx}`]) {
          completed++;
        }
      });
    });
    return total > 0 && completed === total;
  };

  const completedWeeksCount = allWeeks.reduce((sum, _, idx) => sum + (isWeekCompleted(idx) ? 1 : 0), 0);
  const completionPercentage = allWeeks.length ? (completedWeeksCount / allWeeks.length) * 100 : 0;

  // Predict graduation date
  const projectedGradDate = getWeekTargetDate(allWeeks.length);

  return (
    <div className="scheduler-view-container text-white">
      <div style={{ marginBottom: '32px' }}>
        <h1 style={{ fontSize: '2.5rem', margin: 0 }}>Study Scheduler & Tracker</h1>
        <p style={{ color: 'var(--text-secondary)', marginTop: '8px', fontSize: '1.1rem' }}>
          Plan your 42-week journey, maintain streaks, and project your graduation targets.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px', marginBottom: '32px' }}>
        {/* Planner Settings */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.2rem', marginBottom: '16px', color: 'var(--accent-color)' }}>Planner Settings</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                Course Start Date
              </label>
              <input 
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                style={{ width: '100%', padding: '10px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '6px' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                Study Pace: {pace} week(s) per calendar week
              </label>
              <input 
                type="range"
                min="0.5"
                max="3"
                step="0.5"
                value={pace}
                onChange={(e) => setPace(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent-color)' }}
              />
            </div>
          </div>
        </div>

        {/* Learning Statistics */}
        <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <h3 style={{ fontSize: '1.2rem', marginBottom: '16px', color: 'var(--accent-color)' }}>Target Completion</h3>
            <div style={{ display: 'flex', gap: '20px', alignItems: 'center', marginBottom: '16px' }}>
              <div style={{ fontSize: '3rem' }}>🎓</div>
              <div>
                <div style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>
                  {projectedGradDate.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' })}
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                  Projected Graduation Date
                </div>
              </div>
            </div>
          </div>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>
              <span>Completed Weeks</span>
              <span>{completedWeeksCount} / {allWeeks.length} ({Math.round(completionPercentage)}%)</span>
            </div>
            <div style={{ height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${completionPercentage}%`, background: 'var(--accent-color)', borderRadius: '4px' }}></div>
            </div>
          </div>
        </div>

        {/* Daily Streak */}
        <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
          <div style={{
            fontSize: '4.5rem',
            animation: streakVal > 0 ? 'pulse 2s infinite' : 'none',
            filter: streakVal > 0 ? 'drop-shadow(0 0 15px rgba(239, 68, 68, 0.4))' : 'grayscale(1)'
          }}>
            🔥
          </div>
          <h3 style={{ fontSize: '1.8rem', margin: '8px 0 4px 0', background: 'linear-gradient(to right, #ff8a00, #da1b60)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            {streakVal} Day Streak
          </h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            {streakVal > 0 ? 'Keep the fire burning! Complete tasks daily.' : 'Complete any task today to start your learning streak!'}
          </p>
        </div>
      </div>

      {/* Week Target Table */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.3rem', marginBottom: '20px' }}>Curriculum Timeline Targets</h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', minWidth: '500px' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--glass-border)', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                <th style={{ padding: '12px' }}>Week</th>
                <th style={{ padding: '12px' }}>Topic</th>
                <th style={{ padding: '12px' }}>Target Date</th>
                <th style={{ padding: '12px' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {allWeeks.map((week, idx) => {
                const target = getWeekTargetDate(idx);
                const isDone = isWeekCompleted(idx);
                const isOverdue = !isDone && new Date() > target;
                
                return (
                  <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)', fontSize: '0.95rem' }}>
                    <td style={{ padding: '12px', fontWeight: '500' }}>Week {idx}</td>
                    <td style={{ padding: '12px' }}>{week.title}</td>
                    <td style={{ padding: '12px', color: isOverdue ? '#ff6b6b' : 'inherit' }}>
                      {target.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}
                    </td>
                    <td style={{ padding: '12px' }}>
                      {isDone ? (
                        <span style={{ color: '#10b981', background: 'rgba(16, 185, 129, 0.1)', padding: '4px 8px', borderRadius: '4px', fontSize: '0.8rem', fontWeight: 'bold' }}>
                          ✓ Completed
                        </span>
                      ) : isOverdue ? (
                        <span style={{ color: '#ef4444', background: 'rgba(239, 68, 68, 0.1)', padding: '4px 8px', borderRadius: '4px', fontSize: '0.8rem', fontWeight: 'bold' }}>
                          ⚠️ Overdue
                        </span>
                      ) : (
                        <span style={{ color: 'var(--text-secondary)', background: 'rgba(255,255,255,0.05)', padding: '4px 8px', borderRadius: '4px', fontSize: '0.8rem' }}>
                          Pending
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default SchedulerView;
