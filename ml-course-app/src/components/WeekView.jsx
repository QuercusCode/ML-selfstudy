import React, { useState, useEffect } from 'react';
import LessonViewer from './LessonViewer';
import ProgressBar from './ProgressBar';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import quizData from '../data/quizzes.json';

function WeekView({ week, weekIdx, completedTasks, toggleTask, weekProgress, isBookmarked, toggleBookmark }) {
  const [activeTab, setActiveTab] = useState('tasks');
  const [openDay, setOpenDay] = useState('Day 1');
  const [lessonContent, setLessonContent] = useState('');
  const [notes, setNotes] = useState('');
  const [quizAnswers, setQuizAnswers] = useState({});
  const [quizSubmitted, setQuizSubmitted] = useState(false);

  // Load notes when week changes
  useEffect(() => {
    const savedNotes = localStorage.getItem(`ml-notes-week-${weekIdx}`);
    setNotes(savedNotes || '');
  }, [weekIdx]);

  const handleNotesChange = (val) => {
    setNotes(val);
    localStorage.setItem(`ml-notes-week-${weekIdx}`, val);
  };

  // Dynamically load markdown files
  useEffect(() => {
    const loadContent = async () => {
      try {
        // Vite specific dynamic import for raw text
        const modules = import.meta.glob('../content/*.md', { query: '?raw', import: 'default' });
        const filePath = `../content/week${weekIdx}.md`;
        
        if (modules[filePath]) {
          const content = await modules[filePath]();
          setLessonContent(content);
        } else {
          setLessonContent('');
        }
      } catch (err) {
        console.error("Failed to load lesson content", err);
        setLessonContent('');
      }
    };
    
    loadContent();
    setActiveTab('tasks'); // Reset tab when week changes
    setOpenDay('Day 1'); // Reset open day
    setQuizAnswers({});
    setQuizSubmitted(false);
  }, [weekIdx]);

  if (!week) return <div className="week-view">Select a week from the sidebar</div>;

  // Basic markdown parsing for bold text and inline code
  const parseMarkdown = (text) => {
    if (!text) return null;
    const parts = text.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i} style={{ color: 'var(--text-primary)' }}>{part.slice(2, -2)}</strong>;
      }
      // Basic code block parsing
      const codeParts = part.split(/(`.*?`)/g);
      return codeParts.map((cPart, j) => {
        if (cPart.startsWith('`') && cPart.endsWith('`')) {
          return <code key={j} style={{ background: 'rgba(255,255,255,0.1)', padding: '2px 6px', borderRadius: '4px', color: '#60a5fa' }}>{cPart.slice(1, -1)}</code>;
        }
        return cPart;
      });
    });
  };

  return (
    <div className="week-view">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <h2 style={{ fontSize: '2rem', marginBottom: '8px' }}>{week.title}</h2>
            <button 
              onClick={toggleBookmark}
              style={{
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                fontSize: '1.5rem',
                padding: 0,
                outline: 'none',
                transition: 'transform 0.2s',
                display: 'flex',
                alignItems: 'center'
              }}
              title={isBookmarked ? "Remove Bookmark" : "Bookmark Week"}
              onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.2)'}
              onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
            >
              {isBookmarked ? '📌' : '🪧'}
            </button>
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: '1.1rem' }}>{week.blockTitle}</div>
        </div>
        <div style={{ textAlign: 'right', minWidth: '120px' }}>
          <div style={{ marginBottom: '8px', fontWeight: '500' }}>Week Progress</div>
          <ProgressBar progress={weekProgress} />
        </div>
      </div>
      
      <div className="glass-panel" style={{ padding: '24px', marginBottom: '32px' }}>
        <h3 style={{ fontSize: '1.2rem', marginBottom: '8px', color: 'var(--accent-color)' }}>Weekly Goal</h3>
        <p style={{ lineHeight: '1.6' }}>{parseMarkdown(week.goal)}</p>
      </div>

      <div className="tabs-container">
        <button 
          className={`tab-btn ${activeTab === 'tasks' ? 'active' : ''}`}
          onClick={() => setActiveTab('tasks')}
        >
          Tasks & Schedule
        </button>
        <button 
          className={`tab-btn ${activeTab === 'lesson' ? 'active' : ''}`}
          onClick={() => setActiveTab('lesson')}
        >
          Course Lesson
        </button>
        <button 
          className={`tab-btn ${activeTab === 'notes' ? 'active' : ''}`}
          onClick={() => setActiveTab('notes')}
        >
          My Notes
        </button>
        <button 
          className={`tab-btn ${activeTab === 'quiz' ? 'active' : ''}`}
          onClick={() => setActiveTab('quiz')}
        >
          Weekly Quiz
        </button>
      </div>

      {activeTab === 'lesson' && (
        lessonContent ? (
          <LessonViewer content={lessonContent} />
        ) : (
          <div className="glass-panel" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
            <h3 style={{ marginBottom: '16px', color: 'var(--text-primary)' }}>Lesson Not Generated Yet</h3>
            <p>The course content for this week has not been generated by the instructor yet.</p>
            <p style={{ marginTop: '8px' }}>Ask Antigravity to <strong>"Generate the lesson for {week.title}"</strong> to populate this section.</p>
          </div>
        )
      )}

      {activeTab === 'notes' && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
            <div style={{ flex: 1, minWidth: '280px' }}>
              <h3 style={{ fontSize: '1.1rem', marginBottom: '12px', color: 'var(--accent-color)' }}>Edit Notes (Markdown supported)</h3>
              <textarea
                value={notes}
                onChange={(e) => handleNotesChange(e.target.value)}
                placeholder="Write your notes, math derivations, and personal concepts here..."
                style={{
                  width: '100%',
                  height: '400px',
                  padding: '16px',
                  background: 'rgba(255, 255, 255, 0.02)',
                  color: '#fff',
                  border: '1px solid var(--glass-border)',
                  borderRadius: '8px',
                  fontFamily: '"Fira Code", monospace',
                  fontSize: '0.9rem',
                  outline: 'none',
                  resize: 'vertical',
                  lineHeight: '1.5'
                }}
              />
            </div>
            <div style={{ flex: 1, minWidth: '280px', paddingLeft: '24px', borderLeft: '1px solid var(--glass-border)' }}>
              <h3 style={{ fontSize: '1.1rem', marginBottom: '12px', color: 'var(--text-secondary)' }}>Live Rendered Preview</h3>
              <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                {notes ? (
                  <LessonViewer content={notes} />
                ) : (
                  <p style={{ color: 'var(--text-secondary)', fontStyle: 'italic', marginTop: '16px' }}>No notes captured yet. Start writing on the left!</p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'tasks' && (
        <div className="daily-schedule">
          {(() => {
            const days = [
              { day: 'Day 1', title: 'Theory & Concepts', types: ['Read', 'Watch', 'Read / watch'] },
              { day: 'Day 2', title: 'Mathematical Foundations', types: ['Derive by hand', 'Understand'] },
              { day: 'Day 3', title: 'Implementation & Practice', types: ['Code', 'Build'] },
              { day: 'Day 4', title: 'Core Project', types: ['Deliverable'] },
              { day: 'Day 5', title: 'Review & Mastery', types: ['Self-test'] }
            ];

            const plan = days.map(dayConfig => {
              const items = [];
              week.tasks.forEach((task, tIdx) => {
                if (dayConfig.types.includes(task.type)) {
                  task.items.forEach((itemText, iIdx) => {
                    items.push({ text: itemText, type: task.type, tIdx, iIdx });
                  });
                }
              });
              return { ...dayConfig, items };
            });

            const mappedTypes = days.flatMap(d => d.types);
            const unmappedItems = [];
            week.tasks.forEach((task, tIdx) => {
              if (!mappedTypes.includes(task.type)) {
                task.items.forEach((itemText, iIdx) => {
                  unmappedItems.push({ text: itemText, type: task.type, tIdx, iIdx });
                });
              }
            });
            
            if (unmappedItems.length > 0) {
              plan.push({ day: 'Day 6', title: 'Additional Tasks', items: unmappedItems });
            }

            const activePlan = plan.filter(d => d.items.length > 0);
            
            return activePlan.map((day, dIdx) => {
              const isOpen = openDay === day.day;
              return (
                <div key={dIdx} className={`accordion-item ${isOpen ? 'open' : ''}`}>
                  <button 
                    className="accordion-header" 
                    onClick={() => setOpenDay(isOpen ? null : day.day)}
                  >
                    <div>
                      <span className="day-label">{day.day}:</span> {day.title}
                    </div>
                    <div className="accordion-icon">
                      {isOpen ? '▲' : '▼'}
                    </div>
                  </button>
                  
                  {isOpen && (
                    <div className="accordion-content">
                      <ul className="task-list">
                        {day.items.map((item, idx) => {
                          const isCompleted = completedTasks[`${weekIdx}-${item.tIdx}-${item.iIdx}`];
                          return (
                            <li key={idx} className={`task-item ${isCompleted ? 'completed' : ''}`}>
                              <input 
                                type="checkbox" 
                                className="task-checkbox"
                                checked={isCompleted || false}
                                onChange={() => toggleTask(weekIdx, item.tIdx, item.iIdx)}
                                id={`task-${weekIdx}-${item.tIdx}-${item.iIdx}`}
                              />
                              <div className="task-text-container">
                                <div className="task-type-badge">{item.type}</div>
                                <div className="task-text">{parseMarkdown(item.text)}</div>
                              </div>
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  )}
                </div>
              );
            });
          })()}
        </div>
      )}

      {activeTab === 'quiz' && (() => {
        const questions = quizData[weekIdx];
        if (!questions || questions.length === 0) {
          return (
            <div className="glass-panel" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
              <h3 style={{ marginBottom: '16px', color: 'var(--text-primary)' }}>Quiz Coming Soon</h3>
              <p>Self-assessment questions for this week are currently being compiled. Review the lesson or use the Matrix Playground to practice in the meantime!</p>
            </div>
          );
        }

        const handleOptionSelect = (qId, optionIdx) => {
          if (quizSubmitted) return;
          setQuizAnswers(prev => ({ ...prev, [qId]: optionIdx }));
        };

        const handleSubmit = () => {
          if (Object.keys(quizAnswers).length < questions.length) {
            alert("Please answer all questions before submitting!");
            return;
          }
          setQuizSubmitted(true);
        };

        const correctCount = questions.reduce((sum, q) => {
          return sum + (quizAnswers[q.id] === q.correct ? 1 : 0);
        }, 0);

        return (
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h3 style={{ fontSize: '1.4rem', marginBottom: '8px', color: 'var(--accent-color)' }}>Week {parseInt(weekIdx) + 1} Knowledge Check</h3>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '24px', fontSize: '0.95rem' }}>
              Test your understanding of eigenvalues, spaces, gradients, and proofs.
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              {questions.map((q, qIdx) => {
                const selectedOption = quizAnswers[q.id];
                const isQuestionCorrect = selectedOption === q.correct;
                
                return (
                  <div key={q.id} style={{ borderBottom: qIdx < questions.length - 1 ? '1px solid var(--glass-border)' : 'none', paddingBottom: '20px' }}>
                    <div style={{ fontWeight: '600', fontSize: '1.05rem', marginBottom: '12px', display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                      <span style={{ flexShrink: 0 }}>Q{qIdx + 1}:</span>
                      <div style={{ flex: 1 }}>
                        <ReactMarkdown children={q.question} remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]} />
                      </div>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginLeft: '12px' }}>
                      {q.options.map((opt, oIdx) => {
                        const isSelected = selectedOption === oIdx;
                        let btnBg = 'rgba(255, 255, 255, 0.02)';
                        let btnBorder = '1px solid var(--glass-border)';
                        
                        if (quizSubmitted) {
                          if (oIdx === q.correct) {
                            btnBg = 'rgba(16, 185, 129, 0.15)';
                            btnBorder = '1px solid var(--success-color)';
                          } else if (isSelected && !isQuestionCorrect) {
                            btnBg = 'rgba(239, 68, 68, 0.15)';
                            btnBorder = '1px solid #ef4444';
                          }
                        } else if (isSelected) {
                          btnBg = 'var(--accent-faint)';
                          btnBorder = '1px solid var(--accent-color)';
                        }

                        return (
                          <button
                            key={oIdx}
                            onClick={() => handleOptionSelect(q.id, oIdx)}
                            disabled={quizSubmitted}
                            style={{
                              textAlign: 'left',
                              padding: '12px 16px',
                              background: btnBg,
                              border: btnBorder,
                              borderRadius: '8px',
                              color: isSelected || (quizSubmitted && oIdx === q.correct) ? '#fff' : 'var(--text-secondary)',
                              fontSize: '0.9rem',
                              cursor: quizSubmitted ? 'default' : 'pointer',
                              transition: 'var(--transition)'
                            }}
                          >
                            <span style={{ marginRight: '8px', fontWeight: 'bold' }}>{String.fromCharCode(65 + oIdx)}.</span>
                            <span style={{ display: 'inline-block', verticalAlign: 'middle' }}>
                              <ReactMarkdown children={opt} remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]} />
                            </span>
                          </button>
                        );
                      })}
                    </div>

                    {quizSubmitted && (
                      <div style={{ marginTop: '16px', background: 'rgba(255, 255, 255, 0.02)', borderLeft: `4px solid ${isQuestionCorrect ? 'var(--success-color)' : '#ef4444'}`, padding: '16px', borderRadius: '0 8px 8px 0', fontSize: '0.9rem' }}>
                        <div style={{ fontWeight: 'bold', color: isQuestionCorrect ? 'var(--success-color)' : '#ef4444', marginBottom: '6px' }}>
                          {isQuestionCorrect ? 'Correct! ✓' : 'Incorrect ✗'}
                        </div>
                        <ReactMarkdown children={q.explanation} remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]} />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            <div style={{ borderTop: '1px solid var(--glass-border)', paddingTop: '24px', marginTop: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
              {quizSubmitted ? (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <span style={{ fontSize: '1.8rem' }}>{correctCount === questions.length ? '🏆' : '📝'}</span>
                    <div>
                      <div style={{ fontWeight: 'bold', fontSize: '1.1rem' }}>Score: {correctCount} / {questions.length}</div>
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        {correctCount === questions.length ? 'Perfect Score! You mastered this week.' : 'Review explanations above and try again!'}
                      </div>
                    </div>
                  </div>
                  <button
                    className="action-btn"
                    onClick={() => { setQuizAnswers({}); setQuizSubmitted(false); }}
                    style={{ padding: '10px 20px', fontSize: '0.9rem' }}
                  >
                    ↻ Retry Quiz
                  </button>
                </>
              ) : (
                <>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    Make sure you have completed the lesson content and practice tasks.
                  </div>
                  <button
                    className="action-btn"
                    onClick={handleSubmit}
                    style={{ padding: '10px 24px', fontSize: '0.95rem' }}
                  >
                    ✓ Submit Answers
                  </button>
                </>
              )}
            </div>
          </div>
        );
      })()}
    </div>
  );
}

export default WeekView;
