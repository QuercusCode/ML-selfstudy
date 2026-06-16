import { useState, useEffect } from 'react';
import curriculumData from './data/curriculum.json';
import Sidebar from './components/Sidebar';
import WeekView from './components/WeekView';
import ResourcesView from './components/ResourcesView';
import ResourceReader from './components/ResourceReader';
import InteractiveLab from './components/InteractiveLab';
import SchedulerView from './components/SchedulerView';
import CheatSheetDrawer from './components/CheatSheetDrawer';

function App() {
  const [selectedWeekIdx, setSelectedWeekIdx] = useState(0);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [activeResource, setActiveResource] = useState(null); // { url, title }
  const [completedTasks, setCompletedTasks] = useState(() => {
    const saved = localStorage.getItem('ml-curriculum-completed');
    return saved ? JSON.parse(saved) : {};
  });

  const [bookmarks, setBookmarks] = useState(() => {
    const saved = localStorage.getItem('ml-curriculum-bookmarks');
    return saved ? JSON.parse(saved) : [];
  });

  const [activityDates, setActivityDates] = useState(() => {
    const saved = localStorage.getItem('ml-curriculum-activity');
    return saved ? JSON.parse(saved) : [];
  });

  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('ml-curriculum-theme') || 'cyberpunk';
  });

  useEffect(() => {
    localStorage.setItem('ml-curriculum-theme', theme);
    document.body.className = '';
    document.body.classList.add(`theme-${theme}`);
  }, [theme]);

  // Flatten weeks to make indexing easier
  const allWeeks = curriculumData.blocks.reduce((acc, block) => {
    return acc.concat(block.weeks.map(w => ({ ...w, blockTitle: block.title })));
  }, []);

  useEffect(() => {
    localStorage.setItem('ml-curriculum-completed', JSON.stringify(completedTasks));
  }, [completedTasks]);

  useEffect(() => {
    localStorage.setItem('ml-curriculum-bookmarks', JSON.stringify(bookmarks));
  }, [bookmarks]);

  useEffect(() => {
    localStorage.setItem('ml-curriculum-activity', JSON.stringify(activityDates));
  }, [activityDates]);

  const toggleBookmark = (wIdx) => {
    setBookmarks(prev => 
      prev.includes(wIdx) ? prev.filter(x => x !== wIdx) : [...prev, wIdx].sort((a, b) => a - b)
    );
  };

  const toggleTask = (weekIdx, taskIdx, itemIdx) => {
    const key = `${weekIdx}-${taskIdx}-${itemIdx}`;
    const wasCompleted = completedTasks[key];
    
    setCompletedTasks(prev => ({
      ...prev,
      [key]: !prev[key]
    }));

    if (!wasCompleted) {
      // User completed a task! Add today's date to activity logs
      const todayStr = new Date().toISOString().split('T')[0];
      setActivityDates(prev => {
        if (prev.includes(todayStr)) return prev;
        return [...prev, todayStr];
      });
    }
  };

  const calculateWeekProgress = (weekIdx) => {
    const week = allWeeks[weekIdx];
    let total = 0;
    let completed = 0;
    
    if(!week || !week.tasks) return 0;

    week.tasks.forEach((task, tIdx) => {
      task.items.forEach((_, iIdx) => {
        total++;
        if (completedTasks[`${weekIdx}-${tIdx}-${iIdx}`]) {
          completed++;
        }
      });
    });

    return total === 0 ? 0 : (completed / total) * 100;
  };

  const calculateOverallProgress = () => {
    let total = 0;
    let completed = 0;
    allWeeks.forEach((week, wIdx) => {
      if(!week.tasks) return;
      week.tasks.forEach((task, tIdx) => {
        task.items.forEach((_, iIdx) => {
          total++;
          if (completedTasks[`${wIdx}-${tIdx}-${iIdx}`]) {
            completed++;
          }
        });
      });
    });
    return total === 0 ? 0 : (completed / total) * 100;
  };

  const isWeekCompleted = (wIdx) => calculateWeekProgress(wIdx) === 100;

  return (
    <div className="app-container">
      <Sidebar 
        blocks={curriculumData.blocks} 
        allWeeks={allWeeks}
        selectedWeekIdx={selectedWeekIdx}
        setSelectedWeekIdx={setSelectedWeekIdx}
        isWeekCompleted={isWeekCompleted}
        overallProgress={calculateOverallProgress()}
        bookmarks={bookmarks}
        theme={theme}
        setTheme={setTheme}
        isDrawerOpen={isDrawerOpen}
        setIsDrawerOpen={setIsDrawerOpen}
      />
      <main className="main-content">
        {selectedWeekIdx === 'resources' ? (
          <ResourcesView openResource={(url, title) => setActiveResource({ url, title })} />
        ) : selectedWeekIdx === 'lab' ? (
          <InteractiveLab />
        ) : selectedWeekIdx === 'scheduler' ? (
          <SchedulerView allWeeks={allWeeks} completedTasks={completedTasks} activityDates={activityDates} />
        ) : (
          <WeekView 
            week={allWeeks[selectedWeekIdx]} 
            weekIdx={selectedWeekIdx}
            completedTasks={completedTasks}
            toggleTask={toggleTask}
            weekProgress={calculateWeekProgress(selectedWeekIdx)}
            isBookmarked={bookmarks.includes(selectedWeekIdx)}
            toggleBookmark={() => toggleBookmark(selectedWeekIdx)}
          />
        )}
      </main>

      {activeResource && (
        <ResourceReader 
          resourceUrl={activeResource.url} 
          resourceTitle={activeResource.title} 
          onClose={() => setActiveResource(null)} 
        />
      )}

      <CheatSheetDrawer isOpen={isDrawerOpen} onClose={() => setIsDrawerOpen(false)} />
    </div>
  );
}

export default App;
