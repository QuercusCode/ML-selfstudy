import React from 'react';

function ProgressBar({ progress }) {
  return (
    <div className="overall-progress">
      <div className="progress-bar-bg">
        <div 
          className="progress-bar-fill" 
          style={{ width: `${progress}%` }}
        ></div>
      </div>
    </div>
  );
}

export default ProgressBar;
