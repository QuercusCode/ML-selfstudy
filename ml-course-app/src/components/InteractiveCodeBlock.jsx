import React, { useState } from 'react';
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python';
import atomDark from 'react-syntax-highlighter/dist/esm/styles/prism/atom-dark';

SyntaxHighlighter.registerLanguage('python', python);
import { usePyodide } from '../hooks/usePyodide';

function InteractiveCodeBlock({ initialCode }) {
  const [code, setCode] = useState(initialCode.replace(/\n$/, ''));
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const { loading, error, executeCode } = usePyodide();

  const handleRun = async () => {
    setIsRunning(true);
    setOutput('');
    try {
      const res = await executeCode(code);
      setOutput(res || "Code executed successfully (no output).");
    } catch (err) {
      setOutput(err.toString());
    }
    setIsRunning(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const start = e.target.selectionStart;
      const end = e.target.selectionEnd;
      const newCode = code.substring(0, start) + '    ' + code.substring(end);
      setCode(newCode);
      // Move cursor right after the 4 spaces
      setTimeout(() => {
        e.target.selectionStart = e.target.selectionEnd = start + 4;
      }, 0);
    }
  };

  return (
    <div className="interactive-code-block">
      <div className="code-header">
        <span className="lang-label">python</span>
        <button 
          className="run-btn" 
          onClick={handleRun} 
          disabled={loading || isRunning}
        >
          {loading ? (
            <><span className="spinner"></span> Loading Runtime...</>
          ) : isRunning ? (
            'Running...'
          ) : (
            '▶ Run Code'
          )}
        </button>
      </div>
      
      <div className="code-editor-container" style={{ position: 'relative', backgroundColor: '#1d1f21', minHeight: '100px' }}>
        <SyntaxHighlighter
          language="python"
          style={atomDark}
          customStyle={{
            margin: 0,
            padding: '16px',
            fontFamily: '"Fira Code", monospace',
            fontSize: '0.95rem',
            lineHeight: '1.5',
            backgroundColor: 'transparent',
            minHeight: '100px',
            pointerEvents: 'none' // Let clicks pass through to textarea
          }}
        >
          {code || ' '}
        </SyntaxHighlighter>
        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          onKeyDown={handleKeyDown}
          spellCheck={false}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            padding: '16px',
            margin: 0,
            border: 'none',
            background: 'transparent',
            color: 'transparent', // The magic trick to make it look like an editor
            caretColor: '#fff',
            fontFamily: '"Fira Code", monospace',
            fontSize: '0.95rem',
            lineHeight: '1.5',
            resize: 'none',
            outline: 'none',
            overflow: 'hidden',
            whiteSpace: 'pre',
            wrap: 'off'
          }}
        />
      </div>

      {output && (
        <div className="code-output">
          <div className="output-label">Output</div>
          {output.split('[[IMAGE:').map((part, index) => {
            if (index === 0) return part ? <pre key={index}>{part}</pre> : null;
            
            const endIndex = part.indexOf(']]');
            if (endIndex === -1) return <pre key={index}>[[IMAGE:{part}</pre>;
            
            const b64 = part.substring(0, endIndex);
            const rest = part.substring(endIndex + 2);
            
            return (
              <React.Fragment key={index}>
                <div style={{ margin: '16px 0', background: 'white', display: 'inline-block', padding: '16px', borderRadius: '8px' }}>
                  <img src={`data:image/png;base64,${b64}`} alt="matplotlib plot" style={{ maxWidth: '100%', display: 'block' }} />
                </div>
                {rest.trim() && <pre>{rest}</pre>}
              </React.Fragment>
            );
          })}
        </div>
      )}
      
      {error && (
        <div className="code-error">
          Failed to load Python runtime. Please refresh.
        </div>
      )}
    </div>
  );
}

export default InteractiveCodeBlock;
