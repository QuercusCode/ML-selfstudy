import { useState, useEffect } from 'react';

// Global singleton so Pyodide is only loaded once across the app
let pyodideInstance = null;
let loadPromise = null;

export function usePyodide() {
  const [pyodide, setPyodide] = useState(pyodideInstance);
  const [loading, setLoading] = useState(!pyodideInstance);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (pyodideInstance) {
      setPyodide(pyodideInstance);
      setLoading(false);
      return;
    }

    if (!loadPromise) {
      loadPromise = new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js';
        script.onload = async () => {
          try {
            const py = await window.loadPyodide();
            // Preload common ML packages
            await py.loadPackage(['numpy']);
            
            // Setup an output capture mechanism
            await py.runPythonAsync(`
import sys
import io

class OutputCapture:
    def __init__(self):
        self.output = ""
    def write(self, text):
        self.output += text
    def flush(self):
        pass

sys_capture = OutputCapture()
sys.stdout = sys_capture
sys.stderr = sys_capture
            `);
            
            pyodideInstance = py;
            resolve(py);
          } catch (err) {
            reject(err);
          }
        };
        script.onerror = () => reject(new Error('Failed to load Pyodide from CDN'));
        document.body.appendChild(script);
      });
    }

    loadPromise
      .then((py) => {
        setPyodide(py);
        setLoading(false);
      })
      .catch((err) => {
        setError(err);
        setLoading(false);
      });
  }, []);

  const executeCode = async (code) => {
    if (!pyodideInstance) throw new Error("Pyodide not loaded yet");
    
    // 1. Dynamically download any packages (like matplotlib, scipy) imported in the code!
    await pyodideInstance.loadPackagesFromImports(code);

    // 2. If matplotlib is used, monkey-patch plt.show() to output base64 images
    if (code.includes('matplotlib')) {
      await pyodideInstance.runPythonAsync(`
try:
    import matplotlib.pyplot as plt
    import io, base64
    def custom_show(*args, **kwargs):
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        print(f"\\n[[IMAGE:{b64}]]\\n")
        plt.clf() # Clear for next plot
    plt.show = custom_show
except Exception:
    pass
      `);
    }

    // Clear previous output
    await pyodideInstance.runPythonAsync(`sys_capture.output = ""`);
    
    try {
      // Execute the user's code
      await pyodideInstance.runPythonAsync(code);
    } catch (err) {
      // Append runtime errors to the output so the user sees them
      await pyodideInstance.runPythonAsync(`print("""${err.toString().replace(/"/g, '\\"')}""")`);
    }
    
    // Retrieve the captured output
    const output = await pyodideInstance.runPythonAsync(`sys_capture.output`);
    return output;
  };

  return { pyodide, loading, error, executeCode };
}
