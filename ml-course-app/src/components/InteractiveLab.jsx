import React, { useState, useEffect, useRef, lazy, Suspense } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { usePyodide } from '../hooks/usePyodide';
const DataVisualizer3D = lazy(() => import('./DataVisualizer3D'));
const challenges = [
  {
    id: 'proj',
    title: '1. Vector Projection',
    description: 'Project a vector $u$ onto a vector $v$. The projection formula is:\n\n$$\\text{proj}_v u = \\frac{u \\cdot v}{\\|v\\|^2} v$$\n\nImplement a function `project_vector(u, v)` that returns the projected vector as a NumPy array.',
    starterCode: `import numpy as np

def project_vector(u, v):
    # Write your code here
    # u and v are 1D numpy arrays of length N
    pass
`,
    testSuite: `
# Test suite for Vector Projection
u1 = np.array([2, 1])
v1 = np.array([1, 0])
ans1 = project_vector(u1, v1)
expected1 = np.array([2.0, 0.0])

u2 = np.array([3, 4])
v2 = np.array([1, 1])
ans2 = project_vector(u2, v2)
expected2 = np.array([3.5, 3.5])

assert np.allclose(ans1, expected1), f"Test 1 failed: Expected {expected1}, got {ans1}"
assert np.allclose(ans2, expected2), f"Test 2 failed: Expected {expected2}, got {ans2}"
print("All projection test cases passed successfully! 🎉")
`
  },
  {
    id: 'softmax',
    title: '2. Numerically Stable Softmax',
    description: 'Implement the Softmax activation function for a vector $x$:\n\n$$\\sigma(x)_i = \\frac{e^{x_i - \\max(x)}}{\\sum_{j} e^{x_j - \\max(x)}}$$\n\nSubtracting the maximum of $x$ avoids large exponential values that cause overflow. Implement `stable_softmax(x)`.',
    starterCode: `import numpy as np

def stable_softmax(x):
    # Write your code here
    # x is a 1D numpy array
    # Return the softmax probability vector
    pass
`,
    testSuite: `
# Test suite for Softmax
x1 = np.array([1000, 1000, 1000])
ans1 = stable_softmax(x1)
expected1 = np.array([1/3, 1/3, 1/3])

x2 = np.array([1, 2, 3])
ans2 = stable_softmax(x2)
expected2 = np.exp(x2 - np.max(x2)) / np.sum(np.exp(x2 - np.max(x2)))

assert np.allclose(ans1, expected1), f"Test 1 failed: Expected {expected1}, got {ans1}"
assert np.allclose(ans2, expected2), f"Test 2 failed: Expected {expected2}, got {ans2}"
print("All softmax test cases passed successfully! 🎉")
`
  },
  {
    id: 'mse',
    title: '3. MSE Loss & Gradient',
    description: 'Compute Mean Squared Error (MSE) loss and gradients with respect to weights $w$ and bias $b$ for linear regression $y = wx + b$:\n\n$$L = \\frac{1}{N} \\sum_{i=1}^N (w x_i + b - y_i)^2$$\n\n$$\\frac{\\partial L}{\\partial w} = \\frac{2}{N} \\sum_{i=1}^N x_i (w x_i + b - y_i)$$\n\n$$\\frac{\\partial L}{\\partial b} = \\frac{2}{N} \\sum_{i=1}^N (w x_i + b - y_i)$$\n\nImplement `linear_loss_and_grad(w, b, X, y)` returning `(loss, dw, db)`.',
    starterCode: `import numpy as np

def linear_loss_and_grad(w, b, X, y):
    # X, y are 1D numpy arrays of length N
    # Return tuple: (loss, dw, db)
    pass
`,
    testSuite: `
# Test suite for MSE Loss & Gradient
X = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
w, b = 2.0, 0.0
loss1, dw1, db1 = linear_loss_and_grad(w, b, X, y)
assert np.allclose(loss1, 0.0) and np.allclose(dw1, 0.0) and np.allclose(db1, 0.0), "Test 1 failed: Error should be 0 when prediction matches target."

w2, b2 = 1.0, 1.0
loss2, dw2, db2 = linear_loss_and_grad(w2, b2, X, y)
assert np.allclose(loss2, 5/3), f"Test 2 loss failed: got {loss2}"
assert np.allclose(dw2, -16/3), f"Test 2 dw failed: got {dw2}"
assert np.allclose(db2, -2.0), f"Test 2 db failed: got {db2}"
print("All MSE Loss & Gradient test cases passed successfully! 🎉")
`
  },
  {
    id: 'transform',
    title: '4. 2D Coordinate Grid Transformation',
    description: 'Transform coordinates using a transformation matrix $M = \\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}$. For coordinates $X$ of shape $(N, 2)$, compute transformed coordinates $X\' = X M^T$ (which maps to row matrix multiplication $x M^T$).\n\nImplement `transform_grid(X, M)`.',
    starterCode: `import numpy as np

def transform_grid(X, M):
    # X: shape (N, 2) numpy array
    # M: shape (2, 2) transformation matrix
    # Return transformed coordinates of shape (N, 2)
    pass
`,
    testSuite: `
# Test suite for Matrix Transform
X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
M = np.array([[0.0, -1.0], [1.0, 0.0]]) # 90-deg rotation
ans = transform_grid(X, M)
expected = np.array([[0.0, 1.0], [-1.0, 0.0], [-1.0, 1.0]])
assert np.allclose(ans, expected), f"Test failed: Expected {expected} got {ans}"
print("All Coordinate Grid Transformation test cases passed successfully! 🎉")
`
  }
];

function InteractiveLab() {
  const [activeTab, setActiveTab] = useState('sandbox'); // 'sandbox' | 'solver' | 'compiler' | 'nnbuilder' | 'scratchpad' | '3dexplorer'

  // --- PYODIDE HOOK ---
  const { loading: pyodideLoading, error: pyodideError, executeCode: runPython } = usePyodide();
  const [scratchpadCode, setScratchpadCode] = useState(`import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset
X = np.linspace(-3, 3, 100)
Y = 2 * X + 1 + np.random.normal(0, 0.5, 100)

plt.scatter(X, Y, color='purple', alpha=0.6, label='Data points')
plt.plot(X, 2 * X + 1, color='cyan', label='True line: y = 2x + 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Matplotlib Visualizer Sandbox')
plt.legend()
plt.show()`);
  const [scratchpadOutput, setScratchpadOutput] = useState('');
  const [isScratchpadRunning, setIsScratchpadRunning] = useState(false);
  const [scratchpadMode, setScratchpadMode] = useState('free'); // 'free' | 'guided'
  const [currentChallengeIdx, setCurrentChallengeIdx] = useState(0);

  // --- TAB 4: NN BUILDER STATE ---
  const [layers, setLayers] = useState([
    { id: '1', type: 'input', width: 28, height: 28, channels: 1 },
    { id: '2', type: 'conv2d', filters: 16, kernelSize: 3, stride: 1, padding: 'same' },
    { id: '3', type: 'maxpool2d', poolSize: 2, stride: 2 },
    { id: '4', type: 'flatten' },
    { id: '5', type: 'dense', units: 64, activation: 'relu' },
    { id: '6', type: 'dense', units: 10, activation: 'softmax' }
  ]);
  const [exportFramework, setExportFramework] = useState('pytorch'); // 'pytorch' | 'keras' | 'jax'

  // --- TAB 1: SANDBOX STATE ---
  // Modes: 'linear' | 'logistic' | 'optimizer' | 'neural' | 'kmeans' | 'svm' | 'tree'
  const [sandboxMode, setSandboxMode] = useState('linear'); 
  
  // Datasets & Points
  const [points, setPoints] = useState([
    { x: 100, y: 300, label: 0 },
    { x: 150, y: 250, label: 0 },
    { x: 200, y: 200, label: 0 },
    { x: 300, y: 180, label: 1 },
    { x: 350, y: 120, label: 1 },
    { x: 400, y: 80, label: 1 },
  ]);
  const [draggingIdx, setDraggingIdx] = useState(null);
  const [activeLabel, setActiveLabel] = useState(0); // 0 (Red) or 1 (Blue)

  // 1. Linear OLS parameters
  const [linearLine, setLinearLine] = useState({ slope: 0, intercept: 0 });

  // 2. Logistic parameters
  const [logisticBoundary, setLogisticBoundary] = useState({ w1: 0, w2: 0, b: 0 });

  // 3. Optimizer parameters
  const [optStart, setOptStart] = useState({ x: -2.0, y: 1.8 });
  const [optMethod, setOptMethod] = useState('adam');
  const [optLr, setOptLr] = useState(0.05);
  const [optPath, setOptPath] = useState([]);
  const [optFunction, setOptFunction] = useState('bowl'); // 'bowl' | 'rosenbrock' | 'saddle'
  const [isOptimizing, setIsOptimizing] = useState(false);
  const optimizationInterval = useRef(null);

  // 4. Neural Network parameters
  const [nnActivation, setNnActivation] = useState('tanh'); // 'tanh' | 'relu' | 'sigmoid'
  const [nnHiddenSize, setNnHiddenSize] = useState(4);
  const [nnGrid, setNnGrid] = useState([]); // 20x20 grid probabilities
  const [nnWeights, setNnWeights] = useState(null);

  // 5. K-Means parameters
  const [kmeansK, setKmeansK] = useState(3);
  const [centroids, setCentroids] = useState([]);
  const [kmeansLabels, setKmeansLabels] = useState([]); // indices of closest centroid
  const [kmeansStep, setKmeansStep] = useState(0); // 0: init, 1: assigned, 2: updated

  // 6. SVM parameters
  const [svmC, setSvmC] = useState(1.0);
  const [svmBoundary, setSvmBoundary] = useState({ w1: 0, w2: 0, b: 0 });
  const [supportVectors, setSupportVectors] = useState([]);

  // 7. Decision Tree parameters
  const [treeMaxDepth, setTreeMaxDepth] = useState(3);
  const [treePartitions, setTreePartitions] = useState([]); // splits to draw

  // --- TAB 2: MATH SOLVER STATE ---
  const [matrix, setMatrix] = useState({ a: 2, b: 1, c: 1, d: 2 });
  const [vector, setVector] = useState({ x1: 3, x2: 1 });
  const [projVector, setProjVector] = useState({ b1: 1, b2: 2 });
  const [solverMarkdown, setSolverMarkdown] = useState('');

  // --- TAB 3: COMPILER STATE ---
  const [compilerProgress, setCompilerProgress] = useState(null); 
  const [compilerText, setCompilerText] = useState('');

  // ----------------------------------------------------
  // --- ALGORITHM SIMULATIONS & CALCS ---
  // ----------------------------------------------------

  // 1. Linear OLS solver
  useEffect(() => {
    if (sandboxMode !== 'linear' || points.length < 2) return;
    const nPoints = points.map(p => ({ x: p.x / 500, y: (400 - p.y) / 400 }));
    const N = nPoints.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    nPoints.forEach(p => {
      sumX += p.x;
      sumY += p.y;
      sumXY += p.x * p.y;
      sumXX += p.x * p.x;
    });
    const num = N * sumXY - sumX * sumY;
    const den = N * sumXX - sumX * sumX;
    let m = 0, c = 0;
    if (Math.abs(den) > 1e-6) {
      m = num / den;
      c = (sumY - m * sumX) / N;
    } else {
      m = 999;
      c = 0;
    }
    setLinearLine({ slope: m, intercept: c });
  }, [points, sandboxMode]);

  // 2. Logistic Regression solver
  useEffect(() => {
    if (sandboxMode !== 'logistic' || points.length < 2) return;
    const X = points.map(p => [p.x / 500, (400 - p.y) / 400]);
    const y = points.map(p => p.label);
    let w1 = 0.0, w2 = 0.0, b = 0.0;
    const lr = 0.2, epochs = 1000;
    for (let epoch = 0; epoch < epochs; epoch++) {
      let gradW1 = 0, gradW2 = 0, gradB = 0;
      for (let i = 0; i < X.length; i++) {
        const xi = X[i], yi = y[i];
        const z = w1 * xi[0] + w2 * xi[1] + b;
        const pred = 1 / (1 + Math.exp(-z));
        const err = pred - yi;
        gradW1 += err * xi[0];
        gradW2 += err * xi[1];
        gradB += err;
      }
      w1 -= lr * (gradW1 / X.length);
      w2 -= lr * (gradW2 / X.length);
      b -= lr * (gradB / X.length);
    }
    setLogisticBoundary({ w1, w2, b });
  }, [points, sandboxMode]);

  // 3. Neural Network Trainer and Grid Evaluator
  useEffect(() => {
    if (sandboxMode !== 'neural' || points.length < 2) return;
    const X = points.map(p => [p.x / 500 - 0.5, (400 - p.y) / 400 - 0.5]); // centered inputs
    const y = points.map(p => p.label);

    // Initial random weights (Input 2 -> Hidden NnHiddenSize -> Output 1)
    let W1 = Array.from({ length: 2 }, () => Array.from({ length: nnHiddenSize }, () => (Math.random() - 0.5) * 2));
    let B1 = Array(nnHiddenSize).fill(0);
    let W2 = Array.from({ length: nnHiddenSize }, () => (Math.random() - 0.5) * 2);
    let B2 = 0;

    const act = (z) => {
      if (nnActivation === 'tanh') return Math.tanh(z);
      if (nnActivation === 'sigmoid') return 1 / (1 + Math.exp(-z));
      return Math.max(0, z); // ReLU
    };

    const dAct = (a) => {
      if (nnActivation === 'tanh') return 1 - a * a;
      if (nnActivation === 'sigmoid') return a * (1 - a);
      return a > 0 ? 1 : 0; // ReLU derivative
    };

    const lr = 0.1;
    const epochs = 500;

    // Mini Backpropagation loop
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let i = 0; i < X.length; i++) {
        const xi = X[i];
        const yi = y[i];

        // Hidden Layer
        const h = [];
        const z1 = [];
        for (let j = 0; j < nnHiddenSize; j++) {
          const z = xi[0] * W1[0][j] + xi[1] * W1[1][j] + B1[j];
          z1.push(z);
          h.push(act(z));
        }

        // Output Layer
        let z2 = B2;
        for (let j = 0; j < nnHiddenSize; j++) z2 += h[j] * W2[j];
        const pred = 1 / (1 + Math.exp(-z2)); // output layer always sigmoid for BC

        // Gradients Output
        const dLoss_dz2 = pred - yi;
        const dLoss_dW2 = h.map(hj => dLoss_dz2 * hj);
        const dLoss_dB2 = dLoss_dz2;

        // Gradients Hidden
        const dLoss_dh = W2.map(w2j => dLoss_dz2 * w2j);
        const dLoss_dz1 = dLoss_dh.map((dh, j) => dh * dAct(h[j]));

        const dLoss_dW1 = [
          dLoss_dz1.map(dz => dz * xi[0]),
          dLoss_dz1.map(dz => dz * xi[1])
        ];
        const dLoss_dB1 = dLoss_dz1;

        // Updates
        for (let j = 0; j < nnHiddenSize; j++) {
          W1[0][j] -= lr * dLoss_dW1[0][j];
          W1[1][j] -= lr * dLoss_dW1[1][j];
          B1[j] -= lr * dLoss_dB1[j];
          W2[j] -= lr * dLoss_dW2[j];
        }
        B2 -= lr * dLoss_dB2;
      }
    }

    // Evaluate Grid (20x20 squares)
    const newGrid = [];
    for (let gy = 0; gy < 20; gy++) {
      for (let gx = 0; gx < 20; gx++) {
        const px = (gx + 0.5) / 20 - 0.5;
        const py = (20 - (gy + 0.5)) / 20 - 0.5;

        // Forward
        const h = [];
        for (let j = 0; j < nnHiddenSize; j++) {
          h.push(act(px * W1[0][j] + py * W1[1][j] + B1[j]));
        }
        let z2 = B2;
        for (let j = 0; j < nnHiddenSize; j++) z2 += h[j] * W2[j];
        const pred = 1 / (1 + Math.exp(-z2));
        newGrid.push({ gx, gy, pred });
      }
    }

    setNnGrid(newGrid);
  }, [points, sandboxMode, nnActivation, nnHiddenSize]);

  // 4. K-Means clustering state updater
  const initKmeans = () => {
    if (points.length < kmeansK) return;
    const newCentroids = [];
    for (let k = 0; k < kmeansK; k++) {
      const idx = Math.floor(Math.random() * points.length);
      newCentroids.push({ x: points[idx].x, y: points[idx].y, id: k });
    }
    setCentroids(newCentroids);
    setKmeansLabels(Array(points.length).fill(-1));
    setKmeansStep(1); // Ready to Assign
  };

  const stepKmeans = () => {
    if (centroids.length === 0) return;

    if (kmeansStep === 1) {
      // Step 1: Assign points to nearest centroid
      const newLabels = points.map(p => {
        let minDist = Infinity;
        let bestK = 0;
        centroids.forEach((c, kIdx) => {
          const dist = Math.pow(p.x - c.x, 2) + Math.pow(p.y - c.y, 2);
          if (dist < minDist) {
            minDist = dist;
            bestK = kIdx;
          }
        });
        return bestK;
      });
      setKmeansLabels(newLabels);
      setKmeansStep(2); // Ready to Update
    } else {
      // Step 2: Update centroids location to mean of assigned points
      const newCentroids = centroids.map((c, kIdx) => {
        const assignedPoints = points.filter((_, idx) => kmeansLabels[idx] === kIdx);
        if (assignedPoints.length === 0) return c; // keep static if no points assigned
        const meanX = assignedPoints.reduce((sum, p) => sum + p.x, 0) / assignedPoints.length;
        const meanY = assignedPoints.reduce((sum, p) => sum + p.y, 0) / assignedPoints.length;
        return { ...c, x: meanX, y: meanY };
      });
      setCentroids(newCentroids);
      setKmeansStep(1); // Ready to Assign again
    }
  };

  // 5. SVM soft-margin linear classifier via Hinge Loss GD
  useEffect(() => {
    if (sandboxMode !== 'svm' || points.length < 2) return;
    const X = points.map(p => [p.x / 500 - 0.5, (400 - p.y) / 400 - 0.5]);
    const y = points.map(p => (p.label === 0 ? -1 : 1)); // SVM targets are -1 or 1

    let w1 = 0.0, w2 = 0.0, b = 0.0;
    const lr = 0.01;
    const epochs = 2000;

    for (let epoch = 0; epoch < epochs; epoch++) {
      let gradW1 = w1; // regularizer path
      let gradW2 = w2;
      let gradB = 0;

      for (let i = 0; i < X.length; i++) {
        const xi = X[i], yi = y[i];
        const val = yi * (w1 * xi[0] + w2 * xi[1] + b);

        if (val < 1) {
          // Hinge loss gradient violation
          gradW1 -= svmC * yi * xi[0];
          gradW2 -= svmC * yi * xi[1];
          gradB -= svmC * yi;
        }
      }

      w1 -= lr * gradW1;
      w2 -= lr * gradW2;
      b -= lr * gradB;
    }

    setSvmBoundary({ w1, w2, b });

    // Identify Support Vectors: points lying inside or exactly on the margin (yi * f(xi) <= 1.05)
    const svs = [];
    for (let i = 0; i < X.length; i++) {
      const xi = X[i], yi = y[i];
      const val = yi * (w1 * xi[0] + w2 * xi[1] + b);
      if (val <= 1.05) {
        svs.push(i);
      }
    }
    setSupportVectors(svs);
  }, [points, sandboxMode, svmC]);

  // 6. Decision Tree Recursive Partitions
  useEffect(() => {
    if (sandboxMode !== 'tree' || points.length < 2) return;

    const computeGini = (subset) => {
      if (subset.length === 0) return 0;
      const class0 = subset.filter(p => p.label === 0).length;
      const p0 = class0 / subset.length;
      const p1 = 1 - p0;
      return 1 - (p0 * p0 + p1 * p1);
    };

    const findBestSplit = (subset, minX, maxX, minY, maxY) => {
      let bestGiniDecrease = -1;
      let bestAxis = null;
      let bestVal = 0;
      let bestLeft = [];
      let bestRight = [];

      // Test horizontal (x) and vertical (y) thresholds along point coordinates
      subset.forEach(p => {
        ['x', 'y'].forEach(axis => {
          const val = p[axis];
          const left = subset.filter(pt => pt[axis] < val);
          const right = subset.filter(pt => pt[axis] >= val);

          if (left.length === 0 || right.length === 0) return;

          const currentGini = computeGini(subset);
          const leftGini = computeGini(left);
          const rightGini = computeGini(right);

          const gain = currentGini - (left.length / subset.length * leftGini + right.length / subset.length * rightGini);

          if (gain > bestGiniDecrease) {
            bestGiniDecrease = gain;
            bestAxis = axis;
            bestVal = val;
            bestLeft = left;
            bestRight = right;
          }
        });
      });

      return { bestGiniDecrease, bestAxis, bestVal, bestLeft, bestRight };
    };

    const partitionsList = [];
    const buildTree = (subset, depth, minX, maxX, minY, maxY) => {
      const class0 = subset.filter(p => p.label === 0).length;
      const class1 = subset.length - class0;

      // Stop condition: max depth reached, or pure node
      if (depth >= treeMaxDepth || class0 === 0 || class1 === 0 || subset.length < 2) {
        return;
      }

      const { bestAxis, bestVal, bestLeft, bestRight } = findBestSplit(subset, minX, maxX, minY, maxY);

      if (!bestAxis) return;

      // Record partition line segment to draw
      partitionsList.push({
        axis: bestAxis,
        val: bestVal,
        minX, maxX, minY, maxY
      });

      if (bestAxis === 'x') {
        buildTree(bestLeft, depth + 1, minX, bestVal, minY, maxY);
        buildTree(bestRight, depth + 1, bestVal, maxX, minY, maxY);
      } else {
        buildTree(bestLeft, depth + 1, minX, maxX, minY, bestVal);
        buildTree(bestRight, depth + 1, minX, maxX, bestVal, maxY);
      }
    };

    buildTree(points, 0, 0, 500, 0, 400);
    setTreePartitions(partitionsList);
  }, [points, sandboxMode, treeMaxDepth]);

  // Cleanup optimization interval on unmount
  useEffect(() => {
    return () => {
      if (optimizationInterval.current) {
        clearInterval(optimizationInterval.current);
      }
    };
  }, []);

  const getLossVal = (x, y, func) => {
    if (func === 'bowl') return 1.5 * x * x + 10 * y * y;
    if (func === 'rosenbrock') return Math.pow(1 - x, 2) + 5 * Math.pow(y - x * x, 2);
    if (func === 'saddle') return 1.5 * x * x - 1.5 * y * y;
    return 0;
  };

  const handleResetOptimization = () => {
    setIsOptimizing(false);
    if (optimizationInterval.current) {
      clearInterval(optimizationInterval.current);
    }
    setOptPath([{ x: optStart.x, y: optStart.y, vx: 0, vy: 0, m_x: 0, m_y: 0, v_x: 0, v_y: 0, step: 0, loss: getLossVal(optStart.x, optStart.y, optFunction) }]);
  };

  const handleStartOptimization = () => {
    if (isOptimizing) return;
    setIsOptimizing(true);
    
    if (optimizationInterval.current) {
      clearInterval(optimizationInterval.current);
    }

    let path = [...optPath];
    if (path.length === 0 || (path.length === 1 && path[0].x === optStart.x && path[0].y === optStart.y)) {
      path = [{ x: optStart.x, y: optStart.y, vx: 0, vy: 0, m_x: 0, m_y: 0, v_x: 0, v_y: 0, step: 0, loss: getLossVal(optStart.x, optStart.y, optFunction) }];
    }

    optimizationInterval.current = setInterval(() => {
      setOptPath(prevPath => {
        const last = prevPath[prevPath.length - 1] || { x: optStart.x, y: optStart.y, vx: 0, vy: 0, m_x: 0, m_y: 0, v_x: 0, v_y: 0, step: 0, loss: getLossVal(optStart.x, optStart.y, optFunction) };
        
        // Stop if we got very close to center (0, 0) in bowl or saddle, or close to (1, 1) in rosenbrock
        const targetX = optFunction === 'rosenbrock' ? 1.0 : 0.0;
        const targetY = optFunction === 'rosenbrock' ? 1.0 : 0.0;
        const tolerance = optFunction === 'rosenbrock' ? 0.05 : 0.03;
        if (Math.abs(last.x - targetX) < tolerance && Math.abs(last.y - targetY) < tolerance) {
          setIsOptimizing(false);
          clearInterval(optimizationInterval.current);
          return prevPath;
        }

        // Stop if path too long
        if (prevPath.length >= 1000) {
          setIsOptimizing(false);
          clearInterval(optimizationInterval.current);
          return prevPath;
        }

        // Gradients calculation based on function
        let gx = 0;
        let gy = 0;
        let loss = 0;

        if (optFunction === 'bowl') {
          gx = 3 * last.x;
          gy = 20 * last.y;
          loss = 1.5 * last.x * last.x + 10 * last.y * last.y;
        } else if (optFunction === 'rosenbrock') {
          gx = -2 * (1 - last.x) - 20 * last.x * (last.y - last.x * last.x);
          gy = 10 * (last.y - last.x * last.x);
          loss = Math.pow(1 - last.x, 2) + 5 * Math.pow(last.y - last.x * last.x, 2);
        } else if (optFunction === 'saddle') {
          gx = 3 * last.x;
          gy = -3 * last.y;
          loss = 1.5 * last.x * last.x - 1.5 * last.y * last.y;
        }

        let nextX = last.x;
        let nextY = last.y;
        let nextVx = last.vx || 0;
        let nextVy = last.vy || 0;
        let nextMx = last.m_x || 0;
        let nextMy = last.m_y || 0;
        let nextV_x = last.v_x || 0;
        let nextV_y = last.v_y || 0;
        const step = (last.step || 0) + 1;

        if (optMethod === 'sgd') {
          nextX = last.x - optLr * gx;
          nextY = last.y - optLr * gy;
        } else if (optMethod === 'momentum') {
          const beta = 0.9;
          nextVx = beta * nextVx + optLr * gx;
          nextVy = beta * nextVy + optLr * gy;
          nextX = last.x - nextVx;
          nextY = last.y - nextVy;
        } else if (optMethod === 'adam') {
          const beta1 = 0.9;
          const beta2 = 0.999;
          const epsilon = 1e-8;

          nextMx = beta1 * nextMx + (1 - beta1) * gx;
          nextMy = beta1 * nextMy + (1 - beta1) * gy;

          nextV_x = beta2 * nextV_x + (1 - beta2) * gx * gx;
          nextV_y = beta2 * nextV_y + (1 - beta2) * gy * gy;

          const mHatX = nextMx / (1 - Math.pow(beta1, step));
          const mHatY = nextMy / (1 - Math.pow(beta1, step));

          const vHatX = nextV_x / (1 - Math.pow(beta2, step));
          const vHatY = nextV_y / (1 - Math.pow(beta2, step));

          nextX = last.x - (optLr / (Math.sqrt(vHatX) + epsilon)) * mHatX;
          nextY = last.y - (optLr / (Math.sqrt(vHatY) + epsilon)) * mHatY;
        }

        // Clip to drawing area
        nextX = Math.max(-3, Math.min(3, nextX));
        nextY = Math.max(-3, Math.min(3, nextY));

        return [...prevPath, {
          x: nextX,
          y: nextY,
          vx: nextVx,
          vy: nextVy,
          m_x: nextMx,
          m_y: nextMy,
          v_x: nextV_x,
          v_y: nextV_y,
          step: step,
          loss: loss
        }];
      });
    }, 50);
  };

  // --- NN BUILDER METHODS ---
  const addLayer = (type) => {
    const newId = (layers.length + 1).toString();
    let newLayer = { id: newId, type };
    if (type === 'conv2d') {
      newLayer = { ...newLayer, filters: 32, kernelSize: 3, stride: 1, padding: 'same' };
    } else if (type === 'maxpool2d') {
      newLayer = { ...newLayer, poolSize: 2, stride: 2 };
    } else if (type === 'dense') {
      newLayer = { ...newLayer, units: 32, activation: 'relu' };
    } else if (type === 'dropout') {
      newLayer = { ...newLayer, rate: 0.25 };
    } else if (type === 'activation') {
      newLayer = { ...newLayer, activation: 'relu' };
    }
    setLayers(prev => [...prev, newLayer]);
  };

  const removeLayer = (id) => {
    if (layers.length <= 1) return; // Must keep input layer
    setLayers(prev => prev.filter(layer => layer.id !== id));
  };

  const updateLayer = (id, fields) => {
    setLayers(prev => prev.map(layer => layer.id === id ? { ...layer, ...fields } : layer));
  };

  const computeLayerShapes = () => {
    const computed = [];
    let currentShape = null;

    layers.forEach((layer) => {
      let outputShape = '';
      let parameters = 0;
      let details = '';

      if (layer.type === 'input') {
        currentShape = { w: layer.width || 28, h: layer.height || 28, c: layer.channels || 1, flat: null };
        outputShape = `(${currentShape.w}, ${currentShape.h}, ${currentShape.c})`;
        details = 'Input Tensor';
      } else if (currentShape) {
        if (layer.type === 'conv2d') {
          const k = layer.kernelSize || 3;
          const s = layer.stride || 1;
          const p = layer.padding || 'same';
          const f = layer.filters || 32;

          let outW, outH;
          if (p === 'same') {
            outW = currentShape.w;
            outH = currentShape.h;
          } else {
            outW = Math.floor((currentShape.w - k) / s) + 1;
            outH = Math.floor((currentShape.h - k) / s) + 1;
          }
          parameters = (k * k * currentShape.c + 1) * f;
          currentShape = { w: outW, h: outH, c: f, flat: null };
          outputShape = `(${outW}, ${outH}, ${f})`;
          details = `Kernel: ${k}x${k}, Stride: ${s}, Filters: ${f}`;
        } else if (layer.type === 'maxpool2d') {
          const p = layer.poolSize || 2;
          const s = layer.stride || 2;
          const outW = Math.floor(currentShape.w / p);
          const outH = Math.floor(currentShape.h / p);

          currentShape = { ...currentShape, w: outW, h: outH };
          outputShape = `(${outW}, ${outH}, ${currentShape.c})`;
          details = `Pool: ${p}x${p}, Stride: ${s}`;
        } else if (layer.type === 'flatten') {
          const size = currentShape.w * currentShape.h * currentShape.c;
          currentShape = { w: 0, h: 0, c: 0, flat: size };
          outputShape = `(${size})`;
          details = 'Flatten to 1D';
        } else if (layer.type === 'dense') {
          const inSize = currentShape.flat || (currentShape.w * currentShape.h * currentShape.c);
          const u = layer.units || 64;
          parameters = (inSize + 1) * u;
          currentShape = { w: 0, h: 0, c: 0, flat: u };
          outputShape = `(${u})`;
          details = `Units: ${u}, Activation: ${layer.activation}`;
        } else if (layer.type === 'dropout') {
          outputShape = currentShape.flat ? `(${currentShape.flat})` : `(${currentShape.w}, ${currentShape.h}, ${currentShape.c})`;
          details = `Rate: ${layer.rate || 0.25}`;
        } else if (layer.type === 'activation') {
          outputShape = currentShape.flat ? `(${currentShape.flat})` : `(${currentShape.w}, ${currentShape.h}, ${currentShape.c})`;
          details = `Activation: ${layer.activation}`;
        }
      }

      computed.push({
        ...layer,
        outputShape,
        parameters,
        details
      });
    });

    return computed;
  };

  const generateCode = () => {
    if (exportFramework === 'pytorch') {
      let pyLayers = [];
      let inChannels = 1;

      const shapes = computeLayerShapes();
      
      shapes.forEach((layer, idx) => {
        if (layer.type === 'input') {
          inChannels = layer.channels || 1;
        } else if (layer.type === 'conv2d') {
          pyLayers.push(`            nn.Conv2d(in_channels=${inChannels}, out_channels=${layer.filters}, kernel_size=${layer.kernelSize}, padding=${layer.padding === 'same' ? layer.kernelSize >> 1 : 0}),`);
          pyLayers.push(`            nn.ReLU(),`);
          inChannels = layer.filters;
        } else if (layer.type === 'maxpool2d') {
          pyLayers.push(`            nn.MaxPool2d(kernel_size=${layer.poolSize}, stride=${layer.stride}),`);
        } else if (layer.type === 'flatten') {
          pyLayers.push(`            nn.Flatten(),`);
        } else if (layer.type === 'dense') {
          const prevLayer = shapes[idx - 1];
          let size = 1;
          if (prevLayer) {
            const sh = prevLayer.outputShape.replace(/[()]/g, '').split(',').map(Number);
            size = sh.reduce((a, b) => a * b, 1);
          }
          pyLayers.push(`            nn.Linear(in_features=${size}, out_features=${layer.units}),`);
          if (layer.activation === 'relu') pyLayers.push(`            nn.ReLU(),`);
          else if (layer.activation === 'sigmoid') pyLayers.push(`            nn.Sigmoid(),`);
          else if (layer.activation === 'softmax') pyLayers.push(`            nn.Softmax(dim=1),`);
        } else if (layer.type === 'dropout') {
          pyLayers.push(`            nn.Dropout(p=${layer.rate}),`);
        }
      });

      return `import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
${pyLayers.join('\n')}
        )

    def forward(self, x):
        return self.model(x)

# Instantiate the model
model = CustomModel()
print(model)
`;
    } else if (exportFramework === 'keras') {
      let kerasLayers = [];
      let inputShape = '';

      const shapes = computeLayerShapes();
      shapes.forEach((layer) => {
        if (layer.type === 'input') {
          inputShape = `(${layer.width}, ${layer.height}, ${layer.channels})`;
        } else if (layer.type === 'conv2d') {
          kerasLayers.push(`    layers.Conv2D(${layer.filters}, kernel_size=${layer.kernelSize}, padding='${layer.padding}', activation='relu'),`);
        } else if (layer.type === 'maxpool2d') {
          kerasLayers.push(`    layers.MaxPooling2D(pool_size=${layer.poolSize}, strides=${layer.stride}),`);
        } else if (layer.type === 'flatten') {
          kerasLayers.push(`    layers.Flatten(),`);
        } else if (layer.type === 'dense') {
          kerasLayers.push(`    layers.Dense(${layer.units}, activation='${layer.activation}'),`);
        } else if (layer.type === 'dropout') {
          kerasLayers.push(`    layers.Dropout(rate=${layer.rate}),`);
        }
      });

      return `import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=${inputShape}),
${kerasLayers.join('\n')}
])

model.summary()
`;
    } else if (exportFramework === 'jax') {
      let jaxLayers = [];
      const shapes = computeLayerShapes();
      shapes.forEach((layer) => {
        if (layer.type === 'conv2d') {
          jaxLayers.push(`            nn.Conv(features=${layer.filters}, kernel_size=(${layer.kernelSize}, ${layer.kernelSize}), padding='${layer.padding.toUpperCase()}'),`);
          jaxLayers.push(`            nn.relu,`);
        } else if (layer.type === 'maxpool2d') {
          jaxLayers.push(`            lambda x: nn.max_pool(x, window_shape=(${layer.poolSize}, ${layer.poolSize}), strides=(${layer.stride}, ${layer.stride})),`);
        } else if (layer.type === 'flatten') {
          jaxLayers.push(`            lambda x: x.reshape((x.shape[0], -1)), # Flatten`);
        } else if (layer.type === 'dense') {
          jaxLayers.push(`            nn.Dense(features=${layer.units}),`);
          if (layer.activation === 'relu') jaxLayers.push(`            nn.relu,`);
          else if (layer.activation === 'sigmoid') jaxLayers.push(`            nn.sigmoid,`);
          else if (layer.activation === 'softmax') jaxLayers.push(`            nn.softmax,`);
        } else if (layer.type === 'dropout') {
          jaxLayers.push(`            nn.Dropout(rate=${layer.rate}),`);
        }
      });

      return `import flax.linen as nn
import jax.numpy as jnp

class CustomModel(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool = True):
        # Sequential structure
        # Input shape: x.shape
        x = nn.Sequential([
${jaxLayers.join('\n')}
        ])(x)
        return x
`;
    }
    return '';
  };

  // --- PYTHON SCRATCHPAD METHODS ---
  const handleRunScratchpad = async () => {
    setIsScratchpadRunning(true);
    setScratchpadOutput('');
    try {
      let codeToRun = scratchpadCode;
      if (scratchpadMode === 'guided') {
        const challenge = challenges[currentChallengeIdx];
        codeToRun = `${scratchpadCode}\n\n${challenge.testSuite}`;
      }
      const res = await runPython(codeToRun);
      setScratchpadOutput(res || "Code executed successfully with no output.");
    } catch (err) {
      setScratchpadOutput(err.toString());
    }
    setIsScratchpadRunning(false);
  };

  useEffect(() => {
    if (scratchpadMode === 'guided') {
      setScratchpadCode(challenges[currentChallengeIdx].starterCode);
      setScratchpadOutput('');
    }
  }, [scratchpadMode, currentChallengeIdx]);

  const loadScratchpadTemplate = (type) => {
    if (type === 'numpy') {
      setScratchpadCode(`import numpy as np

# Create matrix A and vector x
A = np.array([[2, 1], [1, 2]])
x = np.array([3, 1])

print("Matrix A:\\n", A)
print("Vector x:", x)

# Compute matrix-vector product
b = A.dot(x)
print("A * x =", b)

# Calculate eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\\n", eigenvectors)`);
    } else if (type === 'matplotlib') {
      setScratchpadCode(`import numpy as np
import matplotlib.pyplot as plt

# Generate coordinates for a gradient field
x = np.linspace(-2.0, 2.0, 100)
y = np.linspace(-2.0, 2.0, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2  # Bowl function

plt.contour(X, Y, Z, levels=15, cmap='viridis')
plt.plot(0, 0, 'ro', label='Minimum')
plt.title('2D Contour Bowl Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()`);
    } else if (type === 'regression') {
      setScratchpadCode(`import numpy as np

# Simple batch gradient descent for y = wx + b
X = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 3.8, 6.2, 8.1, 9.9])

w, b = 0.0, 0.0
lr = 0.01
epochs = 100

for epoch in range(epochs):
    # Predictions
    pred = w * X + b
    # Gradients
    dw = -2 * np.mean(X * (y - pred))
    db = -2 * np.mean(y - pred)
    # Update
    w -= lr * dw
    b -= lr * db

print(f"Trained weight (w): {w:.4f}")
print(f"Trained bias (b): {b:.4f}")
print(f"Final predictions: {w*X + b}")`);
    }
  };

  // SVG Click Handler for all modes
  const handleSvgClick = (e) => {
    if (sandboxMode === 'optimizer') {
      const rect = e.currentTarget.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;
      const mathX = (clickX - 250) / 80;
      const mathY = (200 - clickY) / 60;
      setOptStart({ x: mathX, y: mathY });
      setOptPath([{ x: mathX, y: mathY, vx: 0, vy: 0, m_x: 0, m_y: 0, v_x: 0, v_y: 0, step: 0, loss: getLossVal(mathX, mathY, optFunction) }]);
      setIsOptimizing(false);
      return;
    }

    if (sandboxMode === 'kmeans') {
      // Just add raw unlabeled points, centroids managed separately
      const rect = e.currentTarget.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;
      setPoints(prev => [...prev, { x: clickX, y: clickY, label: -1 }]);
      setKmeansStep(1); // Reset step state to recalculate
      return;
    }

    if (draggingIdx !== null) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    setPoints(prev => [...prev, { x: clickX, y: clickY, label: activeLabel }]);
  };

  const startDrag = (idx, e) => {
    e.stopPropagation();
    setDraggingIdx(idx);
  };

  const handleDrag = (e) => {
    if (draggingIdx === null) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const dragX = Math.max(0, Math.min(500, e.clientX - rect.left));
    const dragY = Math.max(0, Math.min(400, e.clientY - rect.top));

    setPoints(prev => {
      const copy = [...prev];
      copy[draggingIdx] = { ...copy[draggingIdx], x: dragX, y: dragY };
      return copy;
    });
  };

  const endDrag = () => {
    setDraggingIdx(null);
  };

  // Helper coordinate getters
  const getLineCoordinates = () => {
    const { slope, intercept } = linearLine;
    const y1 = intercept;
    const y2 = slope * 1 + intercept;
    return { x1: 0, y1: 400 - y1 * 400, x2: 500, y2: 400 - y2 * 400 };
  };

  const getLogisticBoundaryCoordinates = () => {
    const { w1, w2, b } = logisticBoundary;
    if (Math.abs(w2) < 1e-6) {
      const x = -b / w1;
      return { x1: x * 500, y1: 0, x2: x * 500, y2: 400 };
    }
    const y = (x) => -(w1 * x + b) / w2;
    return { x1: 0, y1: 400 - y(0) * 400, x2: 500, y2: 400 - y(1) * 400 };
  };

  const getSvmBoundaryCoordinates = () => {
    const { w1, w2, b } = svmBoundary;
    // Boundary line yi * f(xi) = 0 => w1*(x-0.5) + w2*(y-0.5) + b = 0
    if (Math.abs(w2) < 1e-6) {
      const x = -b / w1 + 0.5;
      return {
        x1: x * 500, y1: 0, x2: x * 500, y2: 400,
        margin1_x1: (x - 1/w1)*500, margin1_y1: 0, margin1_x2: (x - 1/w1)*500, margin1_y2: 400,
        margin2_x1: (x + 1/w1)*500, margin2_y1: 0, margin2_x2: (x + 1/w1)*500, margin2_y2: 400,
      };
    }
    
    // f(x, y) = w1*(x-0.5) + w2*(y-0.5) + b = 0 => y_centered = -(w1*x_centered + b)/w2
    const yVal = (xVal, offset = 0) => -(w1 * (xVal - 0.5) + b - offset) / w2 + 0.5;

    return {
      x1: 0, y1: 400 - yVal(0) * 400, x2: 500, y2: 400 - yVal(1) * 400,
      // margin lines corresponding to f(x, y) = -1 and +1
      margin1_x1: 0, margin1_y1: 400 - yVal(0, -1) * 400, margin1_x2: 500, margin1_y2: 400 - yVal(1, -1) * 400,
      margin2_x1: 0, margin2_y1: 400 - yVal(0, 1) * 400, margin2_x2: 500, margin2_y2: 400 - yVal(1, 1) * 400,
    };
  };

  // Render Contour lines for Optimizer visualizer
  const renderContours = () => {
    const paths = [];
    if (optFunction === 'bowl') {
      for (let cost = 1; cost <= 12; cost += 1.5) {
        const rx = Math.sqrt(cost / 1.5) * 80;
        const ry = Math.sqrt(cost / 10) * 60;
        paths.push(
          <ellipse 
            key={cost} 
            cx="250" 
            cy="200" 
            rx={rx} 
            ry={ry} 
            fill="none" 
            stroke="var(--accent-faint)" 
            strokeWidth="1.5" 
          />
        );
      }
    } else if (optFunction === 'saddle') {
      for (let c = -4; c <= 4; c += 1.5) {
        if (c === 0) continue;
        if (c > 0) {
          let ptsLeft = [];
          let ptsRight = [];
          for (let y = -3; y <= 3; y += 0.2) {
            const xVal = Math.sqrt(y * y + c);
            const cxRight = 250 + xVal * 80;
            const cy = 200 - y * 60;
            const cxLeft = 250 - xVal * 80;
            ptsRight.push(`${cxRight},${cy}`);
            ptsLeft.push(`${cxLeft},${cy}`);
          }
          paths.push(
            <polyline key={`sad-r-${c}`} points={ptsRight.join(' ')} fill="none" stroke="var(--accent-faint)" strokeWidth="1.5" />,
            <polyline key={`sad-l-${c}`} points={ptsLeft.join(' ')} fill="none" stroke="var(--accent-faint)" strokeWidth="1.5" />
          );
        } else {
          let ptsTop = [];
          let ptsBottom = [];
          for (let x = -3; x <= 3; x += 0.2) {
            const yVal = Math.sqrt(x * x - c);
            const cx = 250 + x * 80;
            const cyTop = 200 - yVal * 60;
            const cyBottom = 200 + yVal * 60;
            ptsTop.push(`${cx},${cyTop}`);
            ptsBottom.push(`${cx},${cyBottom}`);
          }
          paths.push(
            <polyline key={`sad-t-${c}`} points={ptsTop.join(' ')} fill="none" stroke="var(--accent-faint)" strokeWidth="1.5" />,
            <polyline key={`sad-b-${c}`} points={ptsBottom.join(' ')} fill="none" stroke="var(--accent-faint)" strokeWidth="1.5" />
          );
        }
      }
    } else if (optFunction === 'rosenbrock') {
      for (let offset = -2.2; offset <= 2.2; offset += 0.6) {
        let pts = [];
        for (let x = -2.2; x <= 2.2; x += 0.1) {
          const y = x * x + offset;
          const cx = 250 + x * 80;
          const cy = 200 - y * 60;
          if (cy >= 0 && cy <= 400 && cx >= 0 && cx <= 500) {
            pts.push(`${cx},${cy}`);
          }
        }
        if (pts.length > 1) {
          paths.push(
            <polyline key={`ros-${offset}`} points={pts.join(' ')} fill="none" stroke="var(--accent-faint)" strokeWidth="1.5" />
          );
        }
      }
    }
    return paths;
  };

  const renderLossChart = () => {
    if (optPath.length <= 1) {
      return (
        <div style={{ height: '100px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', border: '1px solid var(--glass-border)', marginTop: '16px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontStyle: 'italic' }}>Animate to plot loss convergence</span>
        </div>
      );
    }

    const margin = { top: 10, right: 10, bottom: 20, left: 45 };
    const width = 280;
    const height = 110;
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const losses = optPath.map(p => p.loss || 0);
    const maxLoss = Math.max(...losses, 1e-5);
    const minLoss = Math.min(...losses, 0);

    const points = optPath.map((p, idx) => {
      const x = margin.left + (idx / (optPath.length - 1)) * chartWidth;
      const range = maxLoss - minLoss;
      const normalizedLoss = range > 0 ? (p.loss - minLoss) / range : 0.5;
      const y = margin.top + chartHeight - normalizedLoss * chartHeight;
      return `${x},${y}`;
    }).join(' ');

    return (
      <div style={{ marginTop: '16px' }}>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
          <span>Loss Convergence Curve</span>
          <span style={{ color: 'var(--accent-color)', fontWeight: 'bold' }}>L: {losses[losses.length - 1].toFixed(4)}</span>
        </div>
        <svg width="100%" height={height} style={{ background: 'rgba(0,0,0,0.25)', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
          <polyline
            fill="none"
            stroke="var(--accent-color)"
            strokeWidth="2"
            points={points}
          />
          <text x={margin.left} y={height - 4} fill="var(--text-secondary)" fontSize="8">Step 0</text>
          <text x={width - 45} y={height - 4} fill="var(--text-secondary)" fontSize="8">Step {optPath.length - 1}</text>
          <text x={4} y={margin.top + 8} fill="var(--text-secondary)" fontSize="8">{maxLoss.toFixed(1)}</text>
          <text x={4} y={height - margin.bottom} fill="var(--text-secondary)" fontSize="8">{minLoss.toFixed(1)}</text>
        </svg>
      </div>
    );
  };

  // Clear Sandbox dataset helper
  const handleClearSandbox = () => {
    setPoints([]);
    setCentroids([]);
    setOptPath([]);
    setNnGrid([]);
    setTreePartitions([]);
    setIsOptimizing(false);
  };

  // ----------------------------------------------------
  // --- MATH SOLVER DERIVATIONS ---
  // ----------------------------------------------------
  useEffect(() => {
    const { a, b, c, d } = matrix;
    const { x1, x2 } = vector;
    const { b1, b2 } = projVector;

    const det = a * d - b * c;
    const trace = a + d;

    const discriminant = trace * trace - 4 * det;
    let lambdaMath = '';
    if (discriminant > 0) {
      const l1 = (trace + Math.sqrt(discriminant)) / 2;
      const l2 = (trace - Math.sqrt(discriminant)) / 2;
      lambdaMath = `\\lambda_1 = ${l1.toFixed(2)}, \\quad \\lambda_2 = ${l2.toFixed(2)}`;
    } else if (discriminant === 0) {
      const l = trace / 2;
      lambdaMath = `\\lambda_1 = \\lambda_2 = ${l.toFixed(2)}`;
    } else {
      const real = trace / 2;
      const imag = Math.sqrt(-discriminant) / 2;
      lambdaMath = `\\lambda_{1,2} = ${real.toFixed(2)} \\pm ${imag.toFixed(2)}i \\quad \\text{(Complex Eigenvalues)}`;
    }

    const isInversePossible = Math.abs(det) > 1e-6;
    const invA = isInversePossible ? d / det : 0;
    const invB = isInversePossible ? -b / det : 0;
    const invC = isInversePossible ? -c / det : 0;
    const invD = isInversePossible ? a / det : 0;

    const normXSquared = x1 * x1 + x2 * x2;
    const dotProduct = b1 * x1 + b2 * x2;
    const projCoeff = normXSquared > 0 ? dotProduct / normXSquared : 0;
    const projX1 = projCoeff * x1;
    const projX2 = projCoeff * x2;

    const md = `
### Step-by-Step Derivations

---

#### 1. Basic Matrix Metrics
We define our square matrix $A$ as:
$$ A = \\begin{bmatrix} ${a} & ${b} \\\\ ${c} & ${d} \\end{bmatrix} $$

* **Trace (Sum of diagonal elements):**
  $$ \\text{Tr}(A) = a + d = ${a} + ${d} = ${trace} $$

* **Determinant:**
  $$ \\det(A) = ad - bc = (${a} \\times ${d}) - (${b} \\times ${c}) = ${a*d} - ${b*c} = ${det} $$

---

#### 2. Matrix Inverse
The formula for the inverse of a $2 \\times 2$ matrix is:
$$ A^{-1} = \\frac{1}{\\det(A)} \\begin{bmatrix} d & -b \\\\ -c & a \\end{bmatrix} $$

${isInversePossible ? `
Plugging in our values:
$$ A^{-1} = \\frac{1}{${det}} \\begin{bmatrix} ${d} & ${-b} \\\\ ${-c} & ${a} \\end{bmatrix} = \\begin{bmatrix} ${invA.toFixed(2)} & ${invB.toFixed(2)} \\\\ ${invC.toFixed(2)} & ${invD.toFixed(2)} \\end{bmatrix}
$$
` : `
> [!WARNING]
> Since $\\det(A) = 0$, this matrix is singular/non-invertible. It collapses dimensions, meaning its columns are linearly dependent.
`}

---

#### 3. Characteristic Equation & Eigenvalues
To find the eigenvalues $\\lambda$, we solve the characteristic equation:
$$ \\det(A - \\lambda I) = 0 $$
$$ \\det\\begin{bmatrix} ${a} - \\lambda & ${b} \\\\ ${c} & ${d} - \\lambda \\end{bmatrix} = 0 $$
$$ (a - \\lambda)(d - \\lambda) - bc = 0 $$
$$ \\lambda^2 - \\text{Tr}(A)\\lambda + \\det(A) = 0 $$
$$ \\lambda^2 - ${trace}\\lambda + ${det} = 0 $$

Solving using the quadratic formula:
$$ \\lambda = \\frac{-(-${trace}) \\pm \\sqrt{(-${trace})^2 - 4(1)(${det})}}{2} $$
$$ \\lambda = \\frac{${trace} \\pm \\sqrt{${trace*trace} - ${4*det}}}{2} $$
$$ \\lambda = \\frac{${trace} \\pm \\sqrt{${discriminant}}}{2} $$

* **Eigenvalues:**
  $$ ${lambdaMath} $$

---

#### 4. Projection Matrix onto Subspace
Given a vector $\\mathbf{x} = \\begin{bmatrix} ${x1} \\\\ ${x2} \\end{bmatrix}$, the projection of vector $\\mathbf{b} = \\begin{bmatrix} ${b1} \\\\ ${b2} \\end{bmatrix}$ onto the span of $\\mathbf{x}$ is given by:

$$ P_{\\mathbf{x}}(\\mathbf{b}) = \\frac{\\mathbf{b} \\cdot \\mathbf{x}}{\\|\\mathbf{x}\\|_2^2} \\mathbf{x} $$

1. **Dot Product:** $\\mathbf{b} \\cdot \\mathbf{x} = (${b1} \\times ${x1}) + (${b2} \\times ${x2}) = ${dotProduct}$
2. **Norm Squared:** $\\|\\mathbf{x}\\|_2^2 = ${x1}^2 + ${x2}^2 = ${normXSquared}$
3. **Projection:**
   $$ P_{\\mathbf{x}}(\\mathbf{b}) = \\frac{${dotProduct}}{${normXSquared}} \\begin{bmatrix} ${x1} \\\\ ${x2} \\end{bmatrix} = \\begin{bmatrix} ${projX1.toFixed(2)} \\\\ ${projX2.toFixed(2)} \\end{bmatrix} $$
`;

    setSolverMarkdown(md);
  }, [matrix, vector, projVector]);

  // ----------------------------------------------------
  // --- TEXTBOOK BUILDER COMPILER ---
  // ----------------------------------------------------
  const handleCompileTextbook = async () => {
    setCompilerProgress('loading');
    try {
      const modules = import.meta.glob('../content/*.md', { query: '?raw', import: 'default' });
      let fullBook = `# Machine Learning Curriculum Textbook\n\n*Generated locally on ${new Date().toLocaleDateString()}*\n\n---\n\n`;

      for (let i = 0; i <= 41; i++) {
        const filePath = `../content/week${i}.md`;
        if (modules[filePath]) {
          const content = await modules[filePath]();
          fullBook += `\n\n<div class="pdf-page-break"></div>\n\n`;
          fullBook += content;
        }
      }
      setCompilerText(fullBook);
      setCompilerProgress('done');
    } catch (err) {
      console.error(err);
      setCompilerProgress(null);
    }
  };

  const handlePrintTextbook = () => {
    window.print();
  };

  return (
    <div className="interactive-lab-container text-white">
      <div style={{ marginBottom: '32px' }} className="no-print">
        <h1 style={{ fontSize: '2.5rem', margin: 0 }}>Interactive Lab</h1>
        <p style={{ color: 'var(--text-secondary)', marginTop: '8px', fontSize: '1.1rem' }}>
          Hands-on graphical tools, LaTeX calculators, and resource exports.
        </p>
      </div>

      <div className="tabs-container no-print">
        <button 
          className={`tab-btn ${activeTab === 'sandbox' ? 'active' : ''}`}
          onClick={() => setActiveTab('sandbox')}
        >
          Boundary Sandbox
        </button>
        <button 
          className={`tab-btn ${activeTab === 'solver' ? 'active' : ''}`}
          onClick={() => setActiveTab('solver')}
        >
          Math Deriver Solver
        </button>
        <button 
          className={`tab-btn ${activeTab === 'nnbuilder' ? 'active' : ''}`}
          onClick={() => { setActiveTab('nnbuilder'); handleResetOptimization(); }}
        >
          Visual NN Builder
        </button>
        <button 
          className={`tab-btn ${activeTab === 'scratchpad' ? 'active' : ''}`}
          onClick={() => { setActiveTab('scratchpad'); handleResetOptimization(); }}
        >
          Python Scratchpad
        </button>
        <button 
          className={`tab-btn ${activeTab === 'compiler' ? 'active' : ''}`}
          onClick={() => { setActiveTab('compiler'); handleResetOptimization(); }}
        >
          Textbook Compiler
        </button>
        <button
          className={`tab-btn ${activeTab === '3dexplorer' ? 'active' : ''}`}
          onClick={() => { setActiveTab('3dexplorer'); handleResetOptimization(); }}
          style={activeTab === '3dexplorer' ? { background: 'linear-gradient(135deg, #a78bfa, #7c3aed)', boxShadow: '0 0 18px #7c3aed77' } : {}}
        >
          🌐 3D Explorer
        </button>
      </div>

      {/* ---------------------------------------------------- */}
      {/* --- TAB 1: SANDBOX --- */}
      {/* ---------------------------------------------------- */}
      {activeTab === 'sandbox' && (
        <div className="sandbox-panel glass-panel no-print" style={{ padding: '24px' }}>
          
          {/* Scrollable Modes row */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '24px', overflowX: 'auto', paddingBottom: '8px' }}>
            {['linear', 'logistic', 'svm', 'tree', 'neural', 'kmeans', 'optimizer'].map((mode) => (
              <button
                key={mode}
                className={`tab-btn ${sandboxMode === mode ? 'active' : ''}`}
                onClick={() => { setSandboxMode(mode); handleResetOptimization(); }}
                style={{ padding: '8px 16px', fontSize: '0.85rem', whiteSpace: 'nowrap' }}
              >
                {mode === 'linear' && 'Linear OLS'}
                {mode === 'logistic' && 'Logistic Regression'}
                {mode === 'svm' && 'SVM margins'}
                {mode === 'tree' && 'Decision Tree'}
                {mode === 'neural' && 'Neural Network'}
                {mode === 'kmeans' && 'K-Means Clustering'}
                {mode === 'optimizer' && 'Optimizer Trails'}
              </button>
            ))}
          </div>

          <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
            <div style={{ flex: 1, minWidth: '300px' }}>
              
              {/* Mode Specific Controls */}
              {sandboxMode === 'linear' && (
                <div>
                  <h3 style={{ marginBottom: '8px' }}>Linear Regression (OLS)</h3>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', marginBottom: '16px' }}>
                    Click on the canvas to place points. Drag points to recalculate the line of best fit instantly.
                  </p>
                </div>
              )}

              {sandboxMode === 'logistic' && (
                <div>
                  <h3 style={{ marginBottom: '8px' }}>Logistic Regression Classification</h3>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', marginBottom: '16px' }}>
                    Toggle red/blue labels, place points, and drag them to see the classification decision boundary adjust.
                  </p>
                  <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                    <button 
                      className="tab-btn" 
                      style={{ border: activeLabel === 0 ? '1px solid var(--accent-color)' : '1px solid var(--glass-border)', background: activeLabel === 0 ? 'rgba(239, 68, 68, 0.2)' : 'transparent', color: '#ff6b6b', padding: '6px 12px' }}
                      onClick={() => setActiveLabel(0)}
                    >
                      ● Add Class 0 (Red)
                    </button>
                    <button 
                      className="tab-btn" 
                      style={{ border: activeLabel === 1 ? '1px solid var(--accent-color)' : '1px solid var(--glass-border)', background: activeLabel === 1 ? 'rgba(59, 130, 246, 0.2)' : 'transparent', color: '#60a5fa', padding: '6px 12px' }}
                      onClick={() => setActiveLabel(1)}
                    >
                      ● Add Class 1 (Blue)
                    </button>
                  </div>
                </div>
              )}

              {sandboxMode === 'svm' && (
                <div>
                  <h3 style={{ marginBottom: '8px' }}>Support Vector Machines (Margins)</h3>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', marginBottom: '16px' }}>
                    The SVM finds the separating line that maximizes the margin width. Highlighted double-bordered circles represent support vectors.
                  </p>
                  
                  <div style={{ marginBottom: '20px' }}>
                    <label style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>C Parameter (Margin Softness): {svmC}</label>
                    <input 
                      type="range" 
                      min="0.1" 
                      max="10.0" 
                      step="0.1" 
                      value={svmC} 
                      onChange={(e) => setSvmC(parseFloat(e.target.value))}
                      style={{ width: '100%', accentColor: 'var(--accent-color)', marginTop: '8px' }}
                    />
                  </div>
                  
                  <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                    <button 
                      className="tab-btn" 
                      style={{ border: activeLabel === 0 ? '1px solid var(--accent-color)' : '1px solid var(--glass-border)', background: activeLabel === 0 ? 'rgba(239, 68, 68, 0.2)' : 'transparent', color: '#ff6b6b', padding: '6px 12px' }}
                      onClick={() => setActiveLabel(0)}
                    >
                      ● Red Class (-1)
                    </button>
                    <button 
                      className="tab-btn" 
                      style={{ border: activeLabel === 1 ? '1px solid var(--accent-color)' : '1px solid var(--glass-border)', background: activeLabel === 1 ? 'rgba(59, 130, 246, 0.2)' : 'transparent', color: '#60a5fa', padding: '6px 12px' }}
                      onClick={() => setActiveLabel(1)}
                    >
                      ● Blue Class (+1)
                    </button>
                  </div>
                </div>
              )}

              {sandboxMode === 'tree' && (
                <div>
                  <h3 style={{ marginBottom: '8px' }}>Decision Tree Partitions</h3>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', marginBottom: '16px' }}>
                    Visualizes split cuts computed recursively using Gini Impurity. Adjust the tree depth to see underfitting vs overfitting.
                  </p>
                  
                  <div style={{ marginBottom: '20px' }}>
                    <label style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Max Depth: {treeMaxDepth}</label>
                    <input 
                      type="range" 
                      min="1" 
                      max="5" 
                      step="1" 
                      value={treeMaxDepth} 
                      onChange={(e) => setTreeMaxDepth(parseInt(e.target.value))}
                      style={{ width: '100%', accentColor: 'var(--accent-color)', marginTop: '8px' }}
                    />
                  </div>

                  <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                    <button 
                      className="tab-btn" 
                      style={{ border: activeLabel === 0 ? '1px solid var(--accent-color)' : '1px solid var(--glass-border)', background: activeLabel === 0 ? 'rgba(239, 68, 68, 0.2)' : 'transparent', color: '#ff6b6b', padding: '6px 12px' }}
                      onClick={() => setActiveLabel(0)}
                    >
                      ● Red Label
                    </button>
                    <button 
                      className="tab-btn" 
                      style={{ border: activeLabel === 1 ? '1px solid var(--accent-color)' : '1px solid var(--glass-border)', background: activeLabel === 1 ? 'rgba(59, 130, 246, 0.2)' : 'transparent', color: '#60a5fa', padding: '6px 12px' }}
                      onClick={() => setActiveLabel(1)}
                    >
                      ● Blue Label
                    </button>
                  </div>
                </div>
              )}

              {sandboxMode === 'neural' && (
                <div>
                  <h3 style={{ marginBottom: '8px' }}>Neural Network Classifier</h3>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', marginBottom: '16px' }}>
                    A 2-layer MLP runs feedforward/backprop loops to warp space around points. Background colors represent decision probabilities.
                  </p>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '20px' }}>
                    <div>
                      <label style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Activation Function</label>
                      <select 
                        value={nnActivation} 
                        onChange={(e) => setNnActivation(e.target.value)}
                        style={{ width: '100%', padding: '8px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '6px', marginTop: '6px' }}
                      >
                        <option value="tanh">Hyperbolic Tangent (Tanh)</option>
                        <option value="sigmoid">Sigmoid</option>
                        <option value="relu">ReLU (Rectified Linear)</option>
                      </select>
                    </div>

                    <div>
                      <label style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Hidden Neurons: {nnHiddenSize}</label>
                      <input 
                        type="range" 
                        min="2" 
                        max="8" 
                        step="1" 
                        value={nnHiddenSize} 
                        onChange={(e) => setNnHiddenSize(parseInt(e.target.value))}
                        style={{ width: '100%', accentColor: 'var(--accent-color)', marginTop: '8px' }}
                      />
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                    <button 
                      className="tab-btn" 
                      style={{ border: activeLabel === 0 ? '1px solid var(--accent-color)' : '1px solid var(--glass-border)', background: activeLabel === 0 ? 'rgba(239, 68, 68, 0.2)' : 'transparent', color: '#ff6b6b', padding: '6px 12px' }}
                      onClick={() => setActiveLabel(0)}
                    >
                      ● Red Class
                    </button>
                    <button 
                      className="tab-btn" 
                      style={{ border: activeLabel === 1 ? '1px solid var(--accent-color)' : '1px solid var(--glass-border)', background: activeLabel === 1 ? 'rgba(59, 130, 246, 0.2)' : 'transparent', color: '#60a5fa', padding: '6px 12px' }}
                      onClick={() => setActiveLabel(1)}
                    >
                      ● Blue Class
                    </button>
                  </div>
                </div>
              )}

              {sandboxMode === 'kmeans' && (
                <div>
                  <h3 style={{ marginBottom: '8px' }}>K-Means Clustering Simulator</h3>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', marginBottom: '16px' }}>
                    Lloyd's algorithm clusters unlabeled points. Step to assign points or move centroids.
                  </p>

                  <div style={{ marginBottom: '16px' }}>
                    <label style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '8px' }}>
                      K Clusters: {kmeansK}
                    </label>
                    <input 
                      type="range" 
                      min="2" 
                      max="4" 
                      value={kmeansK} 
                      onChange={(e) => { setKmeansK(parseInt(e.target.value)); setCentroids([]); }}
                      style={{ width: '100%', accentColor: 'var(--accent-color)' }}
                    />
                  </div>

                  <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                    <button className="action-btn" onClick={initKmeans} style={{ padding: '8px 16px', fontSize: '0.9rem' }}>
                      Initialize Centroids
                    </button>
                    <button 
                      className="action-btn" 
                      onClick={stepKmeans} 
                      disabled={centroids.length === 0}
                      style={{ padding: '8px 16px', fontSize: '0.9rem', background: centroids.length === 0 ? 'rgba(255,255,255,0.05)' : 'var(--accent-color)' }}
                    >
                      {kmeansStep === 1 ? 'Assign Points' : 'Update Centroids'}
                    </button>
                  </div>
                </div>
              )}

              {sandboxMode === 'optimizer' && (
                <div>
                  <h3 style={{ marginBottom: '8px' }}>Optimizer Trajectories</h3>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', marginBottom: '16px' }}>
                    Watch SGD, Momentum, and Adam navigate ravines. Click canvas to set the starting position.
                  </p>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '24px' }}>
                    <div>
                      <label style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Objective Function</label>
                      <select 
                        value={optFunction}
                        onChange={(e) => {
                          const func = e.target.value;
                          setOptFunction(func);
                          setOptPath([{ x: optStart.x, y: optStart.y, vx: 0, vy: 0, m_x: 0, m_y: 0, v_x: 0, v_y: 0, step: 0, loss: getLossVal(optStart.x, optStart.y, func) }]);
                          setIsOptimizing(false);
                        }}
                        style={{ width: '100%', padding: '10px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '6px', marginTop: '6px' }}
                      >
                        <option value="bowl">Asymmetric Bowl: 1.5x² + 10y²</option>
                        <option value="rosenbrock">Rosenbrock Banana Function</option>
                        <option value="saddle">Saddle Point / Ravine: 1.5x² - 1.5y²</option>
                      </select>
                    </div>

                    <div>
                      <label style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Algorithm</label>
                      <select 
                        value={optMethod}
                        onChange={(e) => setOptMethod(e.target.value)}
                        style={{ width: '100%', padding: '10px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '6px', marginTop: '6px' }}
                      >
                        <option value="sgd">Vanilla SGD</option>
                        <option value="momentum">SGD + Momentum</option>
                        <option value="adam">Adam</option>
                      </select>
                    </div>

                    <div>
                      <label style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Learning Rate: {optLr}</label>
                      <input 
                        type="range" 
                        min="0.005" 
                        max="0.2" 
                        step="0.005" 
                        value={optLr} 
                        onChange={(e) => setOptLr(parseFloat(e.target.value))}
                        style={{ width: '100%', accentColor: 'var(--accent-color)', marginTop: '8px' }}
                      />
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '12px' }}>
                    <button className="action-btn" onClick={handleStartOptimization} disabled={isOptimizing}>
                      Animate Optimization
                    </button>
                    <button 
                      className="external-link-btn" 
                      onClick={handleResetOptimization}
                      style={{ background: 'transparent' }}
                    >
                      Reset
                    </button>
                  </div>

                  {renderLossChart()}
                </div>
              )}

              <button 
                className="external-link-btn" 
                onClick={handleClearSandbox}
                style={{ background: 'transparent', marginTop: '20px' }}
              >
                Clear Canvas / Path
              </button>
            </div>

            <div style={{ position: 'relative', background: '#0b0f19', border: '1px solid var(--glass-border)', borderRadius: '12px', overflow: 'hidden' }}>
              <svg 
                width="500" 
                height="400" 
                onClick={handleSvgClick}
                onMouseMove={handleDrag}
                onMouseUp={endDrag}
                onMouseLeave={endDrag}
                style={{ cursor: draggingIdx !== null ? 'grabbing' : 'crosshair', display: 'block' }}
              >
                {/* 1. Grid Background shading for Neural Network */}
                {sandboxMode === 'neural' && nnGrid.map((cell, idx) => (
                  <rect
                    key={idx}
                    x={cell.gx * 25}
                    y={cell.gy * 20}
                    width="25"
                    height="20"
                    fill={cell.pred > 0.5 ? 'rgba(59, 130, 246, 0.15)' : 'rgba(239, 68, 68, 0.15)'}
                  />
                ))}

                {/* Axes */}
                <line x1="250" y1="0" x2="250" y2="400" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
                <line x1="0" y1="200" x2="500" y2="200" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />

                {/* 2. Linear Regression OLS Line */}
                {sandboxMode === 'linear' && points.length >= 2 && (() => {
                  const { x1, y1, x2, y2 } = getLineCoordinates();
                  return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--accent-color)" strokeWidth="3" />;
                })()}

                {/* 3. Logistic Separator */}
                {sandboxMode === 'logistic' && points.length >= 2 && (() => {
                  const { x1, y1, x2, y2 } = getLogisticBoundaryCoordinates();
                  return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#10b981" strokeWidth="3" strokeDasharray="5,5" />;
                })()}

                {/* 4. SVM Separating line & margins */}
                {sandboxMode === 'svm' && points.length >= 2 && (() => {
                  const { x1, y1, x2, y2, margin1_x1, margin1_y1, margin1_x2, margin1_y2, margin2_x1, margin2_y1, margin2_x2, margin2_y2 } = getSvmBoundaryCoordinates();
                  return (
                    <>
                      {/* Main separator */}
                      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#10b981" strokeWidth="3" />
                      {/* Margin 1 */}
                      <line x1={margin1_x1} y1={margin1_y1} x2={margin1_x2} y2={margin1_y2} stroke="rgba(255,255,255,0.25)" strokeWidth="1.5" strokeDasharray="4,4" />
                      {/* Margin 2 */}
                      <line x1={margin2_x1} y1={margin2_y1} x2={margin2_x2} y2={margin2_y2} stroke="rgba(255,255,255,0.25)" strokeWidth="1.5" strokeDasharray="4,4" />
                    </>
                  );
                })()}

                {/* 5. Decision Tree partition cuts */}
                {sandboxMode === 'tree' && treePartitions.map((part, idx) => {
                  if (part.axis === 'x') {
                    return <line key={idx} x1={part.val} y1={part.minY} x2={part.val} y2={part.maxY} stroke="var(--accent-color)" strokeWidth="2" />;
                  } else {
                    return <line key={idx} x1={part.minX} y1={part.val} x2={part.maxX} y2={part.val} stroke="var(--accent-color)" strokeWidth="2" />;
                  }
                })}

                {/* 6. Optimizer path rendering */}
                {sandboxMode === 'optimizer' && (
                  <>
                    {renderContours()}
                    <circle cx="250" cy="200" r="6" fill="#10b981" />
                    {optPath.length > 1 && (
                      <polyline 
                        points={optPath.map(p => `${250 + p.x * 80},${200 - p.y * 60}`).join(' ')}
                        fill="none" 
                        stroke="var(--accent-color)" 
                        strokeWidth="2.5" 
                      />
                    )}
                    <circle cx={250 + optStart.x * 80} cy={200 - optStart.y * 60} r="7" fill="#8b5cf6" stroke="#fff" strokeWidth="1.5" />
                    {optPath.length > 0 && (
                      <circle cx={250 + optPath[optPath.length - 1].x * 80} cy={200 - optPath[optPath.length - 1].y * 60} r="6" fill="#ff007f" stroke="#fff" strokeWidth="1" />
                    )}
                  </>
                )}

                {/* 7. K-Means Centroids & Points */}
                {sandboxMode === 'kmeans' && (
                  <>
                    {/* Centroids drawn as big triangles */}
                    {centroids.map((c, kIdx) => (
                      <polygon
                        key={`c-${kIdx}`}
                        points={`${c.x},${c.y - 12} ${c.x - 10},${c.y + 8} ${c.x + 10},${c.y + 8}`}
                        fill={kIdx === 0 ? '#ef4444' : kIdx === 1 ? '#3b82f6' : kIdx === 2 ? '#10b981' : '#f59e0b'}
                        stroke="#fff"
                        strokeWidth="2.5"
                      />
                    ))}
                  </>
                )}

                {/* Raw Points rendering */}
                {sandboxMode !== 'optimizer' && points.map((p, idx) => {
                  let color = '#ff007f'; // default
                  let isSV = false;
                  
                  if (sandboxMode === 'logistic' || sandboxMode === 'tree' || sandboxMode === 'neural') {
                    color = p.label === 0 ? '#ef4444' : '#3b82f6';
                  } else if (sandboxMode === 'svm') {
                    color = p.label === 0 ? '#ef4444' : '#3b82f6';
                    isSV = supportVectors.includes(idx);
                  } else if (sandboxMode === 'kmeans') {
                    const label = kmeansLabels[idx];
                    color = label === 0 ? '#ef4444' : label === 1 ? '#3b82f6' : label === 2 ? '#10b981' : label === 3 ? '#f59e0b' : '#9ca3af';
                  }

                  return (
                    <circle 
                      key={idx}
                      cx={p.x}
                      cy={p.y}
                      r={isSV ? "11" : "8"}
                      fill={color}
                      stroke="#fff"
                      strokeWidth={isSV ? "3" : "2"}
                      strokeDasharray={isSV ? "2,2" : "none"}
                      onMouseDown={(e) => startDrag(idx, e)}
                      style={{ cursor: 'grab' }}
                    />
                  );
                })}
              </svg>
            </div>
          </div>
        </div>
      )}

      {/* ---------------------------------------------------- */}
      {/* --- TAB 2: MATH SOLVER --- */}
      {/* ---------------------------------------------------- */}
      {activeTab === 'solver' && (
        <div className="solver-panel glass-panel no-print" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', gap: '32px', flexWrap: 'wrap', marginBottom: '32px' }}>
            <div style={{ flex: 1, minWidth: '240px' }}>
              <h3 style={{ marginBottom: '16px' }}>Input Parameters</h3>
              
              <div style={{ marginBottom: '24px' }}>
                <label style={{ display: 'block', fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                  2x2 Matrix A Elements
                </label>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', maxWidth: '200px' }}>
                  <input 
                    type="number" 
                    value={matrix.a} 
                    onChange={(e) => setMatrix(prev => ({ ...prev, a: parseFloat(e.target.value) || 0 }))}
                    placeholder="a"
                    style={{ padding: '10px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '6px' }}
                  />
                  <input 
                    type="number" 
                    value={matrix.b} 
                    onChange={(e) => setMatrix(prev => ({ ...prev, b: parseFloat(e.target.value) || 0 }))}
                    placeholder="b"
                    style={{ padding: '10px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '6px' }}
                  />
                  <input 
                    type="number" 
                    value={matrix.c} 
                    onChange={(e) => setMatrix(prev => ({ ...prev, c: parseFloat(e.target.value) || 0 }))}
                    placeholder="c"
                    style={{ padding: '10px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '6px' }}
                  />
                  <input 
                    type="number" 
                    value={matrix.d} 
                    onChange={(e) => setMatrix(prev => ({ ...prev, d: parseFloat(e.target.value) || 0 }))}
                    placeholder="d"
                    style={{ padding: '10px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '6px' }}
                  />
                </div>
              </div>

              <div style={{ display: 'flex', gap: '16px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                    Vector x
                  </label>
                  <div style={{ display: 'flex', gap: '8px', maxWidth: '140px' }}>
                    <input 
                      type="number" 
                      value={vector.x1} 
                      onChange={(e) => setVector(prev => ({ ...prev, x1: parseFloat(e.target.value) || 0 }))}
                      style={{ width: '60px', padding: '10px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '6px' }}
                    />
                    <input 
                      type="number" 
                      value={vector.x2} 
                      onChange={(e) => setVector(prev => ({ ...prev, x2: parseFloat(e.target.value) || 0 }))}
                      style={{ width: '60px', padding: '10px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '6px' }}
                    />
                  </div>
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                    Vector b
                  </label>
                  <div style={{ display: 'flex', gap: '8px', maxWidth: '140px' }}>
                    <input 
                      type="number" 
                      value={projVector.b1} 
                      onChange={(e) => setProjVector(prev => ({ ...prev, b1: parseFloat(e.target.value) || 0 }))}
                      style={{ width: '60px', padding: '10px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '6px' }}
                    />
                    <input 
                      type="number" 
                      value={projVector.b2} 
                      onChange={(e) => setProjVector(prev => ({ ...prev, b2: parseFloat(e.target.value) || 0 }))}
                      style={{ width: '60px', padding: '10px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '6px' }}
                    />
                  </div>
                </div>
              </div>

              {/* 2D Vector Grid Transformation Visualizer */}
              <div style={{ marginTop: '24px', textAlign: 'center' }}>
                <h4 style={{ fontSize: '0.95rem', marginBottom: '12px', textAlign: 'left', color: 'var(--accent-color)' }}>Visual Mapping Grid</h4>
                <div style={{ background: '#0b0f19', border: '1px solid var(--glass-border)', borderRadius: '8px', padding: '12px', display: 'inline-block' }}>
                  <svg width="300" height="300" style={{ display: 'block' }}>
                    {/* Background Grid Lines (Faint original) */}
                    {Array.from({ length: 11 }).map((_, i) => {
                      const val = (i - 5) * 25 + 150;
                      return (
                        <React.Fragment key={i}>
                          <line x1="0" y1={val} x2="300" y2={val} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
                          <line x1={val} y1="0" x2={val} y2="300" stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
                        </React.Fragment>
                      );
                    })}

                    {/* Transformed horizontal lines */}
                    {Array.from({ length: 11 }).map((_, i) => {
                      const yVal = i - 5;
                      const tx1 = matrix.a * -5 + matrix.b * yVal;
                      const ty1 = matrix.c * -5 + matrix.d * yVal;
                      const tx2 = matrix.a * 5 + matrix.b * yVal;
                      const ty2 = matrix.c * 5 + matrix.d * yVal;
                      return (
                        <line 
                          key={`th-${i}`} 
                          x1={150 + tx1 * 25} 
                          y1={150 - ty1 * 25} 
                          x2={150 + tx2 * 25} 
                          y2={150 - ty2 * 25} 
                          stroke="rgba(139, 92, 246, 0.2)" 
                          strokeWidth="1.5" 
                        />
                      );
                    })}

                    {/* Transformed vertical lines */}
                    {Array.from({ length: 11 }).map((_, i) => {
                      const xVal = i - 5;
                      const tx1 = matrix.a * xVal + matrix.b * -5;
                      const ty1 = matrix.c * xVal + matrix.d * -5;
                      const tx2 = matrix.a * xVal + matrix.b * 5;
                      const ty2 = matrix.c * xVal + matrix.d * 5;
                      return (
                        <line 
                          key={`tv-${i}`} 
                          x1={150 + tx1 * 25} 
                          y1={150 - ty1 * 25} 
                          x2={150 + tx2 * 25} 
                          y2={150 - ty2 * 25} 
                          stroke="rgba(139, 92, 246, 0.2)" 
                          strokeWidth="1.5" 
                        />
                      );
                    })}

                    {/* Main Axes */}
                    <line x1="150" y1="0" x2="150" y2="300" stroke="rgba(255,255,255,0.15)" strokeWidth="1.5" />
                    <line x1="0" y1="150" x2="300" y2="150" stroke="rgba(255,255,255,0.15)" strokeWidth="1.5" />

                    {/* Eigenvector lines (dashed) */}
                    {(() => {
                      const trace = matrix.a + matrix.d;
                      const det = matrix.a * matrix.d - matrix.b * matrix.c;
                      const disc = trace * trace - 4 * det;
                      if (disc >= 0) {
                        const l1 = (trace + Math.sqrt(disc)) / 2;
                        const l2 = (trace - Math.sqrt(disc)) / 2;
                        const findEv = (l) => {
                          if (Math.abs(matrix.b) > 1e-5) return { x: -matrix.b, y: matrix.a - l };
                          if (Math.abs(matrix.c) > 1e-5) return { x: matrix.d - l, y: -matrix.c };
                          return Math.abs(matrix.a - l) < 1e-5 ? { x: 1, y: 0 } : { x: 0, y: 1 };
                        };
                        const ev1 = findEv(l1);
                        const ev2 = findEv(l2);

                        // Normalize and draw infinite line approximation
                        const drawEvLine = (ev, color) => {
                          const len = Math.sqrt(ev.x * ev.x + ev.y * ev.y);
                          if (len < 1e-6) return null;
                          const dx = (ev.x / len) * 150;
                          const dy = (ev.y / len) * 150;
                          return (
                            <line 
                              x1={150 - dx} y1={150 + dy} x2={150 + dx} y2={150 - dy} 
                              stroke={color} strokeWidth="1.5" strokeDasharray="3,3" 
                            />
                          );
                        };

                        return (
                          <>
                            {drawEvLine(ev1, "#10b981")}
                            {drawEvLine(ev2, "#f59e0b")}
                          </>
                        );
                      }
                      return null;
                    })()}

                    {/* Vector x (Blue) */}
                    <line 
                      x1="150" 
                      y1="150" 
                      x2={150 + vector.x1 * 25} 
                      y2={150 - vector.x2 * 25} 
                      stroke="#3b82f6" 
                      strokeWidth="3" 
                      markerEnd="url(#arrow-blue)"
                    />
                    <circle cx={150 + vector.x1 * 25} cy={150 - vector.x2 * 25} r="4" fill="#3b82f6" />

                    {/* Transformed Vector Ax (Magenta) */}
                    {(() => {
                      const tax = matrix.a * vector.x1 + matrix.b * vector.x2;
                      const tay = matrix.c * vector.x1 + matrix.d * vector.x2;
                      return (
                        <>
                          <line 
                            x1="150" 
                            y1="150" 
                            x2={150 + tax * 25} 
                            y2={150 - tay * 25} 
                            stroke="#ff007f" 
                            strokeWidth="3" 
                          />
                          <circle cx={150 + tax * 25} cy={150 - tay * 25} r="4" fill="#ff007f" />
                        </>
                      );
                    })()}

                    {/* Vector b (Yellow) */}
                    <line 
                      x1="150" 
                      y1="150" 
                      x2={150 + projVector.b1 * 25} 
                      y2={150 - projVector.b2 * 25} 
                      stroke="#f59e0b" 
                      strokeWidth="2" 
                    />
                    <circle cx={150 + projVector.b1 * 25} cy={150 - projVector.b2 * 25} r="3" fill="#f59e0b" />

                    {/* Projection Vector P_x(b) (Green) */}
                    {(() => {
                      const normX = vector.x1 * vector.x1 + vector.x2 * vector.x2;
                      const dot = projVector.b1 * vector.x1 + projVector.b2 * vector.x2;
                      const coeff = normX > 0 ? dot / normX : 0;
                      const px = coeff * vector.x1;
                      const py = coeff * vector.x2;
                      return (
                        <>
                          {/* Projection path */}
                          <line 
                            x1={150 + px * 25} 
                            y1={150 - py * 25} 
                            x2={150 + projVector.b1 * 25} 
                            y2={150 - projVector.b2 * 25} 
                            stroke="rgba(255,255,255,0.4)" 
                            strokeWidth="1" 
                            strokeDasharray="2,2"
                          />
                          <line 
                            x1="150" 
                            y1="150" 
                            x2={150 + px * 25} 
                            y2={150 - py * 25} 
                            stroke="#10b981" 
                            strokeWidth="2.5" 
                          />
                          <circle cx={150 + px * 25} cy={150 - py * 25} r="3" fill="#10b981" />
                        </>
                      );
                    })()}
                  </svg>
                  <div style={{ display: 'flex', justifyContent: 'center', gap: '12px', fontSize: '0.75rem', marginTop: '8px', flexWrap: 'wrap' }}>
                    <span style={{ color: '#3b82f6' }}>● x</span>
                    <span style={{ color: '#ff007f' }}>● Ax</span>
                    <span style={{ color: '#f59e0b' }}>● b</span>
                    <span style={{ color: '#10b981' }}>● Proj(b)</span>
                    <span style={{ color: 'rgba(255,255,255,0.5)' }}>--- Eigenvectors</span>
                  </div>
                </div>
              </div>
            </div>

            <div style={{ flex: 2, minWidth: '300px', padding: '20px', background: 'rgba(255,255,255,0.01)', borderLeft: '1px solid var(--glass-border)' }}>
              <ReactMarkdown
                children={solverMarkdown}
                remarkPlugins={[remarkMath]}
                rehypePlugins={[rehypeKatex]}
              />
            </div>
          </div>
        </div>
      )}

      {/* ---------------------------------------------------- */}
      {/* --- TAB 3: COMPILER --- */}
      {/* ---------------------------------------------------- */}
      {activeTab === 'compiler' && (
        <div className="compiler-panel glass-panel no-print" style={{ padding: '32px', textAlign: 'center' }}>
          {compilerProgress === null && (
            <div>
              <div style={{ fontSize: '3rem', marginBottom: '16px' }}>📚</div>
              <h3 style={{ fontSize: '1.5rem', marginBottom: '12px' }}>Curriculum Textbook Compiler</h3>
              <p style={{ color: 'var(--text-secondary)', maxWidth: '500px', margin: '0 auto 24px auto', lineHeight: '1.6' }}>
                Bundle all 42 weeks of lesson contents (including LaTeX mathematical derivations and programming sandboxes) into a single unified layout formatted for printing or saving directly as a PDF.
              </p>
              <button className="action-btn" onClick={handleCompileTextbook}>
                Compile Complete Textbook
              </button>
            </div>
          )}

          {compilerProgress === 'loading' && (
            <div style={{ padding: '24px' }}>
              <div className="spinner" style={{ border: '4px solid rgba(255,255,255,0.1)', borderTop: '4px solid var(--accent-color)', borderRadius: '50%', width: '40px', height: '40px', animation: 'spin 1s linear infinite', margin: '0 auto 16px auto' }}></div>
              <h4 style={{ color: '#fff', marginBottom: '8px' }}>Compiling All Chapters...</h4>
              <p style={{ color: 'var(--text-secondary)' }}>Reading lesson files week0.md through week41.md</p>
            </div>
          )}

          {compilerProgress === 'done' && (
            <div>
              <div style={{ fontSize: '3rem', marginBottom: '16px' }}>✓</div>
              <h3 style={{ fontSize: '1.5rem', marginBottom: '12px', color: '#10b981' }}>Compilation Complete!</h3>
              <p style={{ color: 'var(--text-secondary)', maxWidth: '500px', margin: '0 auto 24px auto', lineHeight: '1.6' }}>
                All chapters have been compiled into the print viewport. Click the button below to open the system print prompt where you can select "Save to PDF".
              </p>
              
              <div style={{ display: 'flex', gap: '16px', justifyContent: 'center' }}>
                <button className="action-btn" onClick={handlePrintTextbook}>
                  Print / Save to PDF
                </button>
                <button 
                  className="external-link-btn" 
                  onClick={() => setCompilerProgress(null)}
                  style={{ background: 'transparent' }}
                >
                  Compile Again
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ---------------------------------------------------- */}
      {/* --- TAB 4: NN BUILDER --- */}
      {/* ---------------------------------------------------- */}
      {activeTab === 'nnbuilder' && (
        <div className="nnbuilder-panel glass-panel no-print" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
            <div>
              <h2 style={{ fontSize: '1.8rem', margin: 0 }}>Visual NN Architecture Builder</h2>
              <p style={{ color: 'var(--text-secondary)', marginTop: '6px' }}>
                Construct neural networks layer-by-layer. Tensor shapes and total parameters are calculated dynamically.
              </p>
            </div>
            <div style={{ display: 'flex', gap: '10px' }}>
              <button 
                className="action-btn"
                style={{ padding: '8px 16px', fontSize: '0.85rem' }}
                onClick={() => setLayers([
                  { id: '1', type: 'input', width: 28, height: 28, channels: 1 },
                  { id: '2', type: 'conv2d', filters: 32, kernelSize: 3, stride: 1, padding: 'same' },
                  { id: '3', type: 'maxpool2d', poolSize: 2, stride: 2 },
                  { id: '4', type: 'flatten' },
                  { id: '5', type: 'dense', units: 128, activation: 'relu' },
                  { id: '6', type: 'dense', units: 10, activation: 'softmax' }
                ])}
              >
                Preset: MNIST CNN
              </button>
              <button 
                className="action-btn"
                style={{ padding: '8px 16px', fontSize: '0.85rem' }}
                onClick={() => setLayers([
                  { id: '1', type: 'input', width: 10, height: 1, channels: 1 },
                  { id: '2', type: 'dense', units: 64, activation: 'relu' },
                  { id: '3', type: 'dense', units: 32, activation: 'relu' },
                  { id: '4', type: 'dense', units: 1, activation: 'linear' }
                ])}
              >
                Preset: Tabular MLP
              </button>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap-reverse' }}>
            {/* Layers List */}
            <div style={{ flex: 2, minWidth: '320px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {computeLayerShapes().map((layer, idx) => (
                <div 
                  key={layer.id} 
                  className="glass-panel" 
                  style={{ 
                    padding: '16px', 
                    borderLeft: `5px solid ${
                      layer.type === 'input' ? '#8b5cf6' : 
                      layer.type === 'conv2d' ? '#10b981' : 
                      layer.type === 'maxpool2d' ? '#3b82f6' : 
                      layer.type === 'flatten' ? '#f59e0b' : '#ff007f'
                    }` 
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontWeight: 'bold', fontSize: '1.1rem', textTransform: 'uppercase' }}>
                        {idx + 1}. {layer.type}
                      </span>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                        {layer.details}
                      </span>
                    </div>
                    {layer.type !== 'input' && (
                      <button 
                        onClick={() => removeLayer(layer.id)}
                        style={{ background: 'transparent', border: 'none', color: '#ff6b6b', cursor: 'pointer', fontSize: '1.2rem' }}
                        title="Delete Layer"
                      >
                        ✕
                      </button>
                    )}
                  </div>

                  {/* Layer Parameter Editors */}
                  <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', alignItems: 'center' }}>
                    {layer.type === 'input' && (
                      <>
                        <div>
                          <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Width</label>
                          <input 
                            type="number" 
                            value={layer.width} 
                            onChange={(e) => updateLayer(layer.id, { width: parseInt(e.target.value) || 1 })}
                            style={{ width: '80px', padding: '6px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '4px' }}
                          />
                        </div>
                        <div>
                          <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Height</label>
                          <input 
                            type="number" 
                            value={layer.height} 
                            onChange={(e) => updateLayer(layer.id, { height: parseInt(e.target.value) || 1 })}
                            style={{ width: '80px', padding: '6px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '4px' }}
                          />
                        </div>
                        <div>
                          <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Channels</label>
                          <input 
                            type="number" 
                            value={layer.channels} 
                            onChange={(e) => updateLayer(layer.id, { channels: parseInt(e.target.value) || 1 })}
                            style={{ width: '80px', padding: '6px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '4px' }}
                          />
                        </div>
                      </>
                    )}

                    {layer.type === 'conv2d' && (
                      <>
                        <div>
                          <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Filters</label>
                          <input 
                            type="number" 
                            value={layer.filters} 
                            onChange={(e) => updateLayer(layer.id, { filters: parseInt(e.target.value) || 1 })}
                            style={{ width: '80px', padding: '6px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '4px' }}
                          />
                        </div>
                        <div>
                          <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Kernel</label>
                          <select 
                            value={layer.kernelSize} 
                            onChange={(e) => updateLayer(layer.id, { kernelSize: parseInt(e.target.value) })}
                            style={{ padding: '6px', background: '#111', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '4px' }}
                          >
                            <option value="1">1x1</option>
                            <option value="3">3x3</option>
                            <option value="5">5x5</option>
                          </select>
                        </div>
                        <div>
                          <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Padding</label>
                          <select 
                            value={layer.padding} 
                            onChange={(e) => updateLayer(layer.id, { padding: e.target.value })}
                            style={{ padding: '6px', background: '#111', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '4px' }}
                          >
                            <option value="same">Same</option>
                            <option value="valid">Valid</option>
                          </select>
                        </div>
                      </>
                    )}

                    {layer.type === 'maxpool2d' && (
                      <>
                        <div>
                          <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Pool Size</label>
                          <select 
                            value={layer.poolSize} 
                            onChange={(e) => updateLayer(layer.id, { poolSize: parseInt(e.target.value) })}
                            style={{ padding: '6px', background: '#111', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '4px' }}
                          >
                            <option value="2">2x2</option>
                            <option value="3">3x3</option>
                          </select>
                        </div>
                      </>
                    )}

                    {layer.type === 'dense' && (
                      <>
                        <div>
                          <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Neurons</label>
                          <input 
                            type="number" 
                            value={layer.units} 
                            onChange={(e) => updateLayer(layer.id, { units: parseInt(e.target.value) || 1 })}
                            style={{ width: '80px', padding: '6px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '4px' }}
                          />
                        </div>
                        <div>
                          <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Activation</label>
                          <select 
                            value={layer.activation} 
                            onChange={(e) => updateLayer(layer.id, { activation: e.target.value })}
                            style={{ padding: '6px', background: '#111', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '4px' }}
                          >
                            <option value="relu">ReLU</option>
                            <option value="sigmoid">Sigmoid</option>
                            <option value="softmax">Softmax</option>
                            <option value="linear">Linear</option>
                          </select>
                        </div>
                      </>
                    )}

                    {layer.type === 'dropout' && (
                      <div>
                        <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Drop Rate</label>
                        <input 
                          type="number" 
                          min="0" 
                          max="0.9" 
                          step="0.05"
                          value={layer.rate} 
                          onChange={(e) => updateLayer(layer.id, { rate: parseFloat(e.target.value) || 0 })}
                          style={{ width: '80px', padding: '6px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '4px' }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {/* Add Layer Controls */}
              <div className="glass-panel" style={{ padding: '16px', display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center' }}>
                <span style={{ alignSelf: 'center', marginRight: '8px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Add Layer:</span>
                <button className="action-btn" style={{ padding: '6px 12px', fontSize: '0.8rem' }} onClick={() => addLayer('conv2d')}>+ Conv2D</button>
                <button className="action-btn" style={{ padding: '6px 12px', fontSize: '0.8rem' }} onClick={() => addLayer('maxpool2d')}>+ MaxPool2D</button>
                <button className="action-btn" style={{ padding: '6px 12px', fontSize: '0.8rem' }} onClick={() => addLayer('flatten')}>+ Flatten</button>
                <button className="action-btn" style={{ padding: '6px 12px', fontSize: '0.8rem' }} onClick={() => addLayer('dense')}>+ Dense</button>
                <button className="action-btn" style={{ padding: '6px 12px', fontSize: '0.8rem' }} onClick={() => addLayer('dropout')}>+ Dropout</button>
              </div>
            </div>

            {/* Architecture Metrics Card */}
            <div style={{ flex: 1, minWidth: '280px' }}>
              <div className="glass-panel" style={{ padding: '24px', position: 'sticky', top: '20px' }}>
                <h3 style={{ fontSize: '1.2rem', marginBottom: '16px', color: 'var(--accent-color)' }}>Model Summary</h3>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginBottom: '24px' }}>
                  <div>
                    <div style={{ fontSize: '1.8rem', fontWeight: 'bold' }}>
                      {computeLayerShapes().reduce((sum, layer) => sum + layer.parameters, 0).toLocaleString()}
                    </div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                      Total Learnable Parameters
                    </div>
                  </div>

                  <div>
                    <div style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>
                      {computeLayerShapes().length} Layers
                    </div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                      Depth Complexity
                    </div>
                  </div>
                </div>

                <div style={{ borderTop: '1px solid var(--glass-border)', paddingTop: '16px' }}>
                  <h4 style={{ fontSize: '0.9rem', marginBottom: '8px', color: 'var(--text-secondary)' }}>Tensor Shapes Trace</h4>
                  <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: '0.85rem' }}>
                    {computeLayerShapes().map((layer, idx) => (
                      <li key={layer.id} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                        <span>L{idx+1} ({layer.type}):</span>
                        <span style={{ fontFamily: 'monospace', color: 'var(--accent-color)' }}>{layer.outputShape}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div style={{ borderTop: '1px solid var(--glass-border)', paddingTop: '16px', marginTop: '16px' }}>
                  <h4 style={{ fontSize: '0.9rem', marginBottom: '12px', color: 'var(--text-secondary)' }}>Model Code Export</h4>
                  
                  <div style={{ display: 'flex', gap: '6px', marginBottom: '12px' }}>
                    {['pytorch', 'keras', 'jax'].map(fw => (
                      <button
                        key={fw}
                        onClick={() => setExportFramework(fw)}
                        style={{
                          flex: 1,
                          padding: '6px',
                          fontSize: '0.75rem',
                          background: exportFramework === fw ? 'var(--accent-color)' : 'rgba(255,255,255,0.05)',
                          color: '#fff',
                          border: '1px solid var(--glass-border)',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          textTransform: 'capitalize'
                        }}
                      >
                        {fw === 'pytorch' ? 'PyTorch' : fw === 'keras' ? 'Keras' : 'JAX / Flax'}
                      </button>
                    ))}
                  </div>

                  <textarea
                    readOnly
                    value={generateCode()}
                    style={{
                      width: '100%',
                      height: '180px',
                      padding: '8px',
                      background: 'rgba(0,0,0,0.3)',
                      color: '#a9b1d6',
                      border: '1px solid var(--glass-border)',
                      borderRadius: '6px',
                      fontFamily: 'monospace',
                      fontSize: '0.75rem',
                      outline: 'none',
                      resize: 'none'
                    }}
                    onClick={(e) => {
                      e.target.select();
                      navigator.clipboard.writeText(generateCode());
                    }}
                    title="Click to copy code"
                  />
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '4px', textAlign: 'right' }}>
                    Click inside text box to select & copy
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ---------------------------------------------------- */}
      {/* --- TAB 5: PYTHON SCRATCHPAD --- */}
      {/* ---------------------------------------------------- */}
      {activeTab === 'scratchpad' && (
        <div className="scratchpad-panel glass-panel no-print" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
            <div>
              <h2 style={{ fontSize: '1.8rem', margin: 0 }}>NumPy & Matrix Code Playground</h2>
              <p style={{ color: 'var(--text-secondary)', marginTop: '6px' }}>
                Write custom Python scripts or solve guided math challenges. NumPy, SciPy, and Matplotlib are fully supported.
              </p>
            </div>

            {/* Mode Switcher */}
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <button
                className="action-btn"
                style={{
                  background: scratchpadMode === 'free' ? 'var(--accent-color)' : 'rgba(255, 255, 255, 0.05)',
                  padding: '8px 16px',
                  fontSize: '0.85rem'
                }}
                onClick={() => setScratchpadMode('free')}
              >
                Free Play
              </button>
              <button
                className="action-btn"
                style={{
                  background: scratchpadMode === 'guided' ? 'var(--accent-color)' : 'rgba(255, 255, 255, 0.05)',
                  padding: '8px 16px',
                  fontSize: '0.85rem'
                }}
                onClick={() => setScratchpadMode('guided')}
              >
                Guided Challenges
              </button>
            </div>
            
            {scratchpadMode === 'free' && (
              <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Templates:</span>
                <button className="action-btn" style={{ padding: '6px 12px', fontSize: '0.8rem' }} onClick={() => loadScratchpadTemplate('numpy')}>NumPy Product</button>
                <button className="action-btn" style={{ padding: '6px 12px', fontSize: '0.8rem' }} onClick={() => loadScratchpadTemplate('matplotlib')}>Matplotlib Contour</button>
                <button className="action-btn" style={{ padding: '6px 12px', fontSize: '0.8rem' }} onClick={() => loadScratchpadTemplate('regression')}>Gradient Descent</button>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
            {/* Guided Left Panel */}
            {scratchpadMode === 'guided' && (
              <div style={{ flex: '0 0 350px', display: 'flex', flexDirection: 'column', gap: '16px', borderRight: '1px solid var(--glass-border)', paddingRight: '24px', minWidth: '300px' }}>
                <div>
                  <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '8px' }}>Select Challenge:</label>
                  <select
                    value={currentChallengeIdx}
                    onChange={(e) => setCurrentChallengeIdx(parseInt(e.target.value))}
                    style={{
                      width: '100%',
                      padding: '10px',
                      background: 'rgba(0, 0, 0, 0.4)',
                      color: '#fff',
                      border: '1px solid var(--glass-border)',
                      borderRadius: '6px',
                      fontSize: '0.9rem',
                      outline: 'none'
                    }}
                  >
                    {challenges.map((c, idx) => (
                      <option key={c.id} value={idx}>{c.title}</option>
                    ))}
                  </select>
                </div>

                <div className="glass-panel" style={{ padding: '16px', background: 'rgba(255, 255, 255, 0.01)', overflowY: 'auto', flex: 1, minHeight: '280px' }}>
                  <h4 style={{ fontSize: '1.1rem', marginBottom: '12px', color: 'var(--accent-color)' }}>
                    {challenges[currentChallengeIdx].title}
                  </h4>
                  <div style={{ fontSize: '0.9rem', lineHeight: '1.6', color: 'var(--text-primary)' }}>
                    <ReactMarkdown
                      children={challenges[currentChallengeIdx].description}
                      remarkPlugins={[remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                    />
                  </div>
                </div>

                <button
                  className="action-btn"
                  style={{ width: '100%', padding: '8px', fontSize: '0.85rem', background: 'rgba(255, 255, 255, 0.03)' }}
                  onClick={() => setScratchpadCode(challenges[currentChallengeIdx].starterCode)}
                >
                  ↻ Reset Starter Code
                </button>
              </div>
            )}

            {/* Editor Area */}
            <div style={{ flex: 2, minWidth: '320px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>main.py</span>
                <button 
                  className="action-btn" 
                  onClick={handleRunScratchpad}
                  disabled={pyodideLoading || isScratchpadRunning}
                  style={{
                    background: pyodideLoading || isScratchpadRunning ? 'rgba(255,255,255,0.05)' : 'var(--accent-color)',
                    padding: '8px 20px',
                    fontSize: '0.9rem'
                  }}
                >
                  {pyodideLoading ? 'Loading Runtime...' : isScratchpadRunning ? 'Running...' : scratchpadMode === 'guided' ? '▶ Run & Check Answer' : '▶ Run Script'}
                </button>
              </div>

              <textarea
                value={scratchpadCode}
                onChange={(e) => setScratchpadCode(e.target.value)}
                style={{
                  width: '100%',
                  height: '420px',
                  padding: '16px',
                  background: 'rgba(0, 0, 0, 0.4)',
                  color: '#fff',
                  border: '1px solid var(--glass-border)',
                  borderRadius: '12px',
                  fontFamily: '"Fira Code", monospace',
                  fontSize: '0.95rem',
                  lineHeight: '1.5',
                  outline: 'none',
                  resize: 'none'
                }}
              />
            </div>

            {/* Output Area */}
            <div style={{ flex: 1.5, minWidth: '300px', display: 'flex', flexDirection: 'column' }}>
              <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>Console Output</div>
              
              <div 
                style={{
                  flex: 1,
                  background: '#040711',
                  border: '1px solid var(--glass-border)',
                  borderRadius: '12px',
                  padding: '20px',
                  fontFamily: 'monospace',
                  fontSize: '0.9rem',
                  overflowY: 'auto',
                  minHeight: '420px',
                  whiteSpace: 'pre-wrap'
                }}
              >
                {pyodideError && <div style={{ color: '#ff6b6b' }}>Error loading Pyodide: {pyodideError.message}</div>}
                
                {scratchpadOutput ? (
                  scratchpadOutput.split('[[IMAGE:').map((part, index) => {
                    if (index === 0) return part ? <div key={index}>{part}</div> : null;
                    
                    const endIndex = part.indexOf(']]');
                    if (endIndex === -1) return <div key={index}>[[IMAGE:{part}</div>;
                    
                    const b64 = part.substring(0, endIndex);
                    const rest = part.substring(endIndex + 2);
                    
                    return (
                      <React.Fragment key={index}>
                        <div style={{ margin: '16px 0', background: 'white', display: 'inline-block', padding: '16px', borderRadius: '8px', maxWidth: '100%' }}>
                          <img src={`data:image/png;base64,${b64}`} alt="matplotlib output" style={{ maxWidth: '100%', display: 'block' }} />
                        </div>
                        {rest.trim() && <div>{rest}</div>}
                      </React.Fragment>
                    );
                  })
                ) : (
                  <span style={{ color: 'var(--text-secondary)', fontStyle: 'italic' }}>Output console empty. Run script to view output.</span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ---------------------------------------------------- */}
      {/* --- TAB: 3D DATA EXPLORER --- */}
      {/* ---------------------------------------------------- */}
      {activeTab === '3dexplorer' && (
        <div className="glass-panel no-print" style={{
          padding: 0, borderRadius: 12, overflow: 'hidden',
          height: 'calc(100vh - 260px)',
          minHeight: 520,
          display: 'flex', flexDirection: 'column',
        }}>
          <Suspense fallback={
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1, color: 'rgba(255,255,255,0.4)', fontSize: '1rem', gap: 12 }}>
              <span style={{ fontSize: '1.5rem' }}>⟳</span>
              Loading 3D Engine…
            </div>
          }>
            <DataVisualizer3D />
          </Suspense>
        </div>
      )}

      {/* --- HIDDEN COMPILED VIEWPORT FOR PRINTING --- */}
      {compilerProgress === 'done' && (
        <div className="print-only-textbook-viewport">
          <ReactMarkdown
            children={compilerText}
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex]}
            components={{
              code({node, inline, className, children, ...props}) {
                return (
                  <code className={className} {...props}>
                    {children}
                  </code>
                )
              }
            }}
          />
        </div>
      )}
    </div>
  );
}

export default InteractiveLab;
