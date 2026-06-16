import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

function CheatSheetDrawer({ isOpen, onClose }) {
  const [activeTab, setActiveTab] = useState('math');

  const mathRef = `
### Linear Algebra & Calculus Reference

#### 1. Matrix Transpose & Inverse Identities
* **Product Transpose**: $(AB)^T = B^T A^T$
* **Product Inverse**: $(AB)^{-1} = B^{-1} A^{-1}$
* **Orthogonal Matrix**: $Q^T Q = I \\implies Q^{-1} = Q^T$

#### 2. Eigendecomposition & Singular Value Decomposition (SVD)
* **Eigenvalues / Eigenvectors**: $A v = \\lambda v$
* **SVD (Singular Value Decomposition)**: $A = U \\Sigma V^T$
  * $U$: Left singular vectors (eigenvectors of $A A^T$)
  * $V$: Right singular vectors (eigenvectors of $A^T A$)
  * $\\Sigma$: Singular values (square roots of eigenvalues of $A^T A$)

#### 3. Vector & Matrix Derivatives (Gradients)
* **Linear Form**: $\\nabla_x (a^T x) = a$
* **Quadratic Form (Symmetric $A$)**: $\\nabla_x (x^T A x) = 2 A x$
* **Mean Squared Error Gradient**: 
  $$L(w) = \\frac{1}{N} \\|Xw - y\\|^2 \\implies \\nabla_w L(w) = \\frac{2}{N} X^T (Xw - y)$$
* **Chain Rule (Multivariate)**: 
  $$\\frac{\\partial L}{\\partial x} = \\frac{\\partial L}{\\partial y} \\cdot \\frac{\\partial y}{\\partial x}$$
`;

  const pytorchRef = `
### PyTorch Cheat Sheet

#### 1. Common Tensor Operations
\`\`\`python
import torch

# Create Tensors
x = torch.tensor([1.0, 2.0, 3.0])
A = torch.randn(3, 3) # Random normal
I = torch.eye(3)       # Identity matrix

# Dot Product, Matrix-Vector, Matrix-Matrix
y = torch.dot(x, x)       # Inner product of 1D tensors
b = torch.matmul(A, x)    # Or: b = A @ x
C = torch.matmul(A, A)    # Or: C = A @ A

# Transpose & Shape
A_t = A.t()               # Or: A.T
shape = A.shape           # (rows, cols)
x_reshaped = x.view(-1, 1) # Reshape/unsqueeze
\`\`\`

#### 2. Autograd & Training Loop
\`\`\`python
import torch.nn as nn
import torch.optim as optim

# Define Model
class LinearReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)
        
    def forward(self, x):
        return self.fc(x)

model = LinearReg()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training Step
optimizer.zero_grad()      # Clear gradients
outputs = model(inputs)    # Forward pass
loss = criterion(outputs, targets) # Compute loss
loss.backward()            # Backward pass (compute grads)
optimizer.step()           # Update weights
\`\`\`
`;

  const numpyRef = `
### NumPy Cheat Sheet

#### 1. Arrays & Creation
\`\`\`python
import numpy as np

# Create Arrays
a = np.array([1, 2, 3])
B = np.array([[1, 2], [3, 4]])
Z = np.zeros((3, 3))
O = np.ones((2, 2))
R = np.random.randn(3, 3) # Std normal distribution
\`\`\`

#### 2. Mathematical Operations
\`\`\`python
# Vector / Matrix Operations
dot_prod = np.dot(a, a)       # Or: a @ a
matmul = np.dot(B, B)         # Or: B @ B
B_t = B.T                     # Transpose
B_inv = np.linalg.inv(B)      # Inverse matrix

# Element-wise operations
exp_a = np.exp(a)             # Element-wise e^x
sum_a = np.sum(a)             # Sum all elements
max_val = np.max(B, axis=0)   # Max values along columns
\`\`\`

#### 3. Linear Algebra Functions
\`\`\`python
# Eigenvalues & Eigenvectors
eigenvals, eigenvecs = np.linalg.eig(B)

# SVD
U, S, Vt = np.linalg.svd(B)
\`\`\`
`;

  if (!isOpen) return null;

  return (
    <div 
      style={{
        position: 'fixed',
        top: 0,
        right: 0,
        width: '450px',
        height: '100vh',
        background: 'rgba(11, 15, 25, 0.9)',
        backdropFilter: 'blur(25px)',
        WebkitBackdropFilter: 'blur(25px)',
        borderLeft: '1px solid var(--glass-border)',
        boxShadow: '-10px 0 30px rgba(0, 0, 0, 0.5)',
        zIndex: 998,
        display: 'flex',
        flexDirection: 'column',
        animation: 'slideInRight 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        color: 'var(--text-primary)'
      }}
    >
      <style>{`
        @keyframes slideInRight {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
        .ref-tab-btn {
          flex: 1;
          padding: 12px;
          background: transparent;
          border: none;
          color: var(--text-secondary);
          border-bottom: 2px solid transparent;
          font-weight: 500;
          font-size: 0.9rem;
          cursor: pointer;
          transition: var(--transition);
        }
        .ref-tab-btn.active {
          color: var(--accent-color);
          border-bottom-color: var(--accent-color);
          background: var(--accent-faint);
        }
        .ref-content-body h3, .ref-content-body h4 {
          margin-top: 18px;
          margin-bottom: 8px;
          color: var(--accent-color);
        }
        .ref-content-body pre {
          background: rgba(0,0,0,0.4);
          border: 1px solid var(--glass-border);
          border-radius: 6px;
          padding: 12px;
          font-family: monospace;
          font-size: 0.8rem;
          overflow-x: auto;
          margin: 12px 0;
        }
        .ref-content-body p, .ref-content-body li {
          font-size: 0.9rem;
          color: var(--text-secondary);
          margin-bottom: 6px;
          line-height: 1.5;
        }
        .ref-content-body ul {
          margin-left: 20px;
          margin-bottom: 12px;
        }
      `}</style>

      {/* Drawer Header */}
      <div style={{ padding: '24px', borderBottom: '1px solid var(--glass-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '64px' }}>
        <h3 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 'bold' }}>Quick Reference Hub</h3>
        <button 
          onClick={onClose}
          style={{
            background: 'rgba(255, 255, 255, 0.05)',
            border: 'none',
            color: '#fff',
            cursor: 'pointer',
            padding: '6px 12px',
            borderRadius: '4px',
            fontSize: '0.8rem'
          }}
        >
          Hide
        </button>
      </div>

      {/* Drawer Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid var(--glass-border)' }}>
        <button 
          className={`ref-tab-btn ${activeTab === 'math' ? 'active' : ''}`}
          onClick={() => setActiveTab('math')}
        >
          Math Formulas
        </button>
        <button 
          className={`ref-tab-btn ${activeTab === 'pytorch' ? 'active' : ''}`}
          onClick={() => setActiveTab('pytorch')}
        >
          PyTorch
        </button>
        <button 
          className={`ref-tab-btn ${activeTab === 'numpy' ? 'active' : ''}`}
          onClick={() => setActiveTab('numpy')}
        >
          NumPy
        </button>
      </div>

      {/* Drawer Content */}
      <div 
        className="ref-content-body"
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '24px'
        }}
      >
        <ReactMarkdown
          children={
            activeTab === 'math' ? mathRef :
            activeTab === 'pytorch' ? pytorchRef :
            numpyRef
          }
          remarkPlugins={[remarkMath]}
          rehypePlugins={[rehypeKatex]}
        />
      </div>
    </div>
  );
}

export default CheatSheetDrawer;
