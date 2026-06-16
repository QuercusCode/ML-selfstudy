import React, { useRef, useMemo, useState, useCallback, useEffect, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Text, Line } from '@react-three/drei';
import * as THREE from 'three';
import { generateDataset, DATASETS } from '../utils/dataGenerators';

// ──────────────────────────────────────────────
// Error boundary
// ──────────────────────────────────────────────
class CanvasErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { hasError: false }; }
  static getDerivedStateFromError() { return { hasError: true }; }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ display:'flex', alignItems:'center', justifyContent:'center',
          height:'100%', flexDirection:'column', gap:12,
          color:'rgba(255,255,255,0.45)', fontSize:'0.9rem' }}>
          <div style={{ fontSize:'2rem' }}>⚠️</div>
          <div>3D renderer error — try refreshing</div>
          <button onClick={() => this.setState({ hasError: false })}
            style={{ padding:'8px 20px', borderRadius:8, border:'none',
              background:'rgba(167,139,250,0.2)', color:'#a78bfa', cursor:'pointer' }}>
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// ──────────────────────────────────────────────
// Digit template matrix for mock grid rendering (5x5 scaled to 10x10)
// ──────────────────────────────────────────────
const DIGIT_TEMPLATES = {
  '0': [
    [0,1,1,1,0],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [0,1,1,1,0]
  ],
  '1': [
    [0,0,1,0,0],
    [0,1,1,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,1,1,1,0]
  ],
  '2': [
    [0,1,1,1,0],
    [1,0,0,0,1],
    [0,0,1,1,0],
    [0,1,0,0,0],
    [1,1,1,1,1]
  ],
  '3': [
    [1,1,1,1,0],
    [0,0,0,0,1],
    [0,1,1,1,0],
    [0,0,0,0,1],
    [1,1,1,1,0]
  ],
  '4': [
    [1,0,0,1,0],
    [1,0,0,1,0],
    [1,1,1,1,1],
    [0,0,0,1,0],
    [0,0,0,1,0]
  ],
  '5': [
    [1,1,1,1,1],
    [1,0,0,0,0],
    [1,1,1,1,0],
    [0,0,0,0,1],
    [1,1,1,1,0]
  ],
  '6': [
    [0,1,1,1,0],
    [1,0,0,0,0],
    [1,1,1,1,0],
    [1,0,0,0,1],
    [0,1,1,1,0]
  ],
  '7': [
    [1,1,1,1,1],
    [0,0,0,1,0],
    [0,0,1,0,0],
    [0,1,0,0,0],
    [1,0,0,0,0]
  ],
  '8': [
    [0,1,1,1,0],
    [1,0,0,0,1],
    [0,1,1,1,0],
    [1,0,0,0,1],
    [0,1,1,1,0]
  ],
  '9': [
    [0,1,1,1,0],
    [1,0,0,0,1],
    [0,1,1,1,1],
    [0,0,0,0,1],
    [0,1,1,1,0]
  ]
};

function generatePixelGrid(digitStr, seed) {
  const template = DIGIT_TEMPLATES[digitStr] || DIGIT_TEMPLATES['0'];
  const size = 10;
  const grid = Array.from({ length: size }, () => new Float32Array(size));
  
  let randVal = Math.sin(seed) * 10000;
  const pseudoRand = () => {
    randVal = Math.sin(randVal) * 10000;
    return randVal - Math.floor(randVal);
  };

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const tr = Math.floor(r / 2);
      const tc = Math.floor(c / 2);
      const isSet = template[tr][tc] === 1;
      
      if (isSet) {
        grid[r][c] = 0.4 + pseudoRand() * 0.6;
      } else {
        grid[r][c] = pseudoRand() * 0.12;
      }
    }
  }
  return grid;
}

// ──────────────────────────────────────────────
// Instanced mesh with smooth morphing and selection states
// ──────────────────────────────────────────────
const _dummy = new THREE.Object3D();

function DataPoints({ points, hoveredClass, selectedIdx, hoveredIdx, onHoverIndex, onClickIndex }) {
  const meshRef = useRef();
  const count = points.length;
  const currentPositionsRef = useRef([]);

  // Highlighting selected/hovered nodes dynamically in color buffer
  const colorArray = useMemo(() => {
    const temp = new THREE.Color();
    const arr = new Float32Array(count * 3);
    points.forEach((p, i) => {
      let colStr = p.color;
      if (selectedIdx === i) {
        colStr = '#ffffff'; // White for selected
      } else if (hoveredIdx === i) {
        colStr = '#ffffff'; // White for hovered
      }
      temp.set(colStr);
      arr[i * 3] = temp.r;
      arr[i * 3 + 1] = temp.g;
      arr[i * 3 + 2] = temp.b;
    });
    return arr;
  }, [points, count, selectedIdx, hoveredIdx]);

  // Smooth position and scale morph transition updates in render loop
  useFrame((state) => {
    if (!meshRef.current) return;
    const t = state.clock.getElapsedTime();
    const mesh = meshRef.current;
    const cur = currentPositionsRef.current;

    // Rescale & initialize ref positions array
    if (cur.length !== count) {
      const oldLen = cur.length;
      if (oldLen < count) {
        for (let i = oldLen; i < count; i++) {
          cur.push({ x: 0, y: 0, z: 0, scale: 0 });
        }
      } else {
        cur.length = count;
      }
    }

    points.forEach((p, i) => {
      const targetX = p.x;
      const targetY = p.y + Math.sin(t * 0.5 + i * 0.12) * 0.05; // Float effect
      const targetZ = p.z;

      // Determine size scale based on hover/class highlights
      let targetScale = 1.0;
      const isClassHovered = hoveredClass === null || p.classIdx === hoveredClass;

      if (selectedIdx === i) {
        targetScale = 1.6;
      } else if (hoveredIdx === i) {
        targetScale = 1.55;
      } else if (!isClassHovered) {
        targetScale = 0.22;
      } else {
        targetScale = 0.85;
      }

      // Lerp transition calculations
      cur[i].x += (targetX - cur[i].x) * 0.08;
      cur[i].y += (targetY - cur[i].y) * 0.08;
      cur[i].z += (targetZ - cur[i].z) * 0.08;
      cur[i].scale += (targetScale - cur[i].scale) * 0.12;

      _dummy.position.set(cur[i].x, cur[i].y, cur[i].z);
      _dummy.scale.setScalar(cur[i].scale);
      _dummy.updateMatrix();
      mesh.setMatrixAt(i, _dummy.matrix);
    });

    mesh.instanceMatrix.needsUpdate = true;
    mesh.computeBoundingSphere();
    mesh.computeBoundingBox();
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[null, null, count]}
      frustumCulled={false}
      onPointerMove={(e) => {
        e.stopPropagation();
        onHoverIndex(e.instanceId);
      }}
      onPointerOut={(e) => {
        onHoverIndex(null);
      }}
      onClick={(e) => {
        e.stopPropagation();
        onClickIndex(e.instanceId);
      }}
    >
      <sphereGeometry args={[0.13, 10, 10]} />
      <meshStandardMaterial
        roughness={0.2}
        metalness={0.3}
        toneMapped={false}
      />
      <instancedBufferAttribute
        key={colorArray}
        attach="instanceColor"
        args={[colorArray, 3]}
      />
    </instancedMesh>
  );
}

// ──────────────────────────────────────────────
// 3D Distance Ruler Connector Line
// ──────────────────────────────────────────────
function DistanceRuler({ selectedPt, hoveredPt }) {
  if (!selectedPt || !hoveredPt) return null;

  const start = new THREE.Vector3(selectedPt.x, selectedPt.y, selectedPt.z);
  const end = new THREE.Vector3(hoveredPt.x, hoveredPt.y, hoveredPt.z);
  const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
  const distance = Math.sqrt(
    Math.pow(selectedPt.x - hoveredPt.x, 2) +
    Math.pow(selectedPt.y - hoveredPt.y, 2) +
    Math.pow(selectedPt.z - hoveredPt.z, 2)
  );

  return (
    <group>
      <Line
        points={[start, end]}
        color="#10b981"
        lineWidth={2}
        dashed
        dashScale={5}
        gapSize={0.4}
      />
      <Text
        position={[midpoint.x, midpoint.y + 0.25, midpoint.z]}
        fontSize={0.25}
        color="#10b981"
        backgroundColor="#071224"
        padding={[0.06, 0.16]}
        borderRadius={0.04}
        anchorX="center"
        anchorY="middle"
      >
        {`d = ${distance.toFixed(2)}`}
      </Text>
    </group>
  );
}

// ──────────────────────────────────────────────
// Axis lines
// ──────────────────────────────────────────────
function Axes() {
  const origin = new THREE.Vector3(-5, -5, -5);
  const defs = [
    { dir: new THREE.Vector3(1,0,0), color:'#f94144', label:'PC1' },
    { dir: new THREE.Vector3(0,1,0), color:'#90be6d', label:'PC2' },
    { dir: new THREE.Vector3(0,0,1), color:'#577590', label:'PC3' },
  ];

  return (
    <group>
      {defs.map(({ dir, color, label }) => {
        const end = origin.clone().addScaledVector(dir, 5);
        const geom = new THREE.BufferGeometry().setFromPoints([origin, end]);
        const mat  = new THREE.LineBasicMaterial({ color });
        const line = new THREE.Line(geom, mat);
        const labelPos = origin.clone().addScaledVector(dir, 5.6);
        return (
          <group key={label}>
            <primitive object={line} />
            <Text position={[labelPos.x, labelPos.y, labelPos.z]}
              fontSize={0.38} color={color} anchorX="center" anchorY="middle">
              {label}
            </Text>
          </group>
        );
      })}
    </group>
  );
}

// ──────────────────────────────────────────────
// Full scene
// ──────────────────────────────────────────────
function Scene({ points, hoveredClass, selectedIdx, hoveredIdx, onHoverIndex, onClickIndex, showShells, shells, autoRotate }) {
  const selectedPt = selectedIdx !== null ? points[selectedIdx] : null;
  const hoveredPt = hoveredIdx !== null ? points[hoveredIdx] : null;

  return (
    <>
      <OrbitControls
        makeDefault
        autoRotate={autoRotate}
        autoRotateSpeed={1.2}
        enableDamping
        dampingFactor={0.08}
        minDistance={3}
        maxDistance={35}
      />
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]}  intensity={1.6} color="#ffffff" />
      <pointLight position={[-8, -5, -8]}  intensity={0.7} color="#a78bfa" />
      <pointLight position={[0,  8, -5]}   intensity={0.5} color="#43aa8b" />

      <Stars radius={55} depth={40} count={2500} factor={3} saturation={0.4} fade speed={0.5} />
      <gridHelper args={[22, 30, '#1e3a5f', '#0f1f35']} position={[0,-5,0]} />
      <Axes />

      {/* Render wireframe bounding shells */}
      {showShells && shells.map((shell, idx) => {
        const hovered = hoveredClass === null || hoveredClass === shell.classIdx;
        return (
          <mesh key={idx} position={shell.centroid}>
            <sphereGeometry args={[shell.radius, 20, 20]} />
            <meshBasicMaterial
              color={shell.color}
              wireframe
              transparent
              opacity={hovered ? 0.07 : 0.012}
            />
          </mesh>
        );
      })}

      <DataPoints
        points={points}
        hoveredClass={hoveredClass}
        selectedIdx={selectedIdx}
        hoveredIdx={hoveredIdx}
        onHoverIndex={onHoverIndex}
        onClickIndex={onClickIndex}
      />

      <DistanceRuler selectedPt={selectedPt} hoveredPt={hoveredPt} />
    </>
  );
}

// ──────────────────────────────────────────────
// Legend overlay
// ──────────────────────────────────────────────
const CLASS_COLORS = [
  '#f94144','#f3722c','#f8961e','#f9c74f','#90be6d',
  '#43aa8b','#4d908e','#577590','#a98fe5','#f72585',
];

function LegendOverlay({ datasetKey, hoveredClass, onHoverClass }) {
  const spec = DATASETS[datasetKey];
  return (
    <div style={{
      position:'absolute', top:14, right:14,
      background:'rgba(5,15,35,0.88)',
      backdropFilter:'blur(14px)',
      border:'1px solid rgba(255,255,255,0.07)',
      borderRadius:12, padding:'12px 16px',
      minWidth:145, zIndex:10, userSelect:'none',
    }}>
      <div style={{ fontSize:'0.62rem', fontWeight:700, color:'rgba(255,255,255,0.3)',
        letterSpacing:'0.1em', marginBottom:9, textTransform:'uppercase' }}>
        Classes
      </div>
      {spec.classLabels.map((label, i) => (
        <div key={label}
          onMouseEnter={() => onHoverClass(i)}
          onMouseLeave={() => onHoverClass(null)}
          style={{
            display:'flex', alignItems:'center', gap:8,
            marginBottom:5, cursor:'pointer',
            opacity: hoveredClass === null || hoveredClass === i ? 1 : 0.2,
            transition:'opacity 0.15s',
          }}>
          <div style={{
            width:9, height:9, borderRadius:'50%', flexShrink:0,
            background: CLASS_COLORS[i % CLASS_COLORS.length],
            boxShadow:`0 0 5px ${CLASS_COLORS[i % CLASS_COLORS.length]}99`,
          }} />
          <span style={{ fontSize:'0.79rem', color:'rgba(255,255,255,0.82)', fontWeight:500 }}>
            {label}
          </span>
        </div>
      ))}
    </div>
  );
}

// ──────────────────────────────────────────────
// Interactive Details Overlay Card
// ──────────────────────────────────────────────
function InspectPanel({ activePt, allPoints, datasetKey, selectedPt, onClearSelection, onClickIndex, selectedIdx, hoveredIdx }) {
  const neighbors = useMemo(() => {
    if (!activePt || datasetKey !== 'word_embeddings' || !allPoints) return [];
    const siblingPts = allPoints.filter(p => p.classIdx === activePt.classIdx && p.label !== activePt.label);
    const scored = siblingPts.map(p => {
      const dist = Math.sqrt(
        Math.pow(p.x - activePt.x, 2) +
        Math.pow(p.y - activePt.y, 2) +
        Math.pow(p.z - activePt.z, 2)
      );
      const sim = Math.max(0.35, 0.96 - dist * 0.058);
      return { label: p.label, similarity: sim };
    });
    return scored.sort((a, b) => b.similarity - a.similarity).slice(0, 3);
  }, [activePt, allPoints, datasetKey]);

  if (!activePt) {
    return (
      <div style={{
        position:'absolute', top:14, left:14,
        background:'rgba(5,15,35,0.85)', backdropFilter:'blur(14px)',
        border:'1px solid rgba(255,255,255,0.07)', borderRadius:12,
        padding:'16px', width:260, zIndex:10, color:'rgba(255,255,255,0.4)',
        fontSize:'0.82rem', lineHeight:1.4
      }}>
        💡 <strong>Interactive Inspection:</strong>
        <div style={{ marginTop:8, fontSize:'0.76rem', color:'rgba(255,255,255,0.35)' }}>
          Hover over any 3D data point to analyze its local attributes. Click a point to select it as a reference for distance measurement.
        </div>
        {selectedPt && (
          <div style={{ marginTop:12, borderTop:'1px solid rgba(255,255,255,0.08)', paddingTop:10, display:'flex', flexDirection:'column', gap:6 }}>
            <span style={{ fontSize:'0.74rem', color:'#10b981' }}>Selected: <strong>{selectedPt.label}</strong></span>
            <button onClick={onClearSelection} style={{
              background:'rgba(239, 68, 68, 0.15)', color:'#f87171', border:'none',
              padding:'4px 8px', borderRadius:4, fontSize:'0.7rem', cursor:'pointer', alignSelf:'flex-start'
            }}>Clear Selection</button>
          </div>
        )}
      </div>
    );
  }

  // Calculate distance to selected point if exists
  let distanceToSel = null;
  if (selectedPt && selectedPt.label !== activePt.label) {
    distanceToSel = Math.sqrt(
      Math.pow(selectedPt.x - activePt.x, 2) +
      Math.pow(selectedPt.y - activePt.y, 2) +
      Math.pow(selectedPt.z - activePt.z, 2)
    );
  }

  return (
    <div style={{
      position:'absolute', top:14, left:14,
      background:'rgba(5,15,35,0.92)', backdropFilter:'blur(14px)',
      border:'1px solid rgba(255,255,255,0.1)', borderRadius:12,
      padding:'16px', width:260, zIndex:10, color:'#e2e8f0',
      boxShadow:'0 4px 24px rgba(0,0,0,0.4)', transition:'all 0.2s'
    }}>
      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:10 }}>
        <span style={{ fontSize:'0.65rem', textTransform:'uppercase', letterSpacing:'0.05em', color:'rgba(255,255,255,0.4)', fontWeight:700 }}>
          Inspector
        </span>
        <div style={{ width:10, height:10, borderRadius:'50%', background:activePt.color, boxShadow:`0 0 6px ${activePt.color}` }} />
      </div>

      <div style={{ fontSize:'1.1rem', fontWeight:700, color:'#ffffff', marginBottom:4 }}>
        {activePt.label}
      </div>

      <div style={{ fontSize:'0.7rem', color:'rgba(255,255,255,0.4)', marginBottom:12, fontFamily:'monospace' }}>
        X: {activePt.x.toFixed(2)} | Y: {activePt.y.toFixed(2)} | Z: {activePt.z.toFixed(2)}
      </div>

      {/* Dataset-specific widgets */}
      {datasetKey === 'mnist_pca' && (
        <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
          <div style={{ fontSize:'0.78rem', color:'rgba(255,255,255,0.7)' }}>
            Projection Confidence: <strong style={{ color:'#a78bfa' }}>{Math.round(activePt.meta.confidence * 100)}%</strong>
          </div>
          <div style={{ display:'flex', flexDirection:'column', gap:4 }}>
            <span style={{ fontSize:'0.68rem', color:'rgba(255,255,255,0.35)', textTransform:'uppercase', fontWeight:600 }}>Reconstructed Digit Shape</span>
            <div style={{ display:'flex', justifyContent:'center', marginTop:2 }}>
              <div style={{
                display: 'grid', gridTemplateColumns: 'repeat(10, 1fr)', gap: 1.5,
                width: 100, height: 100, background: 'rgba(0,0,0,0.5)',
                padding: 5, borderRadius: 6, border: '1px solid rgba(255,255,255,0.08)'
              }}>
                {Array.from(generatePixelGrid(activePt.meta.classLabel, activePt.meta.idx)).map((row, rIdx) => 
                  Array.from(row).map((val, cIdx) => (
                    <div key={`${rIdx}-${cIdx}`} style={{
                      borderRadius: 1, background: '#a78bfa', opacity: val,
                      boxShadow: val > 0.4 ? '0 0 2px rgba(167,139,250,0.4)' : 'none'
                    }} />
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {datasetKey === 'iris_tsne' && (
        <div style={{ display:'flex', flexDirection:'column', gap:8, fontSize:'0.78rem' }}>
          {[
            { label: 'Sepal Length', value: activePt.meta.sepalLength, max: 8.0, unit:'cm' },
            { label: 'Sepal Width',  value: activePt.meta.sepalWidth,  max: 4.5, unit:'cm' },
            { label: 'Petal Length', value: activePt.meta.petalLength, max: 7.0, unit:'cm' },
            { label: 'Petal Width',  value: activePt.meta.petalWidth,  max: 3.0, unit:'cm' },
          ].map(bar => (
            <div key={bar.label} style={{ display:'flex', flexDirection:'column', gap:2 }}>
              <div style={{ display:'flex', justifyContent:'space-between', color:'rgba(255,255,255,0.7)' }}>
                <span>{bar.label}</span>
                <span>{bar.value} {bar.unit}</span>
              </div>
              <div style={{ height:5, background:'rgba(255,255,255,0.06)', borderRadius:3, overflow:'hidden' }}>
                <div style={{ height:'100%', width:`${(bar.value / bar.max) * 100}%`, background:activePt.color, borderRadius:3 }} />
              </div>
            </div>
          ))}
        </div>
      )}

      {datasetKey === 'word_embeddings' && (
        <div style={{ display:'flex', flexDirection:'column', gap:10, fontSize:'0.78rem' }}>
          <div style={{ display:'flex', flexDirection:'column', gap:3 }}>
            <span style={{ fontSize:'0.68rem', color:'rgba(255,255,255,0.35)', textTransform:'uppercase', fontWeight:600 }}>Embedding vector (PCA)</span>
            <div style={{ display:'flex', gap:4, fontFamily:'monospace', color:'#43aa8b', fontSize:'0.7rem' }}>
              [{activePt.meta.vector.map(v => parseFloat(v).toFixed(1)).join(', ')}]
            </div>
          </div>
          <div style={{ display:'flex', flexDirection:'column', gap:4 }}>
            <span style={{ fontSize:'0.68rem', color:'rgba(255,255,255,0.35)', textTransform:'uppercase', fontWeight:600 }}>Cosine Nearest Neighbors</span>
            <div style={{ display:'flex', flexDirection:'column', gap:4, marginTop:2 }}>
              {neighbors.map(n => (
                <div key={n.label} style={{ display:'flex', justifyContent:'space-between', padding:'4px 8px', background:'rgba(255,255,255,0.02)', borderRadius:4 }}>
                  <span>{n.label}</span>
                  <span style={{ color:'#90be6d', fontWeight:600 }}>{n.similarity.toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Distance ruler detail in panel */}
      {distanceToSel !== null && (
        <div style={{ marginTop:12, borderTop:'1px solid rgba(255,255,255,0.08)', paddingTop:10, fontSize:'0.78rem' }}>
          <div style={{ display:'flex', justifyContent:'space-between', color:'#10b981' }}>
            <span>Distance to <strong>{selectedPt.label}</strong>:</span>
            <strong>{distanceToSel.toFixed(2)}</strong>
          </div>
        </div>
      )}

      <div style={{ marginTop:14, borderTop:'1px solid rgba(255,255,255,0.08)', paddingTop:10, display:'flex', justifyContent:'space-between', gap:8 }}>
        {selectedPt && selectedPt.label === activePt.label ? (
          <button onClick={onClearSelection} style={{
            flex:1, background:'rgba(239, 68, 68, 0.15)', color:'#f87171', border:'none',
            padding:'6px 10px', borderRadius:6, fontSize:'0.75rem', cursor:'pointer'
          }}>Clear Ruler</button>
        ) : (
          <button onClick={() => onClickIndex(hoveredIdx !== null ? hoveredIdx : selectedIdx)} style={{
            flex:1, background:'rgba(16, 185, 129, 0.15)', color:'#34d399', border:'none',
            padding:'6px 10px', borderRadius:6, fontSize:'0.75rem', cursor:'pointer'
          }}>Set Reference</button>
        )}
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────
// Main export
// ──────────────────────────────────────────────
const btnStyle = (active, grad, shadow) => ({
  padding:'6px 14px', fontSize:'0.8rem', fontWeight:600,
  borderRadius:8, border:'none', cursor:'pointer', transition:'all 0.18s',
  background: active ? grad : 'rgba(255,255,255,0.06)',
  color: active ? '#fff' : 'rgba(255,255,255,0.5)',
  boxShadow: active ? shadow : 'none',
});

export default function DataVisualizer3D() {
  const [datasetKey, setDatasetKey]     = useState('mnist_pca');
  const [hoveredClass, setHoveredClass] = useState(null);
  const [autoRotate, setAutoRotate]     = useState(true);
  const [sparse, setSparse]             = useState(false);
  const [showShells, setShowShells]     = useState(false);

  // Reference selection states for the ruler
  const [selectedIdx, setSelectedIdx]   = useState(null);
  const [hoveredIdx, setHoveredIdx]     = useState(null);

  const allPoints = useMemo(() => generateDataset(datasetKey), [datasetKey]);
  const points    = useMemo(
    () => sparse ? allPoints.filter((_, i) => i % 3 === 0) : allPoints,
    [allPoints, sparse]
  );
  const spec = DATASETS[datasetKey];
  const onHoverClass = useCallback(c => setHoveredClass(c), []);

  // Reset local interactive point states when dataset changes
  useEffect(() => {
    setSelectedIdx(null);
    setHoveredIdx(null);
  }, [datasetKey, sparse]);

  // Compute bounding shells (centroids and variance radii) for each class
  const shells = useMemo(() => {
    const list = [];
    for (let c = 0; c < spec.numClasses; c++) {
      const classPoints = points.filter(p => p.classIdx === c);
      if (classPoints.length === 0) continue;

      let sumX = 0, sumY = 0, sumZ = 0;
      classPoints.forEach(p => { sumX += p.x; sumY += p.y; sumZ += p.z; });
      const centroid = [sumX / classPoints.length, sumY / classPoints.length, sumZ / classPoints.length];

      let sumDistSq = 0;
      classPoints.forEach(p => {
        sumDistSq += Math.pow(p.x - centroid[0], 2) + Math.pow(p.y - centroid[1], 2) + Math.pow(p.z - centroid[2], 2);
      });
      const avgDist = Math.sqrt(sumDistSq / classPoints.length);
      const radius = avgDist * 1.6;

      list.push({
        centroid,
        radius,
        color: CLASS_COLORS[c % CLASS_COLORS.length],
        classIdx: c
      });
    }
    return list;
  }, [points, spec]);

  const activePt = hoveredIdx !== null ? points[hoveredIdx] : (selectedIdx !== null ? points[selectedIdx] : null);
  const selectedPt = selectedIdx !== null ? points[selectedIdx] : null;

  const handleClearSelection = useCallback(() => setSelectedIdx(null), []);

  return (
    <div style={{ display:'flex', flexDirection:'column', height:'100%' }}>

      {/* ── Controls ── */}
      <div style={{
        display:'flex', flexWrap:'wrap', gap:14, alignItems:'center',
        padding:'14px 20px',
        background:'rgba(8,18,38,0.8)', backdropFilter:'blur(12px)',
        borderBottom:'1px solid rgba(255,255,255,0.05)',
        borderRadius:'12px 12px 0 0',
        flexShrink: 0,
      }}>
        {/* Dataset */}
        <div style={{ display:'flex', flexDirection:'column', gap:4 }}>
          <span style={{ fontSize:'0.62rem', fontWeight:700,
            color:'rgba(255,255,255,0.32)', letterSpacing:'0.09em', textTransform:'uppercase' }}>
            Dataset
          </span>
          <div style={{ display:'flex', gap:6 }}>
            {Object.entries(DATASETS).map(([key, ds]) => (
              <button key={key}
                onClick={() => { setDatasetKey(key); setHoveredClass(null); }}
                style={btnStyle(datasetKey===key,
                  'linear-gradient(135deg,#a78bfa,#7c3aed)','0 0 14px #7c3aed66')}>
                {ds.name.split('—')[0].trim()}
              </button>
            ))}
          </div>
        </div>

        <div style={{ width:1, height:36, background:'rgba(255,255,255,0.07)' }} />

        {/* View */}
        <div style={{ display:'flex', flexDirection:'column', gap:4 }}>
          <span style={{ fontSize:'0.62rem', fontWeight:700,
            color:'rgba(255,255,255,0.32)', letterSpacing:'0.09em', textTransform:'uppercase' }}>
            View
          </span>
          <button onClick={() => setAutoRotate(r => !r)}
            style={btnStyle(autoRotate,
              'linear-gradient(135deg,#43aa8b,#277154)','0 0 12px #43aa8b55')}>
            {autoRotate ? '⟳ Auto-Rotating' : '⊙ Manual'}
          </button>
        </div>

        <div style={{ width:1, height:36, background:'rgba(255,255,255,0.07)' }} />

        {/* Density */}
        <div style={{ display:'flex', flexDirection:'column', gap:4 }}>
          <span style={{ fontSize:'0.62rem', fontWeight:700,
            color:'rgba(255,255,255,0.32)', letterSpacing:'0.09em', textTransform:'uppercase' }}>
            Density
          </span>
          <div style={{ display:'flex', gap:6 }}>
            <button onClick={() => setSparse(false)}
              style={btnStyle(!sparse,
                'linear-gradient(135deg,#f8961e,#c05e00)','0 0 12px #f8961e44')}>
              All ({allPoints.length})
            </button>
            <button onClick={() => setSparse(true)}
              style={btnStyle(sparse,
                'linear-gradient(135deg,#f8961e,#c05e00)','0 0 12px #f8961e44')}>
              Sparse ({Math.ceil(allPoints.length/3)})
            </button>
          </div>
        </div>

        <div style={{ width:1, height:36, background:'rgba(255,255,255,0.07)' }} />

        {/* Bounding Spheres */}
        <div style={{ display:'flex', flexDirection:'column', gap:4 }}>
          <span style={{ fontSize:'0.62rem', fontWeight:700,
            color:'rgba(255,255,255,0.32)', letterSpacing:'0.09em', textTransform:'uppercase' }}>
            Shells
          </span>
          <button onClick={() => setShowShells(s => !s)}
            style={btnStyle(showShells,
              'linear-gradient(135deg,#577590,#3b4f63)','0 0 12px #57759055')}>
            {showShells ? '⚡ Bounding Spheres On' : '⚡ Show Shells'}
          </button>
        </div>

        {/* Info */}
        <div style={{ marginLeft:'auto', maxWidth:300 }}>
          <div style={{ fontSize:'0.82rem', fontWeight:700, color:'#e2e8f0', marginBottom:2 }}>
            {spec.name}
          </div>
          <div style={{ fontSize:'0.69rem', color:'rgba(255,255,255,0.35)', lineHeight:1.5 }}>
            {spec.description}
          </div>
        </div>
      </div>

      {/* ── Canvas — fills remaining height ── */}
      <div style={{
        position:'relative',
        flex:1,
        minHeight:0,
        background:'#04080f',
        borderRadius:'0 0 12px 12px',
        overflow:'hidden',
      }}>
        <CanvasErrorBoundary>
          <Canvas
            camera={{ position:[8, 5, 8], fov:50 }}
            gl={{ antialias:true, alpha:false, powerPreference:'high-performance' }}
            style={{ position:'absolute', inset:0 }}
            onPointerMissed={handleClearSelection}
          >
            <Suspense fallback={null}>
              <Scene
                points={points}
                hoveredClass={hoveredClass}
                selectedIdx={selectedIdx}
                hoveredIdx={hoveredIdx}
                onHoverIndex={setHoveredIdx}
                onClickIndex={setSelectedIdx}
                showShells={showShells}
                shells={shells}
                autoRotate={autoRotate}
              />
            </Suspense>
          </Canvas>
        </CanvasErrorBoundary>

        <InspectPanel
          activePt={activePt}
          allPoints={points}
          datasetKey={datasetKey}
          selectedPt={selectedPt}
          onClearSelection={handleClearSelection}
          onClickIndex={setSelectedIdx}
          selectedIdx={selectedIdx}
          hoveredIdx={hoveredIdx}
        />

        <LegendOverlay datasetKey={datasetKey} hoveredClass={hoveredClass} onHoverClass={onHoverClass} />

        <div style={{
          position:'absolute', bottom:10, left:'50%', transform:'translateX(-50%)',
          background:'rgba(0,0,0,0.45)', backdropFilter:'blur(6px)',
          borderRadius:20, padding:'4px 14px',
          fontSize:'0.7rem', color:'rgba(255,255,255,0.3)',
          pointerEvents:'none', whiteSpace:'nowrap', zIndex:5
        }}>
          🖱 Drag to rotate · Scroll to zoom · Click point to select ruler reference · Hover legend to highlight clusters
        </div>
      </div>
    </div>
  );
}
