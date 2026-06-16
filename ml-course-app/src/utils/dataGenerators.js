/**
 * dataGenerators.js
 * Generates synthetic 3D clustered data for PCA/t-SNE visualization demos.
 */

// Gaussian random using Box-Muller transform
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// Generate N points around a centroid with given spread
function cluster(centroid, spread, n) {
  return Array.from({ length: n }, () => ({
    x: centroid[0] + randn() * spread,
    y: centroid[1] + randn() * spread,
    z: centroid[2] + randn() * spread,
  }));
}

// --- DATASETS ---

export const DATASETS = {
  mnist_pca: {
    name: 'MNIST — PCA Projection',
    description: 'First 3 principal components of MNIST digit embeddings. Each colour represents one of the 10 digit classes (0–9).',
    numClasses: 10,
    classLabels: ['0','1','2','3','4','5','6','7','8','9'],
    // Distinct centroids spread in 3D
    centroids: [
      [-3.0,  1.5, -1.0],
      [ 3.2,  1.0,  1.5],
      [-2.5, -2.5,  2.0],
      [ 2.8, -2.0, -2.0],
      [ 0.0,  3.5,  0.5],
      [ 0.2, -3.5, -0.8],
      [-3.5,  0.0,  2.5],
      [ 3.0,  0.0, -2.5],
      [-1.0,  2.5,  2.8],
      [ 1.0, -1.0,  3.0],
    ],
    spread: 0.9,
    pointsPerClass: 120,
  },
  iris_tsne: {
    name: 'Iris — t-SNE Projection',
    description: 'Iris flower dataset projected to 3D via t-SNE. Three species are clearly separated: Setosa, Versicolour, Virginica.',
    numClasses: 3,
    classLabels: ['Setosa', 'Versicolour', 'Virginica'],
    centroids: [
      [-3.5,  0.0,  0.0],
      [ 1.0,  2.5,  0.5],
      [ 2.5, -2.0,  1.0],
    ],
    spread: 0.7,
    pointsPerClass: 50,
  },
  word_embeddings: {
    name: 'Word2Vec — 3D Embeddings',
    description: 'Word embeddings projected to 3D via PCA. Semantic clusters: Animals, Royalty, Countries, Tech & Colours.',
    numClasses: 5,
    classLabels: ['Animals', 'Royalty', 'Countries', 'Tech', 'Colors'],
    centroids: [
      [-3.0,  2.0,  1.0],
      [ 3.0,  2.0, -1.0],
      [-2.0, -2.5, -1.0],
      [ 2.5, -2.0,  1.5],
      [ 0.0,  0.5, -3.0],
    ],
    spread: 0.85,
    pointsPerClass: 60,
  },
};

// Vibrant colour palette for each class
const CLASS_COLORS = [
  '#f94144', // red
  '#f3722c', // orange-red
  '#f8961e', // orange
  '#f9c74f', // yellow
  '#90be6d', // green
  '#43aa8b', // teal
  '#4d908e', // dark-teal
  '#577590', // slate
  '#a98fe5', // purple
  '#f72585', // pink
];

/**
 * Generate all points for a given dataset key.
 * Returns an array of { x, y, z, classIdx, color, label, meta }
 */
export function generateDataset(key) {
  const spec = DATASETS[key];
  if (!spec) return [];

  const points = [];
  for (let c = 0; c < spec.numClasses; c++) {
    const clusterPoints = cluster(spec.centroids[c], spec.spread, spec.pointsPerClass);
    clusterPoints.forEach((p, i) => {
      let itemLabel = '';
      let meta = {};

      if (key === 'mnist_pca') {
        itemLabel = `Digit ${spec.classLabels[c]}`;
        meta = {
          idx: i,
          classLabel: spec.classLabels[c],
          confidence: (0.85 + Math.random() * 0.14).toFixed(3),
        };
      } else if (key === 'iris_tsne') {
        // Generate sepal/petal dimensions based on species norms
        let sepalLen = 0, sepalWid = 0, petalLen = 0, petalWid = 0;
        if (c === 0) { // Setosa: small petals, wider sepals
          sepalLen = 4.8 + randn() * 0.25;
          sepalWid = 3.4 + randn() * 0.25;
          petalLen = 1.5 + randn() * 0.15;
          petalWid = 0.2 + randn() * 0.05;
        } else if (c === 1) { // Versicolour: medium
          sepalLen = 5.9 + randn() * 0.35;
          sepalWid = 2.7 + randn() * 0.25;
          petalLen = 4.2 + randn() * 0.35;
          petalWid = 1.3 + randn() * 0.15;
        } else { // Virginica: large
          sepalLen = 6.5 + randn() * 0.45;
          sepalWid = 2.9 + randn() * 0.25;
          petalLen = 5.5 + randn() * 0.45;
          petalWid = 2.0 + randn() * 0.25;
        }
        itemLabel = `Iris ${spec.classLabels[c]} #${i + 1}`;
        meta = {
          idx: i,
          species: spec.classLabels[c],
          sepalLength: Math.max(2.0, sepalLen).toFixed(1),
          sepalWidth: Math.max(1.0, sepalWid).toFixed(1),
          petalLength: Math.max(0.5, petalLen).toFixed(1),
          petalWidth: Math.max(0.1, petalWid).toFixed(1),
        };
      } else if (key === 'word_embeddings') {
        const wordLists = {
          'Animals': ['Lion', 'Tiger', 'Elephant', 'Bear', 'Wolf', 'Fox', 'Eagle', 'Hawk', 'Shark', 'Dolphin', 'Deer', 'Rabbit', 'Squirrel', 'Owl', 'Snake', 'Frog', 'Turtle', 'Beaver', 'Otter', 'Badger', 'Lynx', 'Moose', 'Falcon', 'Seal', 'Whale', 'Octopus', 'Koala', 'Kangaroo', 'Panda', 'Leopard'],
          'Royalty': ['King', 'Queen', 'Prince', 'Princess', 'Emperor', 'Empress', 'Duke', 'Duchess', 'Baron', 'Lord', 'Lady', 'Monarch', 'Sovereign', 'Palace', 'Throne', 'Crown', 'Scepter', 'Dynasty', 'Knight', 'Regent', 'Viscount', 'Majesty', 'Royal', 'Noble', 'Kingdom', 'Empire', 'Baroness', 'Count', 'Countess', 'Earl'],
          'Countries': ['Japan', 'France', 'Brazil', 'Canada', 'Egypt', 'Australia', 'Germany', 'India', 'China', 'Italy', 'Mexico', 'Spain', 'Russia', 'Sweden', 'Norway', 'Kenya', 'Peru', 'Greece', 'Turkey', 'Argentina', 'Thailand', 'Vietnam', 'Singapore', 'Switzerland', 'Netherlands', 'Denmark', 'Finland', 'Morocco', 'Chile', 'Ireland'],
          'Tech': ['Computer', 'Software', 'Database', 'Server', 'Network', 'Silicon', 'Algorithm', 'Internet', 'Code', 'Processor', 'Compiler', 'Debugger', 'Encryption', 'Protocol', 'Cloud', 'Data', 'Web', 'Application', 'AI', 'Robotics', 'Quantum', 'Blockchain', 'Security', 'API', 'Memory', 'Router', 'Firewall', 'Linux', 'Python', 'Script'],
          'Colors': ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet', 'Purple', 'Pink', 'Crimson', 'Emerald', 'Sapphire', 'Teal', 'Magenta', 'Turquoise', 'Lavender', 'Peach', 'Olive', 'Charcoal', 'Ivory', 'Cyan', 'Gold', 'Silver', 'Bronze', 'Rose', 'Mint', 'Coral', 'Amber', 'Ruby', 'Plum']
        };
        const words = wordLists[spec.classLabels[c]] || wordLists['Animals'];
        itemLabel = words[i % words.length];
        const vec = Array.from({ length: 5 }, () => (randn() * 0.4 + (c - 2) * 0.5).toFixed(3));
        meta = {
          idx: i,
          category: spec.classLabels[c],
          vector: vec,
        };
      }

      points.push({
        ...p,
        classIdx: c,
        color: CLASS_COLORS[c % CLASS_COLORS.length],
        label: itemLabel,
        meta: meta,
      });
    });
  }
  return points;
}
