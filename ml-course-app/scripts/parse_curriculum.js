import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const markdownPath = path.resolve(__dirname, '../../ML_AI_Full_42Week_Curriculum.md');
const outputPath = path.resolve(__dirname, '../src/data/curriculum.json');

const markdownContent = fs.readFileSync(markdownPath, 'utf8');
const lines = markdownContent.split('\n');

const curriculum = {
  blocks: []
};

let currentBlock = null;
let currentWeek = null;
let currentSection = null;

for (let i = 0; i < lines.length; i++) {
  const line = lines[i].trim();

  // New Block
  if (line.startsWith('# PART ')) {
    currentBlock = {
      title: line.substring(2).trim(),
      weeks: []
    };
    curriculum.blocks.push(currentBlock);
    continue;
  }

  // New Week
  if (line.startsWith('## Week ')) {
    currentWeek = {
      title: line.substring(3).trim(),
      goal: '',
      tasks: []
    };
    if (currentBlock) {
      currentBlock.weeks.push(currentWeek);
    }
    currentSection = null;
    continue;
  }

  if (!currentWeek) continue;

  // Goal
  if (line.startsWith('**Goal:**')) {
    currentWeek.goal = line.replace('**Goal:**', '').trim();
    continue;
  }

  // Sections
  const sectionMatch = line.match(/^\*\*(Read|Watch|Derive by hand|Code|Deliverable:|Self-test:|Build|Understand|Watch \/ read|Read \/ watch)\*\*/i);
  if (sectionMatch) {
    let type = sectionMatch[1].replace(':', '').trim();
    currentSection = {
      type: type,
      items: []
    };
    currentWeek.tasks.push(currentSection);
    
    // Check if the section text is on the same line (like Deliverable: or Self-test:)
    const content = line.replace(sectionMatch[0], '').trim();
    if (content) {
      currentSection.items.push(content);
    }
    continue;
  }

  // Tasks inside section
  if (currentSection && line) {
    if (line.startsWith('- ')) {
      currentSection.items.push(line.substring(2).trim());
    } else if (!line.startsWith('---') && !line.startsWith('>')) {
      // Sometimes it's a paragraph
      if (currentSection.items.length > 0) {
        currentSection.items[currentSection.items.length - 1] += ' ' + line;
      } else {
        currentSection.items.push(line);
      }
    }
  }
}

fs.writeFileSync(outputPath, JSON.stringify(curriculum, null, 2));
console.log(`Curriculum successfully parsed into ${outputPath}`);
