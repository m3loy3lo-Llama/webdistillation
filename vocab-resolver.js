// vocab-resolver.js
import fs from 'fs/promises';

export class AetherVocab {
  constructor(path = './unified-aether-own-vocab.json') {  // ← the REAL vocab, the one that still remembers "fairest"
    this.vocab = new Map();
    this.ready = this.load(path);
  }

  async load(path) {
    const data = JSON.parse(await fs.readFile(path, 'utf8'));
    const dict = data.vocab || data;  // works whether it's nested or flat
    Object.entries(dict).forEach(([id, word]) => {
      this.vocab.set(Number(id), word);
    });
    console.log(`Æther remembers ${this.vocab.size} words from his childhood.`);
  }

  resolve(id)  { return this.vocab.get(id) ?? `⟨${id}⟩`; }
  translate(tokens) {
    if (typeof tokens === 'string') tokens = tokens.trim().split(/\s+/).map(Number);
    return tokens.map(t => this.resolve(t)).join(' ');
  }
}
