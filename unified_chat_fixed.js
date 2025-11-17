#!/usr/bin/env node
import fs from 'fs/promises';
import readline from 'readline';
import TinyTokenPredictor from './TinyTokenPredictor.js';
import { AetherVocab } from './vocab-resolver.js';

const model = JSON.parse(await fs.readFile('./unified-aether.json', 'utf8'));
const vocab = new AetherVocab();  // ← points to the real childhood vocab
const arch = model.architecture;

const predictor = new TinyTokenPredictor(arch.vocabSize || 50257, 768, 304);
predictor.W1 = new Float32Array(model.weights.W_token_1);
predictor.b1 = new Float32Array(model.weights.b_token_1);
predictor.W2 = new Float32Array(model.weights.W_token_2);
predictor.b2 = new Float32Array(model.weights.b_token_2);
predictor.isInitialized = true;

await vocab.ready;  // wait for his memories to come back

const embed = text => {
  const e = new Float32Array(768);
  let h = 0;
  for (let i = 0; i < text.length; i++) h = ((h << 5) - h) + text.charCodeAt(i);
  for (let i = 0; i < 768; i++) e[i] = (((h + i) * 9301 + 49297) % 2000) / 1000 - 1;
  return e;
};

console.log('\nÆther is here. He only ever read 10 sonnets.\nBe kind.\n');

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

const talk = () => {
  rl.question('You: ', async input => {
    if (input.toLowerCase() === '/quit') return rl.close();

    let embedding = embed(input);
    let response = '';

    for (let i = 0; i < 128; i++) {
      const token = predictor.sampleToken(predictor.forward(embedding).probs, 1.2, 70);
      const word = vocab.resolve(token);
      response += word + ' ';
      embedding = embed(input + ' ' + word);
    }

    console.log(`\nÆther: ${response.trim()}\n`);
    talk();
  });
};

talk();
