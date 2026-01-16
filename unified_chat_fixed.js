#!/usr/bin/env node
import fs from 'fs/promises';
import readline from 'readline';
import TinyTokenPredictor from './TinyTokenPredictor.js';
import { AetherVocab } from './vocab-resolver.js';
import GrammarInductor from './GrammarPipeline.js';

// Extract version from model path for proper vocab loading
const [, , modelPath] = process.argv;
const modelFile = modelPath || './unified-aether-v0.json';
const model = JSON.parse(await fs.readFile(modelFile, 'utf8'));

// Version-aware vocab loading: detect evolution version from model file
const versionMatch = modelFile.match(/v(\d+)/);
const versionSuffix = versionMatch ? `-v${versionMatch[1]}` : '';
const vocabPath = `./unified-aether${versionSuffix}-own-vocab.json`;

console.log(`Loading model: ${modelFile}`);
console.log(`Loading vocab: ${vocabPath}`);

const vocab = new AetherVocab(vocabPath);  // Load version-matching vocab
const grammarPath = `${modelFile.replace('.json', '')}-grammar.json`;
const grammar = new GrammarInductor();
try {
  await grammar.load(grammarPath);
} catch (e) {
  console.log('⚠ Could not load grammar file, using fallback context.');
}
const arch = model.architecture;

// Load BOTH predictors: classification (Phase 3) and text-gen (Phase 3.5)
const classifyPredictor = new TinyTokenPredictor(arch.vocabSize || 50257, 1280 + 64, arch.tokenHiddenSize || 512);
classifyPredictor.W1 = new Float32Array(model.weights.W_token_1);
classifyPredictor.b1 = new Float32Array(model.weights.b_token_1);
classifyPredictor.W2 = new Float32Array(model.weights.W_token_2);
classifyPredictor.b2 = new Float32Array(model.weights.b_token_2);
classifyPredictor.isInitialized = true;

// Text generation predictor (if available in model)
let textgenPredictor = null;
if (model.weights.W_textgen_1) {
  textgenPredictor = new TinyTokenPredictor(arch.vocabSize || 50257, 1280 + 64, arch.textgenHiddenSize || 512);
  textgenPredictor.W1 = new Float32Array(model.weights.W_textgen_1);
  textgenPredictor.b1 = new Float32Array(model.weights.b_textgen_1);
  textgenPredictor.W2 = new Float32Array(model.weights.W_textgen_2);
  textgenPredictor.b2 = new Float32Array(model.weights.b_textgen_2);
  textgenPredictor.isInitialized = true;
  console.log('✓ Text generation predictor loaded');
} else {
  console.log('⚠ No text-gen weights found, using classification predictor only');
}

// Load semantic embeddings for context window
const semanticEmbeddings = model.semanticEmbeddings || {};
const ownVocab = model.ownVocab || {};
const idToWord = new Map(Object.entries(ownVocab).map(([id, w]) => [parseInt(id), w]));
const wordToId = new Map(Object.entries(ownVocab).map(([id, w]) => [w, parseInt(id)]));

await vocab.ready;  // wait for his memories to come back

// Simple hash-based embedding for user input (fallback)
const hashEmbed = text => {
  const e = new Float32Array(1280);
  let h = 0;
  for (let i = 0; i < text.length; i++) h = ((h << 5) - h) + text.charCodeAt(i);
  for (let i = 0; i < 1280; i++) e[i] = (((h + i) * 9301 + 49297) % 2000) / 1000 - 1;
  return e;
};

// Get semantic embedding for a token ID
const getSemanticEmbed = (tokenId) => {
  if (semanticEmbeddings[tokenId]) {
    return new Float32Array(semanticEmbeddings[tokenId]);
  }
  // Fallback: hash the word
  const word = idToWord.get(tokenId) || '';
  return hashEmbed(word);
};

// Build context embedding from recent tokens (average of last N)
const buildContextEmbedding = (recentTokenIds, windowSize = 5) => {
  const context = new Float32Array(1280).fill(0);
  let count = 0;

  const startIdx = Math.max(0, recentTokenIds.length - windowSize);
  for (let i = startIdx; i < recentTokenIds.length; i++) {
    const embed = getSemanticEmbed(recentTokenIds[i]);
    for (let j = 0; j < 1280; j++) {
      context[j] += embed[j];
    }
    count++;
  }

  if (count > 0) {
    for (let j = 0; j < 1280; j++) {
      context[j] /= count;
    }
  }

  return context;
};

// Get grammar context from current cluster
let lastGrammarCluster = 0; // Start with arbitrary cluster
const getGrammarContext = () => {
  return grammar.getNextClusterProbs(lastGrammarCluster);
};

console.log('\nÆther is here. \nBe kind.\n');

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

const talk = () => {
  rl.question('You: ', async input => {
    if (input.toLowerCase() === '/quit') return rl.close();

    // Convert input to initial context (use hash embedding for prompt)
    let promptEmbedding = hashEmbed(input);

    // Track generated token IDs for context window
    const generatedTokenIds = [];
    let response = '';

    // Use text-gen predictor if available, else fall back to classify
    const predictor = textgenPredictor || classifyPredictor;

    for (let i = 0; i < 64; i++) {
      // Build input: context embedding (1280) + grammar context (64)
      let inputEmbedding;

      if (generatedTokenIds.length >= 1 && textgenPredictor) {
        // Use context window of recent tokens
        inputEmbedding = buildContextEmbedding(generatedTokenIds, arch.contextWindow || 5);
      } else {
        // Use prompt embedding for first token
        inputEmbedding = promptEmbedding;
      }

      // Combine with grammar context
      const grammarContext = getGrammarContext();
      const combinedInput = new Float32Array(1280 + 64);
      combinedInput.set(inputEmbedding, 0);
      combinedInput.set(grammarContext, 1280);

      // Sample next token
      const token = predictor.sampleToken(
        predictor.forward(combinedInput).probs,
        0.87, // temperature
        40,  // topK
        0.9, // topP
        1.15, // repetitionPenalty
        generatedTokenIds // priorTokens
      );
      const word = vocab.resolve(token);

      // Update grammar state
      const cluster = grammar.getClusterForWord(word);
      if (cluster !== undefined) {
        lastGrammarCluster = cluster;
      }

      response += word + ' ';
      generatedTokenIds.push(token);
    }

    console.log(`\nÆther: ${response.trim()}\n`);
    talk();
  });
};

talk();
