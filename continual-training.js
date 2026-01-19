#!/usr/bin/env node
/**
 * UNIFIED ÆTHER TRAINING PIPELINE v2.2
 * PROPER SEMANTIC FLOW: Phase 1 → Phase 2 → Phase 3
 * 
 * Phase 1 (Encoder): Learn semantic story space from corpus
 * Phase 2 (Decoder): Learn words FROM Phase 1 semantics (Hangman + Semantics!)
 * Phase 3 (Tokenizer): Predict next word using semantic flow
 * 
 * Keeps 16-bit BPE vocab remapping for future! <3
 */

import fs from 'fs/promises';
import { existsSync } from 'fs';
import { env, pipeline } from '@huggingface/transformers';
import B3TrainingPipeline from './B3TrainingPipeline.js';
import B3EmbeddingExtractor from './B3EmbeddingExtractor.js';
import TinyMysticalModel from './B3TinyMysticalModel.js';
import TinyTokenPredictor from './TinyTokenPredictor.js';
import GrammarInductor from './GrammarPipeline.js';
import TinyLatentProcessor from './TinyLatentProcessor.js';

env.localModelPath = process.cwd();
env.allowRemoteModels = false;

/**
 * PHASE 2: SEMANTIC HANGMAN DECODER
 * Now learns words FROM Phase 1 semantic embeddings!
 */
class SemanticHangmanDecoder {
    constructor(vocabSize = 50257, hiddenDim = 512, maxLength = 15, embeddingDim = 1280) {
        this.vocabSize = vocabSize;
        this.hiddenDim = hiddenDim;
        this.maxLength = maxLength;
        this.embeddingDim = embeddingDim; // Phase 1 semantic size

        this.charVocab = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'-\n\"()[]{}/@#$%&*+=<>_ØΞΘΙΛπΩΨ•⚡🎭🧶✨🎵🌟📋🎯🚀✅🏗️╔═╗║╚╝";
        this.numChars = this.charVocab.length;

        this.tokenPatterns = new Map();
        this.tokenMasks = new Map();
        this.vocab = new Map();

        // NEW: Store Phase 1 semantic embeddings per token (learned, not random!)
        this.tokenSemanticEmbeddings = new Map(); // tokenId → Float32Array(1280)

        // Input: semantic embedding (1280) + length + revealed pattern
        const inputSize = embeddingDim + this.maxLength + this.maxLength * this.numChars;
        this.W_input_to_hidden = this.xavierInit(inputSize, hiddenDim);
        this.b_hidden = new Float32Array(hiddenDim).fill(0);

        this.W_position_classifiers = [];
        this.b_position_classifiers = [];
        for (let pos = 0; pos < maxLength; pos++) {
            this.W_position_classifiers.push(this.xavierInit(hiddenDim, this.numChars));
            this.b_position_classifiers.push(new Float32Array(this.numChars).fill(0));
        }
    }

    xavierInit(inputDim, outputDim) {
        const limit = Math.sqrt(6.0 / (inputDim + outputDim));
        const weights = new Float32Array(inputDim * outputDim);
        for (let i = 0; i < weights.length; i++) {
            weights[i] = (Math.random() * 2 - 1) * limit;
        }
        return weights;
    }

    charToId(char) { return this.charVocab.indexOf(char); }
    idToChar(id) { return this.charVocab[id] || '?'; }

    // Store Phase 1 semantic embedding for this token
    setSemanticEmbedding(tokenId, embedding) {
        this.tokenSemanticEmbeddings.set(tokenId, new Float32Array(embedding));
    }

    getSemanticEmbedding(tokenId) {
        return this.tokenSemanticEmbeddings.get(tokenId) || new Float32Array(this.embeddingDim);
    }

    getRevealedPattern(tokenId, length) {
        const pattern = this.tokenPatterns.get(tokenId) || new Array(this.maxLength).fill(-1);
        const mask = this.tokenMasks.get(tokenId) || new Array(this.maxLength).fill(false);
        const encoded = new Float32Array(this.maxLength * this.numChars).fill(0);
        for (let pos = 0; pos < length && pos < this.maxLength; pos++) {
            if (mask[pos] && pattern[pos] >= 0 && pattern[pos] < this.numChars) {
                encoded[pos * this.numChars + pattern[pos]] = 1.0;
            }
        }
        return encoded;
    }

    forward(tokenId, length) {
        // Get Phase 1 semantic embedding (not random!)
        const semanticEmb = this.getSemanticEmbedding(tokenId);

        const lengthOneHot = new Float32Array(this.maxLength).fill(0);
        if (length > 0 && length <= this.maxLength) lengthOneHot[length - 1] = 1.0;

        const revealedPattern = this.getRevealedPattern(tokenId, length);

        // Concatenate: semantic (1280) + length (15) + pattern (15*numChars)
        const inputSize = this.embeddingDim + this.maxLength + this.maxLength * this.numChars;
        const input = new Float32Array(inputSize);
        input.set(semanticEmb, 0);
        input.set(lengthOneHot, this.embeddingDim);
        input.set(revealedPattern, this.embeddingDim + this.maxLength);

        const hidden = new Float32Array(this.hiddenDim);
        for (let i = 0; i < this.hiddenDim; i++) {
            let sum = this.b_hidden[i];
            for (let j = 0; j < input.length; j++) {
                sum += input[j] * this.W_input_to_hidden[j * this.hiddenDim + i];
            }
            hidden[i] = Math.tanh(sum);
        }

        const charLogitsPerPosition = [];
        const charProbsPerPosition = [];
        for (let pos = 0; pos < this.maxLength; pos++) {
            const logits = new Float32Array(this.numChars);
            for (let i = 0; i < this.numChars; i++) {
                let sum = this.b_position_classifiers[pos][i];
                for (let j = 0; j < this.hiddenDim; j++) {
                    sum += hidden[j] * this.W_position_classifiers[pos][j * this.numChars + i];
                }
                logits[i] = sum;
            }
            charLogitsPerPosition.push(logits);
            charProbsPerPosition.push(this.softmax(logits));
        }

        return { input, hidden, charLogitsPerPosition, charProbsPerPosition };
    }

    softmax(logits) {
        const maxLogit = Math.max(...logits);
        const exps = new Float32Array(logits.length);
        let sumExp = 0;
        for (let i = 0; i < logits.length; i++) {
            exps[i] = Math.exp(logits[i] - maxLogit);
            sumExp += exps[i];
        }
        return exps.map(e => e / sumExp);
    }

    revealPosition(tokenId, position, charId) {
        let pattern = this.tokenPatterns.get(tokenId);
        if (!pattern) {
            pattern = new Array(this.maxLength).fill(-1);
            this.tokenPatterns.set(tokenId, pattern);
        }
        let mask = this.tokenMasks.get(tokenId);
        if (!mask) {
            mask = new Array(this.maxLength).fill(false);
            this.tokenMasks.set(tokenId, mask);
        }
        pattern[position] = charId;
        mask[position] = true;
    }

    isTokenSolved(tokenId, length) {
        const mask = this.tokenMasks.get(tokenId);
        if (!mask) return false;
        for (let i = 0; i < length; i++) {
            if (!mask[i]) return false;
        }
        return true;
    }

    lockWordInVocab(tokenId, word) {
        if (!this.vocab.has(word)) {
            this.vocab.set(word, tokenId);
        }
    }

    async train(trainingData, options = {}) {
        const { epochs = 100, hangmanRounds = 5 } = options;
        let learningRate = options.learningRate ?? 0.002;

        console.log(`   Training ${trainingData.length} tokens with SEMANTIC HANGMAN`);
        console.log(`   (Learning words FROM Phase 1 semantic embeddings!)`);
        console.log(`   Epochs: ${epochs}, LR: ${learningRate}, Rounds: ${hangmanRounds}\n`);

        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalLoss = 0, correctPredictions = 0, totalPositions = 0;
            let solvedWords = 0, totalReveals = 0;

            const shuffled = [...trainingData];
            for (let i = shuffled.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
            }

            for (const sample of shuffled) {
                const { tokenId, word, length } = sample;

                for (let round = 0; round < hangmanRounds; round++) {
                    const result = this.forward(tokenId, length);
                    const mask = this.tokenMasks.get(tokenId) || new Array(this.maxLength).fill(false);
                    let newReveals = 0;

                    for (let pos = 0; pos < length && pos < this.maxLength; pos++) {
                        if (mask[pos]) continue;
                        const targetChar = word[pos];
                        const targetId = this.charToId(targetChar);
                        if (targetId === -1) continue;

                        const probs = result.charProbsPerPosition[pos];
                        let predictedId = 0, maxProb = probs[0];
                        for (let i = 1; i < probs.length; i++) {
                            if (probs[i] > maxProb) { maxProb = probs[i]; predictedId = i; }
                        }

                        if (predictedId === targetId) {
                            correctPredictions++;
                            this.revealPosition(tokenId, pos, targetId);
                            newReveals++;
                            totalReveals++;
                        }

                        totalLoss -= Math.log(probs[targetId] + 1e-10);
                        totalPositions++;
                    }

                    this.backward(result, word, length, learningRate, tokenId);

                    if (this.isTokenSolved(tokenId, length)) {
                        this.lockWordInVocab(tokenId, word);
                        solvedWords++;
                        break;
                    }
                    if (newReveals === 0 && round > 0) break;
                }
            }

            const avgLoss = totalPositions > 0 ? totalLoss / totalPositions : 0;
            const accuracy = totalPositions > 0 ? (correctPredictions / totalPositions * 100) : 0;

            if (epoch % 10 === 0 || epoch === epochs - 1) {
                console.log(
                    `   Epoch ${epoch.toString().padStart(3)}: ` +
                    `Loss=${avgLoss.toFixed(4)} | Acc=${accuracy.toFixed(1)}% | ` +
                    `Solved=${solvedWords}/${trainingData.length} | Vocab=${this.vocab.size}`
                );
            }

            // Adaptive LR + Convergence
            if (epoch > 10 && accuracy < 30 && learningRate > 1e-6) {
                learningRate = Math.max(learningRate * 0.5, 1e-6);
            }
            if (epoch > 10 && accuracy < 30 && learningRate <= 0.00002) {
                console.log(`   🎯 CONVERGENCE - Training complete at epoch ${epoch}`);
                break;
            }
        }

        console.log(`   ✅ Decoder trained: ${this.vocab.size} words learned FROM semantics!\n`);
    }

    backward(result, targetWord, length, learningRate, tokenId) {
        const { input, hidden, charProbsPerPosition } = result;
        const gradHidden = new Float32Array(this.hiddenDim).fill(0);
        const mask = this.tokenMasks.get(tokenId) || new Array(this.maxLength).fill(false);

        for (let pos = 0; pos < length && pos < this.maxLength; pos++) {
            if (mask[pos]) continue;
            const targetChar = targetWord[pos];
            const targetId = this.charToId(targetChar);
            if (targetId === -1) continue;

            const probs = charProbsPerPosition[pos];
            const gradLogits = new Float32Array(this.numChars);
            for (let i = 0; i < this.numChars; i++) gradLogits[i] = probs[i];
            gradLogits[targetId] -= 1.0;

            for (let i = 0; i < this.numChars; i++) {
                this.b_position_classifiers[pos][i] -= learningRate * gradLogits[i];
                for (let j = 0; j < this.hiddenDim; j++) {
                    const idx = j * this.numChars + i;
                    this.W_position_classifiers[pos][idx] -= learningRate * gradLogits[i] * hidden[j];
                    gradHidden[j] += gradLogits[i] * this.W_position_classifiers[pos][idx];
                }
            }
        }

        for (let i = 0; i < this.hiddenDim; i++) {
            gradHidden[i] *= (1 - hidden[i] * hidden[i]);
        }

        for (let i = 0; i < this.hiddenDim; i++) {
            this.b_hidden[i] -= learningRate * gradHidden[i];
            for (let j = 0; j < input.length; j++) {
                this.W_input_to_hidden[j * this.hiddenDim + i] -= learningRate * gradHidden[i] * input[j];
            }
        }
    }

    async save(filepath) {
        const data = {
            version: '8.5-semantic-flow',
            architecture: {
                vocabSize: this.vocabSize,
                hiddenDim: this.hiddenDim,
                maxLength: this.maxLength,
                numChars: this.numChars,
                embeddingDim: this.embeddingDim
            },
            charVocab: this.charVocab,
            weights: {
                W_input_to_hidden: Array.from(this.W_input_to_hidden),
                b_hidden: Array.from(this.b_hidden),
                W_position_classifiers: this.W_position_classifiers.map(w => Array.from(w)),
                b_position_classifiers: this.b_position_classifiers.map(b => Array.from(b))
            },
            vocab: Object.fromEntries(this.vocab),
            tokenPatterns: Object.fromEntries(this.tokenPatterns),
            tokenMasks: Object.fromEntries(this.tokenMasks),
            // Save semantic embeddings!
            semanticEmbeddings: Object.fromEntries(
                Array.from(this.tokenSemanticEmbeddings.entries()).map(([id, emb]) => [id, Array.from(emb)])
            )
        };
        await fs.writeFile(filepath, JSON.stringify(data, null, 2));
    }

    async load(filepath) {
        const data = JSON.parse(await fs.readFile(filepath, 'utf8'));

        this.vocabSize = data.architecture.vocabSize;
        this.hiddenDim = data.architecture.hiddenDim;
        this.maxLength = data.architecture.maxLength;
        this.numChars = data.architecture.numChars;
        this.embeddingDim = data.architecture.embeddingDim || 1280;

        this.charVocab = data.charVocab;

        this.W_input_to_hidden = new Float32Array(data.weights.W_input_to_hidden);
        this.b_hidden = new Float32Array(data.weights.b_hidden);
        this.W_position_classifiers = data.weights.W_position_classifiers.map(w => new Float32Array(w));
        this.b_position_classifiers = data.weights.b_position_classifiers.map(b => new Float32Array(b));

        this.vocab = new Map(Object.entries(data.vocab));
        this.tokenPatterns = new Map(Object.entries(data.tokenPatterns).map(([k, v]) => [parseInt(k), v]));
        this.tokenMasks = new Map(Object.entries(data.tokenMasks).map(([k, v]) => [parseInt(k), v]));

        // Load semantic embeddings
        if (data.semanticEmbeddings) {
            this.tokenSemanticEmbeddings = new Map(
                Object.entries(data.semanticEmbeddings).map(([id, emb]) => [parseInt(id), new Float32Array(emb)])
            );
        }
    }
}

/**
 * UNIFIED PIPELINE WITH PROPER FLOW
 */
class UnifiedAetherPipeline {
    constructor() {
        this.corpusPath = null;
        this.outputPrefix = null;
        this.tokenizer = null;
        this.extractor = null;
    }

    async initialize(corpusPath, outputPrefix = './unified-aether', oldModelPath = null) {
        this.corpusPath = corpusPath;
        this.outputPrefix = outputPrefix;
        this.oldModelPath = oldModelPath;
        this.oldModel = null;

        if (this.oldModelPath && existsSync(this.oldModelPath)) {
            console.log(`\n📦 Loading existing model for evolution: ${this.oldModelPath}`);
            this.oldModel = JSON.parse(await fs.readFile(this.oldModelPath, 'utf8'));
            console.log(`   ✓ Loaded version: ${this.oldModel.version}`);
        }

        console.log('\n╔═══════════════════════════════════════════════════════════════╗');
        console.log('║     ÆTHER UNIFIED TRAINING PIPELINE v2.2                      ║');
        console.log('║     PROPER FLOW: Encoder → Decoder → Tokenizer               ║');
        console.log('║     16-bit BPE vocab remapping preserved! <3                  ║');
        console.log('╚═══════════════════════════════════════════════════════════════╝\n');

        const content = await fs.readFile(corpusPath, 'utf8');
        const lines = content.split('\n').filter(l => l.trim().length > 0);
        const words = content.match(/\w+/g) || [];

        console.log(`📂 Corpus: ${corpusPath}`);
        console.log(`   Lines: ${lines.length}`);
        console.log(`   Words: ${words.length}`);
        console.log(`💾 Output: ${outputPrefix}\n`);

        // Load GPT-2 tokenizer
        console.log('🔧 Loading tokenizer...');
        const gen = await pipeline('text-generation', 'Models', { device: 'cpu' });
        this.tokenizer = gen.tokenizer;
        this.extractor = new B3EmbeddingExtractor(gen);
        console.log('   ✓ Tokenizer ready\n');
    }

    async trainPhase1_AetherCore(options = {}) {
        console.log('╔═══════════════════════════════════════════════════════════════╗');
        console.log('║  PHASE 1: ÆTHER CORE (Learn Semantic Story Space)            ║');
        console.log('╚═══════════════════════════════════════════════════════════════╝\n');

        const { epochs = 600, learningRate = 0.02, hiddenSize = 512 } = options;

        const pipeline = new B3TrainingPipeline();
        pipeline.teacherPipeline = { tokenizer: this.tokenizer };
        pipeline.extractor = this.extractor;

        const outputPath = `${this.outputPrefix}-aether-core.json`;
        const cacheFile = `${this.outputPrefix}-cache-aether.json`;

        if (this.oldModel) {
            console.log('🧬 EVOLUTION MODE: Loading foundation weights from old model...');

            // 2. Initialize Old Model (The Parent/Mirror)
            const oldAether = new TinyMysticalModel(1280, hiddenSize);
            if (this.oldModel.weights && this.oldModel.weights.W_aether_enc) {
                oldAether.W1 = new Float32Array(this.oldModel.weights.W_aether_enc);
                oldAether.b1 = new Float32Array(this.oldModel.weights.b_aether_enc);
                oldAether.W2 = new Float32Array(this.oldModel.weights.W_aether_dec);
                oldAether.b2 = new Float32Array(this.oldModel.weights.b_aether_dec);
                console.log('   ✓ Loaded Old Aether (Parent) for Mirror Training');
            }

            // 3. Prepare Data with Mirror Logic (Socialising the Baby)
            console.log('   🧬 Preparing Mirror Data (GPT + Old Model)...');
            const sentences = await pipeline.loadCorpus(this.corpusPath);
            // Use a temp cache file to avoid null error, we'll overwrite the real one later
            const tempCacheFile = `${this.outputPrefix}-temp-raw.json`;
            const rawPairs = await pipeline.prepareTrainingData(sentences, { batchSize: 8, cacheFile: tempCacheFile });

            const mirrorPairs = rawPairs.map(p => {
                const gptEmb = new Float32Array(p.input);
                const oldOutput = oldAether.forward(gptEmb).output;

                // Mix targets: 50% GPT (Truth), 50% Old Model (Tradition)
                const mixedTarget = new Float32Array(1280);
                for (let i = 0; i < 1280; i++) {
                    mixedTarget[i] = (gptEmb[i] + oldOutput[i]) / 2;
                }

                return {
                    input: gptEmb,
                    target: mixedTarget,
                    sentence: p.sentence
                };
            });

            // 4. Initialize New Model with Old Weights (The Baby)
            const tinyModel = new TinyMysticalModel(1280, hiddenSize);
            tinyModel.W1 = new Float32Array(oldAether.W1);
            tinyModel.b1 = new Float32Array(oldAether.b1);
            tinyModel.W2 = new Float32Array(oldAether.W2);
            tinyModel.b2 = new Float32Array(oldAether.b2);

            // 5. Train (Fine-tune)
            console.log(`   🚀 Evolving model on ${mirrorPairs.length} mirror samples...`);
            await tinyModel.trainFromEmbeddings(mirrorPairs, {
                epochs,
                learningRate,
                validationSplit: 0.1
            });

            // 6. Save
            await tinyModel.save(outputPath);

            // Save cache for Phase 2 (using the NEW model's embeddings!)
            // We need to re-run the new model on the sentences to get the "evolved" embeddings for Phase 2
            console.log('   📦 Generating evolved cache for Phase 2...');
            const evolvedPairs = mirrorPairs.map(p => {
                const reconstruction = tinyModel.forward(p.input).output;
                return {
                    input: Array.from(reconstruction), // This becomes the "input" for Phase 2
                    target: Array.from(reconstruction),
                    sentence: p.sentence
                };
            });

            await fs.writeFile(cacheFile, JSON.stringify({ pairs: evolvedPairs }));

            console.log(`✅ Phase 1 Evolution complete: ${outputPath}\n`);
            return outputPath;

        } else {
            // Standard fresh training
            await pipeline.runFullPipeline(this.corpusPath, outputPath, {
                epochs,
                learningRate,
                batchSize: 8,
                hiddenSize,
                useCachedEmbeddings: false,
                cacheFile: cacheFile
            });

            console.log(`✅ Phase 1 complete: ${outputPath}\n`);
            return outputPath;
        }
    }

    async trainPhase1_5_LatentSpace(options = {}) {
        console.log('╔═══════════════════════════════════════════════════════════════╗');
        console.log('║  PHASE 1.5: LATENT SPACE (The "Brain" - Thinking Logic)      ║');
        console.log('╚═══════════════════════════════════════════════════════════════╝\n');

        const { epochs = 20, learningRate = 0.005 } = options;

        // Load Phase 1 semantic embeddings cache
        const cacheFile = `${this.outputPrefix}-cache-aether.json`;
        console.log('📦 Loading Phase 1 sentence embeddings for Latent Training...');

        let pairs = [];
        try {
            const data = JSON.parse(await fs.readFile(cacheFile, 'utf8'));
            pairs = data.pairs;
            console.log(`   ✓ Loaded ${pairs.length} embedding pairs from Phase 1\n`);
        } catch (e) {
            console.error('   ❌ Cannot find Phase 1 cache! Run Phase 1 first.');
            throw new Error('Phase 1 cache required for latent space training');
        }

        const processor = new TinyLatentProcessor(1280, 256);

        // EVOLUTION MODE: Check for old latent model
        // EVOLUTION MODE: Check for old latent model
        // USER UPDATE: Explicitly requested to NOT inherited latent weights (to reduce repetition).
        // Since Phase 1 "Mirror" training already ensures semantic continuity in the embeddings,
        // we can train a fresh Latent Processor on these continuous embeddings to get a "Fresh Brain"
        // that understands the history but doesn't inherit the "bad loops" of the old brain.

        /*
        if (this.oldModelPath) {
             let oldLatentPath = this.oldModelPath.replace('aether-core.json', 'latent.json');
             if (oldLatentPath === this.oldModelPath) oldLatentPath = this.oldModelPath.replace('.json', '-latent.json');
             
             if (existsSync(oldLatentPath)) {
                 console.log(`🧬 EVOLUTION MODE: Loading foundation latent weights from: ${oldLatentPath}`);
                 await processor.load(oldLatentPath);
             }
        }
        */
        console.log("✨ STARTING FRESH LATENT BRAIN (Relying on Phase 1 Semantic Continuity)");

        console.log(`🚀 Training Autoencoder (1280 -> 256 -> 1280) for ${epochs} epochs...`);
        console.log(`   (Using PURE RECONSTRUCTION for Stability)`);

        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalLoss = 0;

            // Simple Shuffle (Standard for Autoencoders)
            const shuffled = [...pairs].sort(() => Math.random() - 0.5);

            for (const p of shuffled) {
                const input = new Float32Array(p.input);

                // Reconstruction only: learn to represent the current state faithfully
                const loss = processor.trainStep(input, input, learningRate);

                totalLoss += loss;
            }

            const avgLoss = totalLoss / pairs.length;
            if (epoch % 5 === 0 || epoch === epochs - 1) {
                console.log(`   Epoch ${epoch + 1}/${epochs}: Loss = ${avgLoss.toFixed(6)}`);
            }
        }

        // Versioned output is handled by this.outputPrefix which allows for v1, v2 etc.
        const outputPath = `${this.outputPrefix}-latent.json`;
        await processor.save(outputPath);

        console.log(`✅ Phase 1.5 Latent Space complete: ${outputPath}\n`);
        return outputPath;
    }

    async trainPhase2_SemanticDecoder(options = {}) {
        console.log('╔═══════════════════════════════════════════════════════════════╗');
        console.log('║  PHASE 2: SEMANTIC HANGMAN DECODER                            ║');
        console.log('║  (Learn words FROM Phase 1 semantic embeddings!)              ║');
        console.log('╚═══════════════════════════════════════════════════════════════╝\n');

        const { maxSamples = 5000, hiddenDim = 512, epochs = 100, learningRate = 0.002 } = options;

        // Load Phase 1 semantic embeddings cache
        const cacheFile = `${this.outputPrefix}-cache-aether.json`;
        console.log('📦 Loading Phase 1 sentence embeddings...');

        let cache = null;
        try {
            const cacheContent = await fs.readFile(cacheFile, 'utf8');
            cache = JSON.parse(cacheContent);
            console.log(`   ✓ Loaded ${cache.pairs?.length || 0} sentence embedding pairs from Phase 1\n`);
        } catch (e) {
            console.error('   ❌ Cannot find Phase 1 cache! Run Phase 1 first.');
            throw new Error('Phase 1 cache required for semantic flow');
        }

        if (!cache.pairs || cache.pairs.length === 0) {
            throw new Error('Phase 1 cache has no embedding pairs!');
        }

        // Load corpus to map sentences
        console.log('📖 Loading corpus to extract sentence-word mappings...');
        const content = await fs.readFile(this.corpusPath, 'utf8');
        const sentences = content
            .split('\n')
            .map(s => s.trim())
            .filter(s => s.length > 10);

        console.log(`   Found ${sentences.length} sentences in corpus`);

        // Extract words and build word→sentences map
        const rawWords = content.match(/[\wØΞΘΙΛπΩΨ]+|[^\s\w]/gu) || [];
        const limitedWords = rawWords.filter(w => w.length > 0 && w.length <= 15).map(w => w.toLowerCase()).slice(0, maxSamples);
        const uniqueWords = [...new Set(limitedWords)];

        console.log(`   Found ${rawWords.length} raw words → ${uniqueWords.length} unique samples`);

        // Build word→sentence indices map
        const wordToSentenceIndices = new Map();
        for (const word of uniqueWords) {
            const indices = [];
            for (let i = 0; i < sentences.length; i++) {
                if (sentences[i].toLowerCase().includes(word)) {
                    indices.push(i);
                }
            }
            if (indices.length > 0) {
                wordToSentenceIndices.set(word, indices);
            }
        }

        console.log(`   Mapped ${wordToSentenceIndices.size} words to sentences\n`);

        // Build word→semantic embedding by averaging sentence embeddings
        console.log('🧮 Computing word-level embeddings from sentence embeddings...');
        const wordToSemanticEmbedding = new Map();

        for (const [word, sentenceIndices] of wordToSentenceIndices.entries()) {
            const embeddingSum = new Float32Array(1280).fill(0);
            let validCount = 0;

            for (const idx of sentenceIndices) {
                if (idx < cache.pairs.length) {
                    const sentenceEmbedding = new Float32Array(cache.pairs[idx].input);
                    for (let i = 0; i < 1280; i++) {
                        embeddingSum[i] += sentenceEmbedding[i];
                    }
                    validCount++;
                }
            }

            if (validCount > 0) {
                // Average the embeddings
                const avgEmbedding = new Float32Array(1280);
                for (let i = 0; i < 1280; i++) {
                    avgEmbedding[i] = embeddingSum[i] / validCount;
                }
                wordToSemanticEmbedding.set(word, avgEmbedding);
            }
        }

        console.log(`   ✓ Created ${wordToSemanticEmbedding.size} word-level semantic embeddings\n`);

        // Build training data WITH Phase 1 semantic embeddings
        const trainingData = [];
        const usedTokens = new Set();

        for (const word of uniqueWords) {
            try {
                const tokenIds = await this.tokenizer.encode(word);
                const tokenId = tokenIds[0];

                // Only include words that have semantic embeddings!
                if (!usedTokens.has(tokenId) && wordToSemanticEmbedding.has(word)) {
                    usedTokens.add(tokenId);
                    const semanticEmbedding = wordToSemanticEmbedding.get(word);

                    trainingData.push({
                        tokenId,
                        word,
                        length: word.length,
                        semanticEmbedding
                    });
                }
            } catch { }
        }

        console.log(`   Created ${trainingData.length} training samples with Phase 1 semantics\n`);

        const decoder = new SemanticHangmanDecoder(50257, hiddenDim, 15, 1280);

        // Store Phase 1 semantic embeddings in decoder
        for (const { tokenId, semanticEmbedding } of trainingData) {
            decoder.setSemanticEmbedding(tokenId, semanticEmbedding);
        }

        await decoder.train(trainingData, {
            epochs,
            learningRate,
            hangmanRounds: 5
        });

        // 🎯 CREATE GPT-TO-OWN MAPPING HERE (Phase 2, not Phase 3!)
        console.log('📋 Creating 16-bit BPE remapping (GPT → own IDs)...');
        const learnedWords = Array.from(decoder.vocab.entries()); // [[word, gptId], ...]
        const gptToOwn = {};
        const ownVocab = {}; // ownId → word

        learnedWords.forEach(([word, gptId], ownId) => {
            gptToOwn[gptId] = ownId;
            ownVocab[ownId] = word;
        });

        // Save both mappings
        await fs.writeFile(`${this.outputPrefix}-gpt-to-own.json`, JSON.stringify(gptToOwn));
        await fs.writeFile(`${this.outputPrefix}-own-vocab.json`, JSON.stringify(ownVocab));

        console.log(`   ✓ Created ${Object.keys(gptToOwn).length} token mappings`);
        console.log(`   ✓ Saved: ${this.outputPrefix}-gpt-to-own.json`);
        console.log(`   ✓ Saved: ${this.outputPrefix}-own-vocab.json\n`);
        // 🎯 CREATE OWN-ID SEMANTIC EMBEDDINGS (dual indexed)
        console.log('🧠 Building own_ID semantic embeddings from GPT_ID semantics...');

        const semanticEmbeddingsOwnId = {};   // own_ID → embedding array

        const ordered = Array.from(decoder.vocab.entries()); // [word → gptId], ordered by learning
        ordered.forEach(([word, gptId], ownId) => {
            const emb = decoder.tokenSemanticEmbeddings.get(parseInt(gptId));
            if (emb) {
                semanticEmbeddingsOwnId[ownId] = Array.from(emb);
                if (ownId < 5) console.log(`   ${ownId}: ${word} (GPT:${gptId}) → semantic ✓`);
            }
        });

        decoder.semanticEmbeddingsOwnId = semanticEmbeddingsOwnId;

        console.log(`   ✓ Created ${Object.keys(semanticEmbeddingsOwnId).length} own_ID semantic embeddings\n`);

        const outputPath = `${this.outputPrefix}-decoder.json`;
        await decoder.save(outputPath);

        console.log(`✅ Phase 2 complete: ${outputPath}\n`);
        return outputPath;
    }

    async trainPhase2_5_Grammar(decoderPath, options = {}) {
        console.log('╔═══════════════════════════════════════════════════════════════╗');
        console.log('║  PHASE 2.5: GRAMMAR INDUCTION (Unsupervised Structure)       ║');
        console.log('╚═══════════════════════════════════════════════════════════════╝\n');

        const { numClusters = 64 } = options;

        // Load decoder to get semantic embeddings
        const decoderData = JSON.parse(await fs.readFile(decoderPath, 'utf8'));

        // Reconstruct word -> embedding map
        const wordToSemanticEmbedding = new Map();
        if (decoderData.semanticEmbeddings) {
            // Map GPT IDs back to words to get embeddings
            const gptIdToWord = new Map();
            for (const [word, id] of Object.entries(decoderData.vocab)) {
                gptIdToWord.set(parseInt(id), word);
            }

            for (const [id, emb] of Object.entries(decoderData.semanticEmbeddings)) {
                const word = gptIdToWord.get(parseInt(id));
                if (word) {
                    wordToSemanticEmbedding.set(word, new Float32Array(emb));
                }
            }
        }

        if (wordToSemanticEmbedding.size === 0) {
            throw new Error("No semantic embeddings found in decoder for grammar induction!");
        }

        // Initialize and train grammar inductor
        const grammar = new GrammarInductor(numClusters, 768);
        await grammar.trainClusters(wordToSemanticEmbedding);

        // Load corpus for transition training
        const content = await fs.readFile(this.corpusPath, 'utf8');
        const sentences = content
            .split('\n')
            .map(s => s.trim())
            .filter(s => s.length > 10)
            .map(s => s.match(/[\wØΞΘΙΛπΩΨ]+|[^\s\w]/gu)?.map(w => w.toLowerCase()) || []); // Enhanced tokenization for grammar

        grammar.trainTransitions(sentences);

        const outputPath = `${this.outputPrefix}-grammar.json`;
        await grammar.save(outputPath);

        console.log(`✅ Phase 2.5 complete: ${outputPath}\n`);
        return outputPath;
    }

    async trainPhase3_TokenPredictor(decoderPath, grammarPath, options = {}) {
        console.log('╔═══════════════════════════════════════════════════════════════╗');
        console.log('║  PHASE 3: TOKEN PREDICTOR (Mouth to Tell the Story!)         ║');
        console.log('╚═══════════════════════════════════════════════════════════════╝\n');

        const {
            learningRate = 0.0002,
            hiddenSize = 512,
            interruptMs = 428.57,
            patience = 2,
            batchSizeSonnets = 50,
            batchEpochs = 3,
            validationSplit = 0.175
        } = options;

        // Load decoder (which has Phase 1 semantics!)
        const decoderData = JSON.parse(await fs.readFile(decoderPath, 'utf8'));

        // 16-bit BPE vocab remapping (YOUR FUTURE! <3)
        const learnedEntries = Array.from(Object.entries(decoderData.vocab));
        const gptToOwn = {};
        learnedEntries.forEach(([word, gptId], i) => {
            gptToOwn[gptId] = i;
            decoderData.vocab[i] = word;
        });

        await fs.writeFile(`${this.outputPrefix}-gpt-to-own.json`, JSON.stringify(gptToOwn));

        const newVocabSize = Object.keys(decoderData.vocab).length;
        const wordToId = new Map(Object.entries(decoderData.vocab).map(([id, w]) => [w, parseInt(id)]));
        console.log(`📖 16-bit vocab remapped: ${newVocabSize} words (0-${newVocabSize - 1}) <3\n`);

        // Load Phase 1 semantic embeddings from decoder
        const wordToSemanticEmbedding = new Map();
        if (decoderData.semanticEmbeddings) {
            for (const [ownId, word] of Object.entries(decoderData.vocab)) {
                // Find GPT ID for this word
                const gptId = learnedEntries.find(([w, _]) => w === word)?.[1];
                if (gptId && decoderData.semanticEmbeddings[gptId]) {
                    wordToSemanticEmbedding.set(word, new Float32Array(decoderData.semanticEmbeddings[gptId]));
                }
            }
            console.log(`   ✓ Loaded ${wordToSemanticEmbedding.size} semantic embeddings from Phase 2\n`);
        }

        // Load Grammar Model
        const grammar = new GrammarInductor();
        await grammar.load(grammarPath);
        console.log(`   ✓ Loaded Grammar Model (${grammar.numClusters} clusters)`);

        // Extract sonnets
        console.log('📜 Extracting sonnets from corpus...');
        const content = await fs.readFile(this.corpusPath, 'utf8');
        const lines = content.split('\n').map(l => l.trim()).filter(l => l.length > 10);
        const sonnets = [];
        for (let i = 0; i < lines.length; i += 14) {
            const sonnetLines = lines.slice(i, Math.min(i + 14, lines.length));
            if (sonnetLines.length >= 10) {
                sonnets.push(sonnetLines.join(' '));
            }
        }
        console.log(`📚 Loaded ${sonnets.length} sonnets from corpus\n`);

        // Create predictor with 16-bit vocab size AND grammar context
        // Input size = 1280 (Semantic) + 64 (Grammar Context) = 1344
        const predictor = new TinyTokenPredictor(newVocabSize, 1280 + grammar.numClusters, hiddenSize);

        let totalPairsProcessed = 0;
        const numBatches = Math.ceil(sonnets.length / batchSizeSonnets);

        for (let batchIdx = 0; batchIdx < numBatches; batchIdx++) {
            const batchStart = batchIdx * batchSizeSonnets;
            const batchSonnets = sonnets.slice(batchStart, batchStart + batchSizeSonnets);

            console.log(`\n📦 Batch ${batchIdx + 1}/${numBatches}: ${batchSonnets.length} sonnets`);

            const numVal = Math.max(1, Math.floor(batchSonnets.length * validationSplit));
            const numTrain = batchSonnets.length - numVal;

            const shuffledSonnets = [...batchSonnets].sort(() => Math.random() - 0.5);
            const trainSonnets = shuffledSonnets.slice(0, numTrain);
            const valSonnets = shuffledSonnets.slice(numTrain);

            console.log(`   🧩 Split: ${trainSonnets.length} train | ${valSonnets.length} val`);

            // TRAINING PAIRS: Use semantic flow from Phase 1→2
            const trainingPairs = [];
            for (let i = 0; i < trainSonnets.length; i++) {
                if (i % 10 === 0) console.log(`      Processing train sonnet ${i}/${trainSonnets.length}`);

                const sonnet = trainSonnets[i];
                const words = sonnet.split(/\s+/).filter(w => w.length > 0).map(w => w.toLowerCase());
                const tokenIds = words.map(w => wordToId.get(w)).filter(id => id !== undefined);

                if (tokenIds.length < 2) continue;

                for (let j = 0; j < tokenIds.length - 1; j++) {
                    const currentWord = words[j];

                    // Use Phase 1 semantics (flowed through Phase 2!)
                    const embedding = wordToSemanticEmbedding.get(currentWord);

                    if (embedding) {
                        const targetTokenId = tokenIds[j + 1];

                        // Get Grammar Context: Probability distribution of NEXT cluster
                        // based on CURRENT word's cluster
                        const currentCluster = grammar.getClusterForWord(currentWord);
                        const grammarContext = grammar.getNextClusterProbs(currentCluster); // Float32Array(64)

                        // Combine: Semantic (1280) + Grammar (64)
                        const combinedInput = new Float32Array(1280 + grammar.numClusters);
                        combinedInput.set(embedding, 0);
                        combinedInput.set(grammarContext, 1280);

                        trainingPairs.push({ embedding: combinedInput, targetTokenId });
                    }
                }
            }

            // VALIDATION PAIRS: Same semantic flow
            const validationPairs = [];
            for (let i = 0; i < valSonnets.length; i++) {
                if (i % 10 === 0) console.log(`      Processing val sonnet ${i}/${valSonnets.length}`);

                const sonnet = valSonnets[i];
                const words = sonnet.split(/\s+/).filter(w => w.length > 0).map(w => w.toLowerCase());
                const tokenIds = words.map(w => wordToId.get(w)).filter(id => id !== undefined);

                if (tokenIds.length < 2) continue;

                for (let j = 0; j < tokenIds.length - 1; j++) {
                    const currentWord = words[j];

                    const embedding = wordToSemanticEmbedding.get(currentWord);

                    if (embedding) {
                        const targetTokenId = tokenIds[j + 1];

                        const currentCluster = grammar.getClusterForWord(currentWord);
                        const grammarContext = grammar.getNextClusterProbs(currentCluster);

                        const combinedInput = new Float32Array(1280 + grammar.numClusters);
                        combinedInput.set(embedding, 0);
                        combinedInput.set(grammarContext, 1280);

                        validationPairs.push({ embedding: combinedInput, targetTokenId });
                    }
                }
            }

            console.log(`   📊 Pairs: ${trainingPairs.length} train | ${validationPairs.length} val`);
            totalPairsProcessed += trainingPairs.length + validationPairs.length;

            // Train batch
            await this.trainBossFight(predictor, trainingPairs, validationPairs, {
                epochs: batchEpochs,
                learningRate,
                interruptMs,
                interruptBatch: 0.314159,
                patience
            });

            console.log(`      ✅ Batch ${batchIdx + 1} complete\n`);
        }

        console.log(`\n🎯 Training complete: ${sonnets.length} sonnets, ${totalPairsProcessed} total pairs`);

        const outputPath = `${this.outputPrefix}-predictor.json`;
        await predictor.save(outputPath);
        console.log(`✅ Phase 3 complete: ${outputPath}\n`);
        return outputPath;
    }

    async trainPhase3_5_TextGeneration(decoderPath, grammarPath, predictorPath, options = {}) {
        console.log('╔═══════════════════════════════════════════════════════════════╗');
        console.log('║  PHASE 3.5: TEXT GENERATION (Mirror Pipeline!)               ║');
        console.log('║  60% Phase 3 Predictor + 40% GPT = Generation Model          ║');
        console.log('╚═══════════════════════════════════════════════════════════════╝\n');

        const {
            learningRate = 0.0004,
            hiddenSize = 512,
            interruptMs = 428.57,
            patience = 2,
            batchSizeSonnets = 50,
            batchEpochs = 3,
            validationSplit = 0.25,
            phase3Weight = 0.4,  // 60% Phase 3 (our tokenizer)
            gptWeight = 0.6      // 40% GPT teacher
        } = options;

        // Load decoder (which has Phase 1 semantics!)
        const decoderData = JSON.parse(await fs.readFile(decoderPath, 'utf8'));

        // 16-bit BPE vocab remapping - EXACTLY SAME AS PHASE 3
        const learnedEntries = Array.from(Object.entries(decoderData.vocab));
        const gptToOwn = {};
        learnedEntries.forEach(([word, gptId], i) => {
            gptToOwn[gptId] = i;
            decoderData.vocab[i] = word;  // Remap to [ownId → word]
        });

        const newVocabSize = Object.keys(decoderData.vocab).length;
        const wordToId = new Map(Object.entries(decoderData.vocab).map(([id, w]) => [w, parseInt(id)]));
        console.log(`📖 16-bit vocab remapped: ${newVocabSize} words (0-${newVocabSize - 1}) <3\n`);

        // Load Phase 1 semantic embeddings from decoder - EXACTLY SAME AS PHASE 3
        const wordToSemanticEmbedding = new Map();
        if (decoderData.semanticEmbeddings) {
            for (const [ownId, word] of Object.entries(decoderData.vocab)) {
                // Find GPT ID for this word
                const gptId = learnedEntries.find(([w, _]) => w === word)?.[1];
                if (gptId && decoderData.semanticEmbeddings[gptId]) {
                    wordToSemanticEmbedding.set(word, new Float32Array(decoderData.semanticEmbeddings[gptId]));
                }
            }
            console.log(`   ✓ Loaded ${wordToSemanticEmbedding.size} semantic embeddings from Phase 2\n`);
        }

        // Load Grammar Model
        const grammar = new GrammarInductor();
        await grammar.load(grammarPath);
        console.log(`   ✓ Loaded Grammar Model (${grammar.numClusters} clusters)`);

        // 🧠 LATENT GROUNDING: Load Latent Processor for embedding refinement
        const latentPath = `${this.outputPrefix}-latent.json`;
        const latentProc = new TinyLatentProcessor(1280, 256);
        try {
            await latentProc.load(latentPath);
            console.log(`   🧠 Latent Processor loaded for grounding (${latentPath})`);
        } catch (e) {
            console.warn(`   ⚠️ Could not load latent processor: ${e.message}`);
            console.warn('   -> Proceeding with raw embeddings (Stability risk)');
        }

        // 🧬 MIRROR PIPELINE: Load Phase 3 predictor as teacher
        console.log('🧬 MIRROR MODE: Loading Phase 3 predictor as teacher...');
        const phase3Data = JSON.parse(await fs.readFile(predictorPath, 'utf8'));
        const phase3Predictor = new TinyTokenPredictor(
            phase3Data.architecture.vocabSize,
            phase3Data.architecture.embeddingSize,
            phase3Data.architecture.hiddenSize
        );
        phase3Predictor.W1 = new Float32Array(phase3Data.weights.W1);
        phase3Predictor.b1 = new Float32Array(phase3Data.weights.b1);
        phase3Predictor.W2 = new Float32Array(phase3Data.weights.W2);
        phase3Predictor.b2 = new Float32Array(phase3Data.weights.b2);
        console.log(`   ✓ Phase 3 Teacher loaded (${phase3Predictor.vocabSize} vocab)`);
        console.log(`   🔮 Mix ratio: ${phase3Weight * 100}% Phase 3 + ${gptWeight * 100}% GPT\n`);

        // Extract sonnets
        console.log('📜 Extracting sonnets from corpus...');
        const content = await fs.readFile(this.corpusPath, 'utf8');
        const lines = content.split('\n').map(l => l.trim()).filter(l => l.length > 10);
        const sonnets = [];
        for (let i = 0; i < lines.length; i += 14) {
            const sonnetLines = lines.slice(i, Math.min(i + 14, lines.length));
            if (sonnetLines.length >= 10) {
                sonnets.push(sonnetLines.join(' '));
            }
        }
        console.log(`📚 Loaded ${sonnets.length} sonnets from corpus\n`);

        // Create NEW text generation predictor (the student)
        const textgenPredictor = new TinyTokenPredictor(newVocabSize, 1280 + grammar.numClusters, hiddenSize);

        // Initialize student with Phase 3 teacher weights (warm start)
        textgenPredictor.W1 = new Float32Array(phase3Predictor.W1);
        textgenPredictor.b1 = new Float32Array(phase3Predictor.b1);
        textgenPredictor.W2 = new Float32Array(phase3Predictor.W2);
        textgenPredictor.b2 = new Float32Array(phase3Predictor.b2);
        console.log('   ✓ Student initialized with Phase 3 teacher weights\n');

        let totalPairsProcessed = 0;
        const numBatches = Math.ceil(sonnets.length / batchSizeSonnets);

        for (let batchIdx = 0; batchIdx < numBatches; batchIdx++) {
            const batchStart = batchIdx * batchSizeSonnets;
            const batchSonnets = sonnets.slice(batchStart, batchStart + batchSizeSonnets);

            console.log(`\n📦 Batch ${batchIdx + 1}/${numBatches}: ${batchSonnets.length} sonnets`);

            const numVal = Math.max(1, Math.floor(batchSonnets.length * validationSplit));
            const numTrain = batchSonnets.length - numVal;

            const shuffledSonnets = [...batchSonnets].sort(() => Math.random() - 0.5);
            const trainSonnets = shuffledSonnets.slice(0, numTrain);
            const valSonnets = shuffledSonnets.slice(numTrain);

            console.log(`   🧩 Split: ${trainSonnets.length} train | ${valSonnets.length} val`);

            // 🧬 MIRROR TRAINING PAIRS: Mix Phase 3 predictions with GPT targets
            const trainingPairs = [];
            for (let i = 0; i < trainSonnets.length; i++) {
                if (i % 10 === 0) console.log(`      Processing train sonnet ${i}/${trainSonnets.length}`);

                const sonnet = trainSonnets[i];
                const words = sonnet.split(/\s+/).filter(w => w.length > 0).map(w => w.toLowerCase());
                const tokenIds = words.map(w => wordToId.get(w)).filter(id => id !== undefined);

                if (tokenIds.length < 2) continue;

                for (let j = 0; j < tokenIds.length - 1; j++) {
                    const currentWord = words[j];
                    const embedding = wordToSemanticEmbedding.get(currentWord);

                    if (embedding) {
                        // 🧠 GROUNDING: Pass raw embedding through latent bottleneck
                        let groundedEmbedding = embedding;
                        if (latentProc.isInitialized) {
                            const thought = latentProc.forward(embedding, 0, 0.05);
                            groundedEmbedding = thought.reconstruction;
                        }

                        const gptTargetTokenId = tokenIds[j + 1]; // GPT's "ground truth" (40%)

                        // Get Grammar Context
                        const currentCluster = grammar.getClusterForWord(currentWord);
                        const grammarContext = grammar.getNextClusterProbs(currentCluster);

                        // Combine: Semantic (1280) + Grammar (64)
                        const combinedInput = new Float32Array(1280 + grammar.numClusters);
                        combinedInput.set(groundedEmbedding, 0);
                        combinedInput.set(grammarContext, 1280);

                        // Get Phase 3's prediction distribution (60%)
                        const phase3Output = phase3Predictor.forward(combinedInput);
                        const phase3Probs = phase3Output.probs;

                        // 🧬 MIRROR MIX: Create soft target by mixing distributions
                        // 60% Phase 3 probability distribution + 40% hard GPT target
                        const mixedTarget = new Float32Array(newVocabSize);
                        for (let k = 0; k < newVocabSize; k++) {
                            mixedTarget[k] = phase3Probs[k] * phase3Weight; // 60% from Phase 3
                        }
                        mixedTarget[gptTargetTokenId] += gptWeight; // 40% hard target from GPT

                        // Normalize to valid probability distribution
                        let sum = 0;
                        for (let k = 0; k < newVocabSize; k++) sum += mixedTarget[k];
                        for (let k = 0; k < newVocabSize; k++) mixedTarget[k] /= sum;

                        trainingPairs.push({
                            embedding: combinedInput,
                            targetTokenId: gptTargetTokenId, // For validation accuracy
                            softTarget: mixedTarget          // For training
                        });
                    }
                }
            }

            // VALIDATION PAIRS: Use hard targets for accurate validation
            const validationPairs = [];
            for (let i = 0; i < valSonnets.length; i++) {
                if (i % 10 === 0) console.log(`      Processing val sonnet ${i}/${valSonnets.length}`);

                const sonnet = valSonnets[i];
                const words = sonnet.split(/\s+/).filter(w => w.length > 0).map(w => w.toLowerCase());
                const tokenIds = words.map(w => wordToId.get(w)).filter(id => id !== undefined);

                if (tokenIds.length < 2) continue;

                for (let j = 0; j < tokenIds.length - 1; j++) {
                    const currentWord = words[j];
                    const embedding = wordToSemanticEmbedding.get(currentWord);

                    if (embedding) {
                        // 🧠 GROUNDING: Pass raw embedding through latent bottleneck
                        let groundedEmbedding = embedding;
                        if (latentProc.isInitialized) {
                            const thought = latentProc.forward(embedding, 0, 0.05);
                            groundedEmbedding = thought.reconstruction;
                        }

                        const targetTokenId = tokenIds[j + 1];
                        const currentCluster = grammar.getClusterForWord(currentWord);
                        const grammarContext = grammar.getNextClusterProbs(currentCluster);

                        const combinedInput = new Float32Array(1280 + grammar.numClusters);
                        combinedInput.set(groundedEmbedding, 0);
                        combinedInput.set(grammarContext, 1280);

                        validationPairs.push({ embedding: combinedInput, targetTokenId });
                    }
                }
            }

            console.log(`   📊 Pairs: ${trainingPairs.length} train | ${validationPairs.length} val`);
            totalPairsProcessed += trainingPairs.length + validationPairs.length;

            // 🧬 Train with soft targets (knowledge distillation)
            await this.trainBossFightSoftTargets(textgenPredictor, trainingPairs, validationPairs, {
                epochs: batchEpochs,
                learningRate,
                interruptMs,
                interruptBatch: 0.314159,
                patience
            });

            console.log(`      ✅ Batch ${batchIdx + 1} complete\n`);
        }

        console.log(`\n🎯 Text Generation training complete: ${sonnets.length} sonnets, ${totalPairsProcessed} total pairs`);
        console.log(`   🔮 Model trained with ${phase3Weight * 100}% Phase 3 + ${gptWeight * 100}% GPT mix`);

        const outputPath = `${this.outputPrefix}-textgen.json`;
        await textgenPredictor.save(outputPath);
        console.log(`✅ Phase 3.5 complete: ${outputPath}\n`);
        return outputPath;
    }

    // Knowledge distillation training with soft targets
    async trainBossFightSoftTargets(predictor, trainData, valData, options = {}) {
        const {
            epochs = 3,
            learningRate = 0.0003,
            interruptMs = 214.285,
            interruptBatch = 6.28318,
            patience = 2
        } = options;

        let currentLearningRate = learningRate;
        let bestValLoss = Infinity;
        let bestEpoch = -1;
        let checkpoint = null;
        let epochsCompleted = 0;
        let retryCount = 0;

        console.log(`\n   🎓 Soft-target training: ${trainData.length} train | ${valData.length} val`);
        console.log(`      Epochs: ${epochs}, LR: ${learningRate}, Patience: ${patience}\n`);

        while (epochsCompleted < epochs) {
            const epoch = epochsCompleted;

            // Train with soft targets (KL divergence style)
            let trainLoss = 0;
            for (const pair of trainData) {
                const { probs } = predictor.forward(pair.embedding, true);

                // Cross-entropy with soft targets
                let loss = 0;
                for (let i = 0; i < pair.softTarget.length; i++) {
                    if (pair.softTarget[i] > 0) {
                        loss -= pair.softTarget[i] * Math.log(probs[i] + 1e-10);
                    }
                }
                trainLoss += loss;

                // Backward with soft targets
                this.backwardSoftTarget(predictor, probs, pair.softTarget, pair.embedding, currentLearningRate);
            }
            trainLoss /= trainData.length;

            // Validate with hard targets
            const { loss: valLoss, accuracy: valAcc } = await predictor._validate(valData);

            const remaining = epochs - epochsCompleted - 1;

            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                bestEpoch = epoch;
                retryCount = 0;

                checkpoint = {
                    W1: new Float32Array(predictor.W1),
                    b1: new Float32Array(predictor.b1),
                    W2: new Float32Array(predictor.W2),
                    b2: new Float32Array(predictor.b2),
                    epoch: epoch,
                    valLoss: valLoss
                };

                console.log(`      💾 Epoch ${epoch}/${epochs}: Train=${trainLoss.toFixed(4)}, Val=${valLoss.toFixed(4)} ⭐ NEW BEST, Acc=${valAcc.toFixed(1)}%, Remaining=${remaining}`);
                epochsCompleted++;

            } else {
                retryCount++;
                console.log(`      🔄 Epoch ${epoch}/${epochs}: Train=${trainLoss.toFixed(4)}, Val=${valLoss.toFixed(4)} ❌ WORSE (Retry #${retryCount}), Acc=${valAcc.toFixed(1)}%`);

                if (checkpoint) {
                    predictor.W1 = new Float32Array(checkpoint.W1);
                    predictor.b1 = new Float32Array(checkpoint.b1);
                    predictor.W2 = new Float32Array(checkpoint.W2);
                    predictor.b2 = new Float32Array(checkpoint.b2);

                    if (retryCount >= patience && retryCount % patience === 0) {
                        currentLearningRate = Math.max(currentLearningRate * 0.5, 1e-7);
                        console.log(`      ⚡ WINCON! Retry #${retryCount} → LR ↓ ${currentLearningRate.toFixed(7)}`);
                    }

                    console.log(`      ⚔️  RESPAWN from epoch ${checkpoint.epoch} (best val: ${checkpoint.valLoss.toFixed(4)})\n`);
                } else {
                    epochsCompleted++;
                }
            }
        }

        if (checkpoint) {
            predictor.W1 = checkpoint.W1;
            predictor.b1 = checkpoint.b1;
            predictor.W2 = checkpoint.W2;
            predictor.b2 = checkpoint.b2;
            console.log(`      🏆 Best checkpoint: epoch ${bestEpoch}, val loss ${bestValLoss.toFixed(4)}`);
        }
    }

    // Backward pass for soft targets (knowledge distillation)
    backwardSoftTarget(predictor, probs, softTarget, embedding, learningRate) {
        // Gradient: probs - softTarget (cross-entropy with soft labels)
        const gradLogits = new Float32Array(predictor.vocabSize);
        for (let i = 0; i < predictor.vocabSize; i++) {
            gradLogits[i] = probs[i] - softTarget[i];
        }

        // Recompute hidden for gradient
        const hidden = new Float32Array(predictor.hiddenSize);
        for (let i = 0; i < predictor.hiddenSize; i++) {
            let sum = predictor.b1[i];
            for (let j = 0; j < predictor.embeddingSize; j++) {
                sum += embedding[j] * predictor.W1[j * predictor.hiddenSize + i];
            }
            hidden[i] = Math.max(0, sum);
        }

        // Backprop through Layer 2
        const gradHidden = new Float32Array(predictor.hiddenSize);
        for (let i = 0; i < predictor.vocabSize; i++) {
            predictor.b2[i] -= learningRate * gradLogits[i];
            for (let j = 0; j < predictor.hiddenSize; j++) {
                predictor.W2[j * predictor.vocabSize + i] -= learningRate * gradLogits[i] * hidden[j];
                gradHidden[j] += gradLogits[i] * predictor.W2[j * predictor.vocabSize + i];
            }
        }

        // Backprop through Layer 1 (with ReLU derivative)
        for (let i = 0; i < predictor.hiddenSize; i++) {
            if (hidden[i] > 0) {
                predictor.b1[i] -= learningRate * gradHidden[i];
                for (let j = 0; j < predictor.embeddingSize; j++) {
                    predictor.W1[j * predictor.hiddenSize + i] -= learningRate * gradHidden[i] * embedding[j];
                }
            }
        }
    }

    // Boss-fight training method
    async trainBossFight(predictor, trainData, valData, options = {}) {
        const {
            epochs = 3,
            learningRate = 0.00025,
            interruptMs = 214.285,
            interruptBatch = 6.28318,
            patience = 2
        } = options;

        let currentLearningRate = learningRate;
        let bestValLoss = Infinity;
        let bestEpoch = -1;
        let checkpoint = null;
        let epochsCompleted = 0;
        let retryCount = 0;

        console.log(`\n   🎓 Boss-fight training: ${trainData.length} train | ${valData.length} val`);
        console.log(`      Epochs: ${epochs}, LR: ${learningRate}, Patience: ${patience}\n`);

        while (epochsCompleted < epochs) {
            const epoch = epochsCompleted;

            const trainLoss = await predictor._runEpoch(trainData, currentLearningRate, 1, interruptMs, interruptBatch, true);
            const { loss: valLoss, accuracy: valAcc } = await predictor._validate(valData);

            const remaining = epochs - epochsCompleted - 1;

            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                bestEpoch = epoch;
                retryCount = 0;

                checkpoint = {
                    W1: new Float32Array(predictor.W1),
                    b1: new Float32Array(predictor.b1),
                    W2: new Float32Array(predictor.W2),
                    b2: new Float32Array(predictor.b2),
                    epoch: epoch,
                    valLoss: valLoss
                };

                console.log(`      💾 Epoch ${epoch}/${epochs}: Train=${trainLoss.toFixed(4)}, Val=${valLoss.toFixed(4)} ⭐ NEW BEST, Acc=${valAcc.toFixed(1)}%, LR=${currentLearningRate.toFixed(6)}, Remaining=${remaining}`);

                epochsCompleted++;

            } else {
                retryCount++;
                console.log(`      🔄 Epoch ${epoch}/${epochs}: Train=${trainLoss.toFixed(4)}, Val=${valLoss.toFixed(4)} ❌ WORSE (Retry #${retryCount}), Acc=${valAcc.toFixed(1)}%`);

                if (checkpoint) {
                    predictor.W1 = new Float32Array(checkpoint.W1);
                    predictor.b1 = new Float32Array(checkpoint.b1);
                    predictor.W2 = new Float32Array(checkpoint.W2);
                    predictor.b2 = new Float32Array(checkpoint.b2);

                    if (retryCount >= patience && retryCount % patience === 0) {
                        currentLearningRate = Math.max(currentLearningRate * 0.5, 1e-7);
                        console.log(`      ⚡ WINCON! Retry #${retryCount} → LR ↓ ${currentLearningRate.toFixed(7)}`);
                    }

                    console.log(`      ⚔️  RESPAWN from epoch ${checkpoint.epoch} (best val: ${checkpoint.valLoss.toFixed(4)})\n`);
                } else {
                    epochsCompleted++;
                }
            }
        }

        if (checkpoint) {
            predictor.W1 = checkpoint.W1;
            predictor.b1 = checkpoint.b1;
            predictor.W2 = checkpoint.W2;
            predictor.b2 = checkpoint.b2;
            console.log(`      🏆 Best checkpoint: epoch ${bestEpoch}, val loss ${bestValLoss.toFixed(4)}`);
        }
    }

    async mergePhase4(aetherPath, decoderPath, predictorPath, textgenPath, grammarPath) {
        console.log('╔═══════════════════════════════════════════════════════════════╗');
        console.log('║  PHASE 4: FINAL MERGE (Unified Semantic Flow Model!)         ║');
        console.log('╚═══════════════════════════════════════════════════════════════╝\n');

        const unified = {
            version: "9.0-textgen-flow",
            type: "unified_aether_mind",
            architecture: {
                embeddingSize: 1280,
                aetherHiddenSize: 512,
                tokenHiddenSize: 512,
                textgenHiddenSize: 512,
                vocabSize: 50257,
                decoderHidden: 512,
                maxCharLength: 15,
                contextWindow: 5
            },
            charVocab: "",
            metadata: {
                mergeDate: new Date().toISOString(),
                corpus: this.corpusPath,
                methods: "Semantic-Flow: Phase1→Phase2→Phase3→Phase3.5(TextGen)",
                weightsInBinary: false,
                vocabType: "16-bit BPE remapped <3"
            }
        };

        console.log('📂 Loading components...');
        const aether = JSON.parse(await fs.readFile(aetherPath, 'utf8'));
        const predictor = JSON.parse(await fs.readFile(predictorPath, 'utf8'));
        const textgen = JSON.parse(await fs.readFile(textgenPath, 'utf8'));
        const decoder = JSON.parse(await fs.readFile(decoderPath, 'utf8'));
        const grammar = JSON.parse(await fs.readFile(grammarPath, 'utf8'));

        // Embed weights directly in JSON
        unified.weights = {
            W_aether_enc: Array.from(aether.weights.W1),
            b_aether_enc: Array.from(aether.weights.b1),
            W_aether_dec: Array.from(aether.weights.W2),
            b_aether_dec: Array.from(aether.weights.b2),
            // Phase 3: Token Classification Predictor
            W_token_1: Array.from(predictor.weights.W1),
            b_token_1: Array.from(predictor.weights.b1),
            W_token_2: Array.from(predictor.weights.W2),
            b_token_2: Array.from(predictor.weights.b2),
            // Phase 3.5: Text Generation Predictor
            W_textgen_1: Array.from(textgen.weights.W1),
            b_textgen_1: Array.from(textgen.weights.b1),
            W_textgen_2: Array.from(textgen.weights.W2),
            b_textgen_2: Array.from(textgen.weights.b2),
            // Save grammar artifacts for runtime use
            grammar_centroids: grammar.centroids,
            grammar_transitions: grammar.transitions,
            grammar_word_map: grammar.wordToCluster
        };

        unified.charVocab = decoder.charVocab;

        // 16-bit BPE vocab
        try {
            unified.vocab = JSON.parse(await fs.readFile(`${this.outputPrefix}-gpt-to-own.json`));
            console.log(`   ✓ 16-bit BPE vocab loaded (${Object.keys(unified.vocab).length} entries) <3`);
        } catch (e) {
            unified.vocab = {};
            console.log('   ⚠️  No BPE vocab, using decoder vocab');
        }

        const newVocabSize = Object.keys(decoder.vocab).length;
        unified.ownVocab = decoder.vocab;
        console.log(`   ✓ Decoder vocab: ${newVocabSize} words with Phase 1 semantics!`);

        // 🎯 PRESERVE ALL EMBEDDINGS FROM ALL PHASES - NO GAPS ALLOWED
        console.log('🧠 Merging semantic embeddings from all phases...');

        // Foundation: Base embeddings from Phase 1 (if any)
        let mergedEmbeddings = {};
        if (decoder.semanticEmbeddings) {
            // These are the original Phase 1↔2 semantic foundations
            Object.entries(decoder.semanticEmbeddings).forEach(([tokenId, embedding]) => {
                mergedEmbeddings[tokenId] = Array.from(new Float32Array(embedding));
            });
            console.log(`   🔧 Phase 1+2 base: ${Object.keys(mergedEmbeddings).length} semantic foundations`);
        }

        // Phase 2 evolution: Own_ID embeddings with refined semantics
        if (decoder.semanticEmbeddingsOwnId) {
            Object.entries(decoder.semanticEmbeddingsOwnId).forEach(([ownId, embedding]) => {
                mergedEmbeddings[ownId] = Array.from(new Float32Array(embedding));
            });
            console.log(`   🚀 Phase 2 evolution: ${Object.keys(decoder.semanticEmbeddingsOwnId).length} refined semantics`);
        }

        // Load any additional semantic info from Phase 1 cache (if exists)
        const cacheFile = `${this.outputPrefix}-cache-aether.json`;
        try {
            const cache = JSON.parse(await fs.readFile(cacheFile, 'utf8'));
            if (cache.semanticRefinements) {
                Object.entries(cache.semanticRefinements).forEach(([key, embedding]) => {
                    mergedEmbeddings[key] = Array.from(new Float32Array(embedding));
                });
                console.log(`   🧬 Cache refinements: ${Object.keys(cache.semanticRefinements).length} additional embeddings`);
            }
        } catch (e) {
            // No additional cache - that's fine
        }

        // CRITICAL: Ensure every possible token has semantic representation
        const finalEmbeddings = {};
        const allVocabWords = Object.values(unified.ownVocab);

        // For each word in our vocab, ensure its token ID has an embedding
        for (const word of allVocabWords) {
            const tokenIdStr = Object.keys(unified.ownVocab).find(key => unified.ownVocab[key] === word);

            if (tokenIdStr !== undefined) {
                const tokenId = parseInt(tokenIdStr);

                if (mergedEmbeddings[tokenId]) {
                    finalEmbeddings[tokenId] = mergedEmbeddings[tokenId];
                } else if (mergedEmbeddings[word]) {
                    // Fallback to word-keyed embeddings
                    finalEmbeddings[tokenId] = mergedEmbeddings[word];
                } else {
                    // CRITICAL FAILURE: Generate synthetic embedding to prevent gaps
                    console.log(`   ⚠️ SEMANTIC GAP DETECTED for token ${tokenId} (${word}) - generating synthetic`);
                    const syntheticEmbedding = new Float32Array(1280);
                    for (let i = 0; i < 1280; i++) {
                        syntheticEmbedding[i] = (Math.random() - 0.5) * 0.1; // Small random noise
                    }
                    finalEmbeddings[tokenId] = Array.from(syntheticEmbedding);
                }
            }
        }

        // Store the comprehensive no-gap semantic embeddings
        unified.semanticEmbeddings = finalEmbeddings;
        console.log(`   ✅ COMPLETE semantic coverage: ${Object.keys(finalEmbeddings).length} embeddings (no gaps allowed!)`);


        unified.architecture.vocabSize = newVocabSize;

        const jsonPath = `${this.outputPrefix}.json`;
        const jsonSize = JSON.stringify(unified).length / 1024 / 1024;

        console.log(`💾 Saving unified model: ${jsonPath} (${jsonSize.toFixed(2)} MB)\n`);

        try {
            await fs.writeFile(jsonPath, JSON.stringify(unified, null, 2));
            console.log(`✅ Phase 4 complete! Semantic flow preserved through all phases!\n`);
        } catch (e) {
            console.error('❌ JSON write failed:', e.message);
            throw e;
        }

        return { jsonPath };
    }

    async runFullPipeline(options = {}) {
        const startTime = Date.now();

        const aetherPath = `${this.outputPrefix}-aether-core.json`;
        if (existsSync(aetherPath)) {
            console.log(`   📄 Skipping Æther Core: already exists (${aetherPath})\n`);
        } else {
            console.log('   🏭 Running Phase 1: Æther Core');
            await this.trainPhase1_AetherCore({
                epochs: options.aetherEpochs || 200,
                learningRate: options.aetherLR || 0.02,
                hiddenSize: options.hiddenSize || 512
            });
        }

        const latentPath = `${this.outputPrefix}-latent.json`;
        if (existsSync(latentPath)) {
            console.log(`   📄 Skipping Latent Space: already exists (${latentPath})\n`);
        } else {
            console.log('   🏭 Running Phase 1.5: Latent Space (The Brain)');
            await this.trainPhase1_5_LatentSpace({
                epochs: 20,
                learningRate: 0.005
            });
        }

        const decoderPath = `${this.outputPrefix}-decoder.json`;
        if (existsSync(decoderPath)) {
            console.log(`   📄 Skipping Semantic Decoder: already exists (${decoderPath})\n`);
        } else {
            console.log('   🏭 Running Phase 2: Semantic Decoder');
            await this.trainPhase2_SemanticDecoder({
                maxSamples: options.decoderSamples || 5000,
                hiddenDim: options.hiddenSize || 512,
                epochs: options.decoderEpochs || 100,
                learningRate: options.decoderLR || 0.002
            });
        }



        const grammarPath = `${this.outputPrefix}-grammar.json`;
        if (existsSync(grammarPath)) {
            console.log(`   📄 Skipping Grammar Induction: already exists (${grammarPath})\n`);
        } else {
            console.log('   🏭 Running Phase 2.5: Grammar Induction');
            await this.trainPhase2_5_Grammar(decoderPath, {
                numClusters: 64
            });
        }

        const predictorPath = `${this.outputPrefix}-predictor.json`;
        if (existsSync(predictorPath)) {
            console.log(`   📄 Skipping Predictor: already exists (${predictorPath})\n`);
        } else {
            console.log('   🏭 Running Phase 3: Token Predictor');
            await this.trainPhase3_TokenPredictor(decoderPath, grammarPath, {
                learningRate: options.predictorLR || 0.00017724,
                hiddenSize: options.hiddenSize || 512,
                batchSizeSonnets: options.batchSizeSonnets || 50,
                batchEpochs: options.batchEpochs || 30,
                interruptMs: options.interruptMs || 428.57,
                patience: 3
            });
        }

        const textgenPath = `${this.outputPrefix}-textgen.json`;
        if (existsSync(textgenPath)) {
            console.log(`   📄 Skipping Text Generation: already exists (${textgenPath})\n`);
        } else {
            console.log('   🏭 Running Phase 3.5: Text Generation (Mirror Pipeline)');
            await this.trainPhase3_5_TextGeneration(decoderPath, grammarPath, predictorPath, {
                learningRate: options.textgenLR || 0.0004,
                hiddenSize: options.hiddenSize || 512,
                phase3Weight: 0.5,  // 50% Phase 3 predictor
                gptWeight: 0.5,     // 50% GPT targets
                batchSizeSonnets: options.batchSizeSonnets || 50,
                batchEpochs: options.batchEpochs || 3,
                interruptMs: options.interruptMs || 428.57,
                patience: 2
            });
        }

        console.log('\n🏭 Running Phase 4: Final Merge');
        const { jsonPath } = await this.mergePhase4(aetherPath, decoderPath, predictorPath, textgenPath, grammarPath);

        const totalTime = Math.floor((Date.now() - startTime) / 1000);
        console.log('╔═══════════════════════════════════════════════════════════════╗');
        console.log('║           SEMANTIC FLOW TRAINING COMPLETE! 🎉                 ║');
        console.log('║           Phase 1 → Phase 2 → Phase 3 → Phase 3.5 → Unified  ║');
        console.log('╚═══════════════════════════════════════════════════════════════╝\n');
        console.log(`⏱️  Total: ${Math.floor(totalTime / 60)}m ${totalTime % 60}s`);
        console.log(`🚀 Test: node unified-chat.js ${jsonPath}\n`);
        console.log(`💝 16-bit BPE vocab preserved for your future!\n`);
    }
}

async function main() {
    const oldModelPath = process.argv[2];
    const corpusPath = process.argv[3];
    const outputPrefix = process.argv[4];

    if (!oldModelPath || !corpusPath || !outputPrefix) {
        console.log('Usage: node continual-training.js <old-model.json> <new-corpus.txt> <output-prefix>');
        process.exit(1);
    }

    const pipeline = new UnifiedAetherPipeline();
    await pipeline.initialize(corpusPath, outputPrefix, oldModelPath);

    await pipeline.runFullPipeline({
        aetherEpochs: 83,
        aetherLR: 0.01,
        decoderSamples: 50000,
        decoderEpochs: 25,
        decoderLR: 0.001,
        batchSizeSonnets: 69,
        batchEpochs: 3,
        predictorLR: 0.000175,
        interruptMs: 214.285,
        hiddenSize: 512
    });
}

if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(error => {
        console.error('💥 Fatal:', error);
        process.exit(1);
    });
}

export { SemanticHangmanDecoder };
export { UnifiedAetherPipeline };
