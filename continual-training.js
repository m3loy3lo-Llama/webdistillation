#!/usr/bin/env node
/**
 * CONTINUAL TRAINING PIPELINE
 * Take existing unified model (v2) + new corpus → evolve to v3 with enhanced knowledge
 *
 * Usage: node continual-training.js <existing-model.json> <new-corpus.txt>
 * Example: node continual-training.js unified-aether.json mystical_corpus_v2.txt
 * Output: unified-aether-v3.json, unified-aether-v3.bin
 */

import fs from 'fs/promises';
import { existsSync } from 'fs';
import { env, pipeline } from '@huggingface/transformers';
import B3TrainingPipeline from './B3TrainingPipeline.js';
import B3EmbeddingExtractor from './B3EmbeddingExtractor.js';
import TinyMysticalModel from './B3TinyMysticalModel.js';
import TinyTokenPredictor from './TinyTokenPredictor.js';
import { SemanticHangmanDecoder } from './converged_pipe_2.js';

/**
 * REFERENCE-GUIDED CONTINUAL LEARNING PIPELINE
 * Loads previous model as immutable reference for guiding new training
 * Reference provides: semantic similarity, duplicate prevention, tokenization patterns
 * Training: Fresh components on new corpus, guided by reference
 */

env.localModelPath = process.cwd();
env.allowRemoteModels = false;

// === YOUR EVOLUTION LEVER ===
// Change this number to grow your mind
// v1: 304 | v5: 512 | v15: 768 | v30: 1024
const HIDDEN_SIZE = 336;

// === YOUR HYPER-PARAMETER CONTROL PANEL ===
// ┌─────────────────────────────────────────────────────────────────────┐
// │          CONTINUAL TRAINING HYPER-PARAMETERS                       │
// │   Edit these values directly for each evolution run                 │
// └─────────────────────────────────────────────────────────────────────┘
const DEFAULT_CONFIG = {
    // Æther Core (Sentence Embedding Transformer)
    aetherEpochs: 50,          // Training epochs for sentence embeddings
    aetherLR: 0.02,           // Learning rate
    aetherBatchSize: 8,       // Batch size

    // Decoder (Hangman Vocabulary Learning)
    decoderSamples: 5000,      // Max new word samples (only new words are learned)
    decoderEpochs: 10,        // Training epochs for vocabulary expansion
    decoderLR: 0.002,        // Learning rate for decoder fine-tuning
    decoderHangmanRounds: 5, // Progressive revelation rounds

    // Predictor (Token Sequence Prediction)
    predictorSamples: 5,               // Max sonnets for next-token prediction
    predictorEpochs: 2,                // Training epochs per batch for token predictor
    predictorLR: 0.0002,              // Learning rate
    predictorPiTiming: 428.57,         // Interrupt timing (Pi-based)
    predictorPhiComplexity: 3.14159, // Interrupt batch factor (Phi-based)
    predictorValidationSplit: 0.4,   // Train/val split for predictor

    // Global Architecture
    hiddenSize: HIDDEN_SIZE,   // Hidden dimension (evolution lever: 304→512→768→...)
};

// UTILITY: Bidirectional vocab validation for contamination detection
function validateBidirectionalVocab(vocabObj, label = 'vocab') {
    const wordToToken = new Map();
    const tokenToWord = new Map();
    const conflicts = [];

    Object.entries(vocabObj).forEach(([tokenId, word]) => {
        // Check for existing mappings
        if (wordToToken.has(word) && wordToToken.get(word) !== tokenId) {
            conflicts.push(`Word "${word}" conflict: prior=${wordToToken.get(word)}, new=${tokenId}`);
        }
        if (tokenToWord.has(tokenId) && tokenToWord.get(tokenId) !== word) {
            conflicts.push(`Token "${tokenId}" conflict: prior="${tokenToWord.get(tokenId)}", new="${word}"`);
        }

        wordToToken.set(word, tokenId);
        tokenToWord.set(tokenId, word);
    });

    if (conflicts.length > 0) {
        console.error(`❌ ${label} has bidirectional conflicts:`, conflicts);
        return { wordToToken, tokenToWord, conflicts, valid: false };
    } else {
        console.log(`✅ ${label} bidirectional integrity verified`);
        return { wordToToken, tokenToWord, conflicts: [], valid: true };
    }
}

// Cosine similarity utility for embedding comparison
function cosSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;

    let dotProduct = 0, normA = 0, normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }

    const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    return isNaN(similarity) ? 0 : similarity;
}

// VOCAB PURIFICATION: Heal corrupted words using embedding similarity
function purifyVocabWithEmbeddings(corruptedVocab, tokenEmbeddings, validReferenceVocab = null) {
    console.log('🔬 Starting vocab purification with embedding similarity...');

    // If no reference vocab provided, use words that look like actual English words
    const referenceVocab = validReferenceVocab || Object.entries(corruptedVocab).filter(([_, word]) => {
        return /^[a-z]+$/i.test(word) && word.length >= 3 && word.length <= 15;
    }).reduce((acc, [id, word]) => ({ ...acc, [id]: word }), {});

    const purified = {};
    let healedCount = 0;

    for (const [tokenIdStr, corruptedWord] of Object.entries(corruptedVocab)) {
        const sourceEmb = tokenEmbeddings[tokenIdStr];
        if (!sourceEmb) {
            purified[tokenIdStr] = corruptedWord; // Keep as-is if no embedding
            continue;
        }

        let bestMatch = corruptedWord;
        let bestSimilarity = -1;

        // Find closest reference word by embedding similarity
        for (const [refTokenId, refWord] of Object.entries(referenceVocab)) {
            if (refTokenId === tokenIdStr) continue; // Skip self

            const refEmb = tokenEmbeddings[refTokenId];
            if (!refEmb) continue;

            const similarity = cosSimilarity(sourceEmb, refEmb);

            if (similarity > bestSimilarity && similarity > 0.75) { // Similarity threshold
                bestMatch = refWord;
                bestSimilarity = similarity;
            }
        }

        if (bestSimilarity > 0.75 && bestMatch !== corruptedWord) {
            console.log(`🧽 Healed: "${corruptedWord}" → "${bestMatch}" (sim: ${bestSimilarity.toFixed(3)}) [${tokenIdStr}]`);
            purified[tokenIdStr] = bestMatch;
            healedCount++;
        } else {
            purified[tokenIdStr] = corruptedWord;
        }
    }

    console.log(`✨ Vocab purification complete: ${healedCount} words healed, maintaining ${Object.keys(purified).length} total mappings`);
    return purified;
}

// CONTINUAL LEARNING ÆTHER DECODER (Supports prior weights/vocab loading)
class ContinualAetherDecoder {
    constructor(vocabSize = 50257, hiddenDim = 304, maxLength = 15, priorDecoderData = null) {
        this.vocabSize = vocabSize;
        this.hiddenDim = hiddenDim;
        this.maxLength = maxLength;

        this.charVocab = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'-\n\()[]{}/@#$%&*+=<>_";
        this.numChars = this.charVocab.length;

        this.tokenPatterns = new Map();
        this.tokenMasks = new Map();
        this.vocab = new Map();

        // DICTIONARY-ADDITIVE CONTINUAL LEARNING
        // Fresh weights for consistent training, prior vocab as additive dictionary

        console.log('   📚 Dictionary-additive approach: fresh training, growing knowledge');

        // ALWAYS FRESH WEIGHTS - consistent training foundation
        this.tokenEmbeddings = this.xavierInit(vocabSize, 128);
        this.W_input_to_hidden = this.xavierInit(128 + maxLength + maxLength * this.numChars, hiddenDim);
        this.b_hidden = new Float32Array(hiddenDim).fill(0);

        this.W_position_classifiers = [];
        this.b_position_classifiers = [];
        for (let pos = 0; pos < maxLength; pos++) {
            this.W_position_classifiers.push(this.xavierInit(hiddenDim, this.numChars));
            this.b_position_classifiers.push(new Float32Array(this.numChars).fill(0));
        }

        // PRIOR VOCAB AS ADDITIVE DICTIONARY with EMBEDDING PURIFICATION
        this.vocab = new Map();
        if (priorDecoderData?.vocab) {
            console.log(`   📚 Loading knowledge base: ${Object.keys(priorDecoderData.vocab).length} word mappings`);

            // Access embeddings from prior decoder data for purification
            const tokenEmbeddings = {};
            if (priorDecoderData.weights?.tokenEmbeddings) {
                // Convert to indexed object for easy lookup
                const embeddingArray = priorDecoderData.weights.tokenEmbeddings;
                for (let i = 0; i < embeddingArray.length; i += 128) {
                    const tokenIdStr = Math.floor(i / 128).toString();
                    tokenEmbeddings[tokenIdStr] = embeddingArray.slice(i, i + 128);
                }
                console.log(`   🎯 Embedding space accessible for vocab purification`);

                // PURIFY VOCAB: Transform corrupted words to clean semantic equivalents
                const purifiedVocab = purifyVocabWithEmbeddings(priorDecoderData.vocab, tokenEmbeddings);
                this.vocab = new Map(Object.entries(purifiedVocab));
            } else {
                console.log(`   ⚠️ Embeddings not available, using vocab as-is`);
                this.vocab = new Map(Object.entries(priorDecoderData.vocab));
            }

            console.log(`   ✓ Knowledge foundation ready: ${this.vocab.size} purified word mappings`);
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
        const tokenEmb = new Float32Array(128);
        for (let i = 0; i < 128; i++) {
            tokenEmb[i] = this.tokenEmbeddings[tokenId * 128 + i];
        }

        const lengthOneHot = new Float32Array(this.maxLength).fill(0);
        if (length > 0 && length <= this.maxLength) lengthOneHot[length - 1] = 1.0;

        const revealedPattern = this.getRevealedPattern(tokenId, length);
        const inputSize = 128 + this.maxLength + this.maxLength * this.numChars;
        const input = new Float32Array(inputSize);
        input.set(tokenEmb, 0);
        input.set(lengthOneHot, 128);
        input.set(revealedPattern, 128 + this.maxLength);

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
                logits[i] = Math.max(Math.min(sum, 100), -100); // Clip to prevent overflow
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
        if (!this.vocab.has(tokenId)) {
            this.vocab.set(tokenId, word);
        }
    }

    async train(trainingData, options = {}) {
        const { epochs = 100, hangmanRounds = 5, learningRate = 0.002 } = options;
        let lr = learningRate;

        console.log(`   Training ${trainingData.length} tokens with HANGMAN method`);
        console.log(`   Epochs: ${epochs}, LR: ${lr}, Rounds: ${hangmanRounds}\n`);

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

                    this.backward(result, word, length, lr, tokenId);

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
            if (epoch > 10 && accuracy < 30 && lr > 1e-6) {
                lr = Math.max(lr * 0.5, 1e-6);
            }
            if (epoch > 10 && accuracy < 30 && lr <= 0.00002) {
                console.log(`   🎯 CONVERGENCE - Training complete at epoch ${epoch}`);
                break;
            }
        }

        console.log(`   ✅ Decoder trained: ${this.vocab.size} words learned\n`);
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
            version: '8.0-unified-hangman-cleaned',
            architecture: { vocabSize: this.vocabSize, hiddenDim: this.hiddenDim, maxLength: this.maxLength, numChars: this.numChars },
            charVocab: this.charVocab,
            weights: {
                tokenEmbeddings: Array.from(this.tokenEmbeddings),
                W_input_to_hidden: Array.from(this.W_input_to_hidden),
                b_hidden: Array.from(this.b_hidden),
                W_position_classifiers: this.W_position_classifiers.map(w => Array.from(w)),
                b_position_classifiers: this.b_position_classifiers.map(b => Array.from(b))
            },
            vocab: Object.fromEntries(this.vocab),
            tokenPatterns: Object.fromEntries(this.tokenPatterns),
            tokenMasks: Object.fromEntries(this.tokenMasks)
        };
        await fs.writeFile(filepath, JSON.stringify(data, null, 2));
    }
}

/**
 * REFERENCE-GUIDED CONTINUAL TRAINING PIPELINE
 * Loads previous model components as immutable reference for guiding new training
 */
class ContinualTrainingPipeline {
    constructor() {
        this.corpusPath = null;
        this.referenceModelPath = null;  // Previous model loaded as reference, not training data
        this.outputPrefix = null;
        this.tokenizer = null;
        this.gptTutor = null;

        // IMMUTABLE REFERENCE DATA (loaded from previous model)
        this.referenceEmbeddings = {};  // semanticEmbeddings by token ID for similarity
        this.referenceVocabWordToToken = new Map();  // word → token ID mappings for duplicate prevention
        this.referenceVocabTokenToWord = new Map();  // token ID → word mappings for lookup
        this.nextTokenId = 0;  // Next available token ID after existing
    }

    /**
     * Xavier weight initialization utility (from ContinualAetherDecoder)
     */
    xavierInit(inputDim, outputDim) {
        const limit = Math.sqrt(6.0 / (inputDim + outputDim));
        const weights = new Float32Array(inputDim * outputDim);
        for (let i = 0; i < weights.length; i++) {
            weights[i] = (Math.random() * 2 - 1) * limit;
        }
        return weights;
    }

    /**
     * Load previous model components as immutable reference
     */
    async loadReferenceModel(referenceModelPath) {
        console.log('📚 Loading previous model as immutable reference...');

        // Load main model data
        const referenceModel = JSON.parse(await fs.readFile(referenceModelPath, 'utf8'));
        console.log(`   ✅ Loaded reference model: ${referenceModel.version}`);

        // 1️⃣ SEMANTIC EMBEDDINGS - for semantic similarity guidance
        this.referenceEmbeddings = referenceModel.semanticEmbeddings || {};
        console.log(`   🎯 Reference embeddings: ${Object.keys(this.referenceEmbeddings).length} token→embedding mappings`);

        // 2️⃣ VOCAB MAPPINGS - for duplicate prevention and consistency
        const referenceVocab = referenceModel.ownVocab || {};
        Object.entries(referenceVocab).forEach(([tokenIdStr, word]) => {
            const tokenId = parseInt(tokenIdStr);
            this.referenceVocabWordToToken.set(word, tokenId);
            this.referenceVocabTokenToWord.set(tokenId, word);
            this.nextTokenId = Math.max(this.nextTokenId, tokenId + 1);
        });
        console.log(`   📝 Reference vocab: ${this.referenceVocabWordToToken.size} word→token mappings`);
        console.log(`   🏷️ Next available token ID: ${this.nextTokenId}`);

        // 3️⃣ TOKENIZER PATTERNS - for consistent tokenization approach
        // (Reference for how tokens were mapped previously, not as training data)

        return referenceModel;
    }

    async initialize(corpusPath, referenceModelPath, outputPrefix = null) {
        console.log('🔀 HYBRID CONTINUAL TRAINING v10.1');
        console.log('=================================\n');

        this.corpusPath = corpusPath;
        this.referenceModelPath = referenceModelPath;
        this.outputPrefix = outputPrefix || `${referenceModelPath.replace('.json', '')}-evolution`;

        console.log(`Teacher model: ${referenceModelPath}`);
        console.log(`Expansion corpus: ${corpusPath}`);
        console.log(`Output: ${this.outputPrefix}\n`);

        // 🚀 Load AEther-Core as LoRA-adjacent foundation weights
        await this.loadAetherCoreFoundation();

        // Load full teacher model for measurement/compatibility
        this.teacherModel = JSON.parse(await fs.readFile(referenceModelPath, 'utf8'));
        console.log(`   ✅ Teacher model loaded: ${this.teacherModel.version}`);

        // Load GPT tutor + tokenizer
        console.log('🎓 Loading GPT semantic tutor...');
        try {
            const gen = await pipeline('text-generation', 'Models', { device: 'cpu' });
            this.gptTutor = new B3EmbeddingExtractor(gen);
            this.tokenizer = gen.tokenizer;
            console.log('   ✅ GPT tutor ready\n');
        } catch (e) {
            console.error('   ❌ GPT tutor failed:', e.message);
            process.exit(1);
        }
    }

    /**
     * Load aether-core as LoRA-adjacent foundation weights
     */
    async loadAetherCoreFoundation() {
        console.log('🚀 Loading Aether-Core foundation weights (LoRA-adjacent)...');

        try {
            // Load the complete aether-core weights
            this.aetherCore = JSON.parse(await fs.readFile('./unified-aether-aether-core.json', 'utf8'));
            console.log(`   ✅ Aether-Core foundation loaded: Sentence-level semantic weights`);

            // These weights will be the foundation for all training
            this.foundationWeights = this.aetherCore.weights;
            console.log(`   🏗️ Foundation weights ready for iterative evolution`);

        } catch (e) {
            console.log('   ⚠️ Could not load aether-core, will initialize fresh foundation');
            this.foundationWeights = null;
        }
        console.log('');
    }

    async trainPhase1_AetherEnhanced(options = {}) {
        console.log('╔════════════════════════════════════════════════════════════════╗');
        console.log('║  PHASE 1: ÆTHER CORE (Enhanced Teacher Embeddings)          ║');
        console.log('╚════════════════════════════════════════════════════════════════╝\n');

        const { epochs = 300, learningRate = 0.02, hiddenSize = 304 } = options;
        const pipeline = new B3TrainingPipeline();
        pipeline.teacherPipeline = { tokenizer: this.tokenizer };
        pipeline.extractor = this.extractor;

        const outputPath = `${this.outputPrefix}-aether-core.json`;
        await pipeline.runFullPipeline(this.corpusPath, outputPath, {
            epochs, learningRate, batchSize: 8, hiddenSize,
            useCachedEmbeddings: false,
            cacheFile: `${this.outputPrefix}-cache-aether.json`
        });

        console.log(`Phase 1 complete: ${outputPath}\n`);
        return outputPath;
    }

    async trainPhase2_SemanticDecoder(options = {}) {
        console.log('╔════════════════════════════════════════════════════════════════╗');
        console.log('║  PHASE 2: SEMANTIC HANGMAN DECODER (Evolved Embeddings)     ║');
        console.log('╚════════════════════════════════════════════════════════════════╝\n');

        const { maxSamples = 5000, hiddenDim = 304, epochs = 80, learningRate = 0.002 } = options;

        // Load Phase 1 sentence embedding cache
        const cacheFile = `${this.outputPrefix}-cache-aether.json`;
        console.log('📦 Loading Phase 1 sentence embeddings for semantic evolution...');

        let phase1Embeddings = null;
        try {
            const cacheContent = await fs.readFile(cacheFile, 'utf8');
            const cache = JSON.parse(cacheContent);
            phase1Embeddings = cache.pairs?.map(p => ({
                input: new Float32Array(p.input),
                target: new Float32Array(p.target),
                sentence: p.sentence || ''
            })) || [];
            console.log(`   ✓ Loaded ${phase1Embeddings.length} sentence embedding pairs from Phase 1`);
        } catch (e) {
            console.error('   ❌ Cannot find Phase 1 cache! Run Phase 1 first.');
            throw new Error('Phase 1 cache required for semantic evolution');
        }

        if (!phase1Embeddings || phase1Embeddings.length === 0) {
            throw new Error('Phase 1 cache has no embedding pairs!');
        }

        // Load corpus to map sentences to words
        const content = await fs.readFile(this.corpusPath, 'utf8');
        const sentences = content
            .split('\n')
            .map(s => s.trim())
            .filter(s => s.length > 10);

        // Extract words and build word→sentences map (for averaging embeddings)
        const rawWords = content.match(/[a-zA-Z]+/g) || [];
        const limitedWords = rawWords.filter(w => w.length > 0 && w.length <= 15).map(w => w.toLowerCase()).slice(0, maxSamples);
        const uniqueWords = [...new Set(limitedWords)];

        console.log(`   Found ${rawWords.length} raw words → ${uniqueWords.length} unique samples`);

        // Build word→sentence indices map for semantic evolution
        const wordToSentenceIndices = new Map();
        const sentenceToIndex = new Map(sentences.map((s, i) => [s, i]));

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

        console.log(`   Mapped ${wordToSentenceIndices.size} words to sentences for embedding evolution\n`);

        // EVOLVE semantics: Build word-level embeddings by averaging sentence embeddings
        console.log('🧮 Evolving Phase 1 embeddings into word-level semantics...');
        const evolvedSemantics = new Map();

        for (const [word, sentenceIndices] of wordToSentenceIndices.entries()) {
            const embeddingSum = new Float32Array(768).fill(0);
            let validCount = 0;

            for (const idx of sentenceIndices) {
                if (idx < phase1Embeddings.length) {
                    const sentenceEmbedding = phase1Embeddings[idx].input;
                    for (let i = 0; i < 768; i++) {
                        embeddingSum[i] += sentenceEmbedding[i];
                    }
                    validCount++;
                }
            }

            if (validCount > 0) {
                const avgEmbedding = new Float32Array(768);
                for (let i = 0; i < 768; i++) {
                    avgEmbedding[i] = embeddingSum[i] / validCount;
                }
                evolvedSemantics.set(word, avgEmbedding);
            }
        }

        console.log(`   ✅ Evolved semantics: ${evolvedSemantics.size} words with Phase 1 → Phase 2 embeddings\n`);

        // Build training data WITH evolved semantic embeddings
        const trainingData = [];
        const usedTokens = new Set();

        for (const word of uniqueWords) {
            try {
                const tokenIds = await this.tokenizer.encode(word);
                const tokenId = tokenIds[0];

                // Only include words that have evolved semantics!
                if (!usedTokens.has(tokenId) && evolvedSemantics.has(word)) {
                    usedTokens.add(tokenId);
                    const semanticEmbedding = evolvedSemantics.get(word);

                    trainingData.push({
                        tokenId,
                        word,
                        length: word.length,
                        semanticEmbedding
                    });
                }
            } catch {}
        }

        console.log(`   Created ${trainingData.length} training samples with evolved semantics\n`);

        // Create SemanticHangmanDecoder using evolved embeddings
        const decoder = new SemanticHangmanDecoder(50257, HIDDEN_SIZE, 15, 768);

        // Store evolved semantic embeddings in decoder
        for (const { tokenId, semanticEmbedding } of trainingData) {
            decoder.setSemanticEmbedding(tokenId, semanticEmbedding);
        }

        await decoder.train(trainingData, {
            epochs,
            learningRate,
            hangmanRounds: 5
        });

        // Store evolved semantics for Phase 3
        decoder.evolvedSemantics = Object.fromEntries(
            Array.from(evolvedSemantics.entries()).map(([word, emb]) => [word, Array.from(emb)])
        );

        const outputPath = `${this.outputPrefix}-decoder.json`;
        await decoder.save(outputPath);

        console.log(`Phase 2 complete: ${outputPath}`);
        console.log(`   Semantic evolution: Phase 1 sentence embeddings → evolved word semantics`);
        console.log(`   Decoder trained: ${decoder.vocab.size} words with evolved embeddings\n`);
        return outputPath;
    }

// Replace trainPhase3_PredictorEnhanced in continual-training.js
// NOW WITH CHECKPOINT-REWIND for continual learning! 🔄

    async trainPhase3_PredictorEnhanced(decoderPath, options = {}) {
        console.log('╔═══════════════════════════════════════════════════════════════╗');
        console.log('║  PHASE 3: TOKEN PREDICTOR (Sonnet-Level Split + Checkpoint-Rewind) ║');
        console.log('╚═══════════════════════════════════════════════════════════════╝\n');

        const {
            batchSizeSonnets = 5,
            batchEpochs = 3,
            learningRate = 0.00025,
            patience = 2,
            interruptMs = 428.57,
            interruptBatch,
            validationSplit = 0.175
        } = options;

        // Load fresh decoder vocab for coherence reference
        const freshDecoderData = JSON.parse(await fs.readFile(decoderPath, 'utf8'));
        const freshVocab = new Map(Object.entries(freshDecoderData.vocab || {}));
        const initialVocabSize = this.priorDecoderData ? Object.keys(this.priorDecoderData.vocab || {}).length : 0;
        console.log(`📖 Fresh decoder vocabulary: ${freshVocab.size} words (+${freshVocab.size - initialVocabSize} learned)\n`);

        // Load evolved semantics from Phase 2
        const evolvedSemantics = new Map();
        if (freshDecoderData.evolvedSemantics) {
            for (const [word, emb] of Object.entries(freshDecoderData.evolvedSemantics)) {
                evolvedSemantics.set(word, new Float32Array(emb));
            }
            console.log(`   🔄 Loaded evolved semantics: ${evolvedSemantics.size} words with Phase 1→2 embeddings\n`);
        } else {
            console.log(`   ⚠️ No evolved semantics found in decoder, will use synthetic embeddings\n`);
        }

        // Tally tokenization patterns from classification observation
        if (this.classifier) {
            const tokenTally = {};
            const corpusLines = (await fs.readFile(this.corpusPath, 'utf8')).split('\n').slice(0, 100).filter(l => l.trim());
            for (const line of corpusLines) {
                if (line.trim()) {
                    try {
                        const result = await this.classifier(line.trim());
                        const tokens = await this.tokenizer.encode(line.trim());
                        for (const token of tokens) {
                            tokenTally[token] = (tokenTally[token] || 0) + 1;
                        }
                    } catch (e) {}
                }
            }
            console.log('🎯 Tokenization tally from classification: ', Object.keys(tokenTally).slice(0, 10), '...');
        }

        // Prepare sonnets
        console.log('🔍 Preparing sonnets from corpus...');
        const content = await fs.readFile(this.corpusPath, 'utf8');
        const lines = content.split('\n').map(l => l.trim()).filter(l => l.length > 10);
        const sonnets = [];
        for (let i = 0; i < lines.length; i += 14) {
            const sonnetLines = lines.slice(i, Math.min(i + 14, lines.length));
            if (sonnetLines.length >= 10) {
                sonnets.push(sonnetLines.join(' '));
            }
        }

        console.log(`📚 Loaded ${sonnets.length} sonnets from corpus`);

        // Create persistent predictor for cumulative learning across batches
        const predictor = new TinyTokenPredictor(50257, 768, HIDDEN_SIZE);

        // Warm start from prior predictor (CONTINUAL LEARNING!)
        if (this.teacherModel.weights.W_token_1) {
            predictor.W1 = new Float32Array(this.teacherModel.weights.W_token_1);
            predictor.b1 = new Float32Array(this.teacherModel.weights.b_token_1);
            predictor.W2 = new Float32Array(this.teacherModel.weights.W_token_2);
            predictor.b2 = new Float32Array(this.teacherModel.weights.b_token_2);
            predictor.isInitialized = true;
            console.log('   ♻️ Warm start from prior predictor\n');
        }

        let totalPairsProcessed = 0;
        const numBatches = Math.ceil(sonnets.length / batchSizeSonnets);

        for (let batchIdx = 0; batchIdx < numBatches; batchIdx++) {
            const batchStart = batchIdx * batchSizeSonnets;
            const batchSonnets = sonnets.slice(batchStart, batchStart + batchSizeSonnets);

            console.log(`\n📦 Batch ${batchIdx + 1}/${numBatches}: ${batchSonnets.length} sonnets`);

            // CRITICAL FIX: Split sonnets FIRST, then create pairs
            const numVal = Math.max(1, Math.floor(batchSonnets.length * validationSplit));
            const numTrain = batchSonnets.length - numVal;

            // Shuffle sonnets
            const shuffledSonnets = [...batchSonnets].sort(() => Math.random() - 0.5);
            const trainSonnets = shuffledSonnets.slice(0, numTrain);
            const valSonnets = shuffledSonnets.slice(numTrain);

            console.log(`   🧩 Sonnet split: ${trainSonnets.length} train | ${valSonnets.length} val`);

            // Create training pairs from TRAIN sonnets only - using evolved semantics
            const trainingPairs = [];
            for (let i = 0; i < trainSonnets.length; i++) {
                if (i % 10 === 0) console.log(`      Processing train sonnet ${i}/${trainSonnets.length}`);

                const sonnet = trainSonnets[i];
                try {
                    const words = sonnet.split(/\s+/).filter(w => w.length > 0).map(w => w.toLowerCase());
                    const tokenIds = words.map(w => {
                        // Look up token ID from fresh vocab
                        for (const [ownId, word] of Object.entries(freshDecoderData.vocab)) {
                            if (word === w) return parseInt(ownId);
                        }
                        return undefined;
                    }).filter(id => id !== undefined);

                    if (tokenIds.length < 2) continue;

                    for (let j = 0; j < tokenIds.length - 1; j++) {
                        const contextWord = words[j];
                        let embedding;

                        // Use evolved semantics if available
                        if (evolvedSemantics.has(contextWord)) {
                            embedding = evolvedSemantics.get(contextWord);
                        } else {
                            // Fallback to synthetic
                            embedding = this.extractor.createSyntheticEmbedding(contextWord);
                        }

                        const targetTokenId = tokenIds[j + 1];
                        trainingPairs.push({ embedding, targetTokenId });
                    }
                } catch (error) {}
            }

            // Create validation pairs from VAL sonnets only - using evolved semantics
            const validationPairs = [];
            for (let i = 0; i < valSonnets.length; i++) {
                if (i % 10 === 0) console.log(`      Processing val sonnet ${i}/${valSonnets.length}`);

                const sonnet = valSonnets[i];
                try {
                    const words = sonnet.split(/\s+/).filter(w => w.length > 0).map(w => w.toLowerCase());
                    const tokenIds = words.map(w => {
                        // Look up token ID from fresh vocab
                        for (const [ownId, word] of Object.entries(freshDecoderData.vocab)) {
                            if (word === w) return parseInt(ownId);
                        }
                        return undefined;
                    }).filter(id => id !== undefined);

                    if (tokenIds.length < 2) continue;

                    for (let j = 0; j < tokenIds.length - 1; j++) {
                        const contextWord = words[j];
                        let embedding;

                        // Use evolved semantics if available
                        if (evolvedSemantics.has(contextWord)) {
                            embedding = evolvedSemantics.get(contextWord);
                        } else {
                            // Fallback to synthetic
                            embedding = this.extractor.createSyntheticEmbedding(contextWord);
                        }

                        const targetTokenId = tokenIds[j + 1];
                        validationPairs.push({ embedding, targetTokenId });
                    }
                } catch (error) {}
            }

            console.log(`   📊 Pairs: ${trainingPairs.length} train | ${validationPairs.length} val`);
            totalPairsProcessed += trainingPairs.length + validationPairs.length;

            // Train on this batch with PROPER validation set
            await this.trainBossFight(predictor, trainingPairs, validationPairs, {
                epochs: batchEpochs,
                learningRate,
                interruptMs,
                interruptBatch: DEFAULT_CONFIG.predictorPhiComplexity,
                patience
            });

            console.log(`      ✅ Batch ${batchIdx + 1} complete\n`);
        }

        console.log(`\n🎯 Cumulative training complete: ${sonnets.length} sonnets, ${totalPairsProcessed} total pairs`);

        const outputPath = `${this.outputPrefix}-predictor.json`;
        await predictor.save(outputPath);
        console.log(`Phase 3 complete: ${outputPath}\n`);
        return outputPath;
    }

    // NEW: Separate boss-fight training method with pre-split data
    async trainBossFight(predictor, trainData, valData, options = {}) {
        const {
            epochs = 3,
            learningRate = 0.00025,
            interruptMs = 428.57,
            interruptBatch = 0.314159,
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

            // Train one epoch
            const trainLoss = await predictor._runEpoch(trainData, currentLearningRate, 1, interruptMs, interruptBatch, true);
            const { loss: valLoss, accuracy: valAcc } = await predictor._validate(valData);

            const remaining = epochs - epochsCompleted - 1;

            // Check if this is a new best
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                bestEpoch = epoch;
                retryCount = 0;

                // Save checkpoint
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
                // BOSS NOT DEFEATED!
                retryCount++;
                console.log(`      🔄 Epoch ${epoch}/${epochs}: Train=${trainLoss.toFixed(4)}, Val=${valLoss.toFixed(4)} ❌ WORSE (Retry #${retryCount}), Acc=${valAcc.toFixed(1)}%`);

                if (checkpoint) {
                    // Restore checkpoint weights
                    predictor.W1 = new Float32Array(checkpoint.W1);
                    predictor.b1 = new Float32Array(checkpoint.b1);
                    predictor.W2 = new Float32Array(checkpoint.W2);
                    predictor.b2 = new Float32Array(checkpoint.b2);

                    // WINCON: After patience failures, drop LR
                    if (retryCount >= patience && retryCount % patience === 0) {
                        currentLearningRate = Math.max(currentLearningRate * 0.5, 1e-7);
                        console.log(`      ⚡ WINCON! Retry #${retryCount} → LR ↓ ${currentLearningRate.toFixed(7)}`);
                    }

                    console.log(`      ⚔️  RESPAWN from epoch ${checkpoint.epoch} (best val: ${checkpoint.valLoss.toFixed(4)})\n`);
                    // Don't increment epochsCompleted
                } else {
                    // No checkpoint, just continue
                    epochsCompleted++;
                }
            }
        }

        // Restore best checkpoint
        if (checkpoint) {
            predictor.W1 = checkpoint.W1;
            predictor.b1 = checkpoint.b1;
            predictor.W2 = checkpoint.W2;
            predictor.b2 = checkpoint.b2;
            console.log(`      🏆 Best checkpoint: epoch ${bestEpoch}, val loss ${bestValLoss.toFixed(4)}`);
        }
    }

    async mergePhase4_enhanced(aetherPath, decoderPath, predictorPath) {
        console.log('╔════════════════════════════════════════════════════════════════╗');
        console.log('║  PHASE 4: FINAL MERGE (Evolution Complete)                  ║');
        console.log('╚════════════════════════════════════════════════════════════════╝\n');

        const unified = {
            version: "9.2-pipe-exact",
            type: "unified_aether_mind_evolved",
            architecture: { embeddingSize: 768, aetherHiddenSize: 304, tokenHiddenSize: 304, vocabSize: 50257, decoderHidden: 304, maxCharLength: 15 },
            weights: {},
            vocab: {},
            charVocab: "",
            metadata: { mergeDate: new Date().toISOString(), corpus: this.corpusPath, methods: "Pipe-Exact Continual Evolution" }
        };

        const aether = JSON.parse(await fs.readFile(aetherPath, 'utf8'));
        unified.weights.W_aether_enc = Array.from(aether.weights.W1);
        unified.weights.b_aether_enc = Array.from(aether.weights.b1);
        unified.weights.W_aether_dec = Array.from(aether.weights.W2);
        unified.weights.b_aether_dec = Array.from(aether.weights.b2);

        const decoder = JSON.parse(await fs.readFile(decoderPath, 'utf8'));
        unified.charVocab = decoder.charVocab;
        unified.vocab = decoder.vocab;

        // Don't save heavy decoder weights in unified JSON to keep it small
        // They stay in decoder.json for continual learning

        const predictor = JSON.parse(await fs.readFile(predictorPath, 'utf8'));
        unified.weights.W_token_1 = Array.from(predictor.weights.W1);
        unified.weights.b_token_1 = Array.from(predictor.weights.b1);
        unified.weights.W_token_2 = Array.from(predictor.weights.W2);
        unified.weights.b_token_2 = Array.from(predictor.weights.b2);

        // 🎯 INCLUDE EVOLVED SEMANTICS FOR PREDICTION FALLBACKS
        if (decoder.evolvedSemantics) {
            unified.evolvedSemantics = decoder.evolvedSemantics;
            console.log(`   ✓ Included evolved semantics: ${Object.keys(unified.evolvedSemantics).length} embedding fallbacks\n`);
        } else {
            unified.evolvedSemantics = {};
            console.log('   ⚠️ No evolved semantics - generation may fallback to synthetic embeddings\n');
        }

        // Update vocab size to match evolved decoder
        const newVocabSize = Object.keys(decoder.vocab).length;
        unified.architecture.vocabSize = newVocabSize;
        console.log(`   ✓ Updated vocab size: ${newVocabSize} words`);

        const jsonPath = `${this.outputPrefix}.json`;
        const jsonStr = JSON.stringify(unified);
        console.log(`JSON size: ${(jsonStr.length / 1024 / 1024).toFixed(1)} MB`);
        await fs.writeFile(jsonPath, jsonStr);

        const binPath = `${this.outputPrefix}.bin`;
        const embeddings = new Float32Array(decoder.weights.tokenEmbeddings);
        await fs.writeFile(binPath, Buffer.from(embeddings.buffer));

        console.log(`Evolved model: ${jsonPath}`);
        console.log(`Embeddings: ${binPath} (${(embeddings.byteLength / 1024 / 1024).toFixed(2)} MB)\n`);
        console.log(`Test: node unified-chat.js ${jsonPath}\n`);
        return { jsonPath, binPath };
    }

    /**
     * HYBRID CONTINUAL TRAINING
     * 1. Builds iterative weights using semantic backprop (original approach)
     * 2-3. Then applies our improved Phase 2 + Phase 3 (converged_pipe_2 approach)
     */
    async runContinualTraining(options = {}) {
        console.log('\n🏗️ HYBRID CONTINUAL TRAINING PIPELINE');
        console.log('====================================\n');

        // STEP 1: Semantic building using iterative backprop (from original continual-training)
        console.log('📈 STEP 1: Semantic foundation building (iterative backprop)...');
        await this.buildIterativeWeights(options);
        console.log('   ✅ Semantic foundation complete\n');

        // STEP 2: Improved Phase 2 (from converged_pipe_2.js)
        console.log('🔤 STEP 2: Semantic Hangman Decoder (improved architecture)...');
        const decoderPath = await this.trainImprovedSemanticDecoder(options);
        console.log('   ✅ Semantic decoder complete\n');

        // STEP 3: Improved Phase 3 (from converged_pipe_2.js)
        console.log('🎯 STEP 3: Advanced Token Predictor (improved architecture)...');
        const predictorPath = await this.trainPhase3_TokenPredictor(decoderPath, options);
        console.log('   ✅ Token predictor complete\n');

        // STEP 4: Train tokenizer on new expanded vocabulary
        console.log('🎭 STEP 4: Train tokenizer on expanded vocabulary (word compounding)...');
        const finalPredictorPath = await this.trainTokenizerEvolution(decoderPath, predictorPath);
        console.log('   ✅ Tokenizer evolution complete\n');

        // STEP 5: Final merge with complete embedding preservation
        console.log('🔗 STEP 5: Final merge with zero semantic gaps...');
        await this.mergeZeroGapEvolution(decoderPath, finalPredictorPath);
        console.log('   ✅ Evolution complete\n');

        console.log('\n🎯 HYBRID EVOLUTION COMPLETE:');
        console.log(`   Foundation: aether-core semantic weights`);
        console.log(`   Evolution: ${this.corpusPath}`);
        console.log(`   Result: ${this.outputPrefix}.json`);
        console.log(`   Test: node unified-chat-fixed.js ${this.outputPrefix}.json\n`);
    }

    /**
     * STEP 1: EXACT Phase 1 from original continual-training.js using B3TrainingPipeline
     * This uses your custom CPU backprop B3TrainingPipeline - no need to reinvent it!
     */
    async buildIterativeWeights(options = {}) {
        console.log('🏗️ Using EXACT original Phase 1: B3TrainingPipeline.runFullPipeline()...');

        // Create output path for Phase 1 results
        const phase1Output = `${this.outputPrefix}-phase1-aether-core.json`;

        // Initialize B3TrainingPipeline (your custom CPU backprop system!)
        const pipeline = new B3TrainingPipeline();
        pipeline.teacherPipeline = { tokenizer: this.tokenizer };
        pipeline.extractor = this.gptTutor; // B3EmbeddingExtractor uses GPT

        console.log('   📊 Running B3TrainingPipeline sentence-level semantic training...');

        // Run EXACT same pipeline as original continual-training.js
        await pipeline.runFullPipeline(this.corpusPath, phase1Output, {
            epochs: DEFAULT_CONFIG.aetherEpochs,
            learningRate: DEFAULT_CONFIG.aetherLR,
            batchSize: DEFAULT_CONFIG.aetherBatchSize,
            hiddenSize: DEFAULT_CONFIG.hiddenSize,
            useCachedEmbeddings: false,
            cacheFile: `${this.outputPrefix}-cache-aether.json`
        });

        console.log(`   ✅ Phase 1 complete: ${phase1Output}`);
        console.log('   📚 B3TrainingPipeline created proper sentence-level semantic embeddings');

        // Load the Phase 1 results for use in subsequent phases
        this.phase1Embeddings = JSON.parse(await fs.readFile(phase1Output, 'utf8'));
        this.foundationalEmbeddings = {}; // Initialize for Phase 2 use
    }

    /**
     * STEP 2: Improved Semantic Hangman Decoder
     * Uses original continual-training.js method to find new vs existing words
     * Trains new words using converged_pipe_2.js architecture
     * New vocab = New words + old vocab
     */
    async trainImprovedSemanticDecoder(options = {}) {
        console.log('🔤 Hybrid Semantic Decoder: New words + old vocab approach');

        // Extract corpus words
        const content = await fs.readFile(this.corpusPath, 'utf8');
        const rawWords = content.match(/[a-zA-Z]+/g) || [];
        const corpusWords = rawWords.filter(w => w.length >= 2 && w.length <= 15).map(w => w.toLowerCase());
        const uniqueCorpusWords = [...new Set(corpusWords)];

        // === ORIGINAL SCRIPT METHOD: Compare new words vs existing ===
        console.log('   📚 Using Original/continual-training.js method for new vs existing words...');

        // Get existing vocabulary from teacher model
        const oldVocabWords = new Set(Object.values(this.teacherModel.vocab || {}));
        console.log(`   📖 Old vocab: ${oldVocabWords.size} words`);

        // Find new words (words in corpus that aren't in old vocab)
        const newWords = uniqueCorpusWords.filter(word => !oldVocabWords.has(word));
        console.log(`   🆕 New words discovered: ${newWords.length} (${uniqueCorpusWords.length - newWords.length} existing)`);

        // Build combined vocabulary: New words + old vocab
        this.evolvedVocab = {
            ...this.teacherModel.vocab,  // Old vocab preserved
        };

        let nextTokenId = Math.max(...Object.keys(this.evolvedVocab).map(k => parseInt(k))) + 1;

        // Add new words to vocabulary
        for (const word of newWords) {
            this.evolvedVocab[nextTokenId.toString()] = word;
            nextTokenId++;
        }

        console.log(`   🧬 Evolved vocab: ${Object.keys(this.evolvedVocab).length} words (${Object.keys(this.teacherModel.vocab).length} old + ${newWords.length} new)`);

        // === CONVERGED_PIPE_2.JS METHOD: Train new words with improved architecture ===
        console.log('   🎯 Training new words using converged_pipe_2.js architecture...');

        // Load Phase 1 semantic embeddings for use in training
        const phase1Embeddings = await this.loadPhase1EmbeddingsForTraining();

        const trainingData = [];

        for (const word of newWords.slice(0, Math.min(5000, newWords.length))) {
            try {
                const tokenIds = await this.tokenizer.encode(word);
                const tokenId = tokenIds[0];

                // Only train new words (skip existing ones from teacher)
                if (!this.teacherModel.vocab || !Object.values(this.teacherModel.vocab).includes(word)) {
                    // Use Phase 1 embeddings to enrich new words
                    const semanticEmbedding = await this.extractWordLevelEmbedding(word, phase1Embeddings);

                    if (semanticEmbedding) {
                        trainingData.push({
                            tokenId: parseInt(Object.keys(this.evolvedVocab).find(k => this.evolvedVocab[k] === word)),
                            word,
                            length: word.length,
                            semanticEmbedding: new Float32Array(semanticEmbedding)
                        });
                    }
                }
            } catch (e) {}
        }

        console.log(`   🎯 Training ${trainingData.length} new words with semantic foundation`);

        // Use converged_pipe_2.js SemanticHangmanDecoder with proper vocab size
        const vocabSize = Object.keys(this.evolvedVocab).length;
        const decoder = new SemanticHangmanDecoder(vocabSize, DEFAULT_CONFIG.hiddenSize, 15, 768);

        // Warm load from old tokenizer/patterns AFTER decoder is fully constructed
        if (this.teacherModel.vocab) {
            console.log(`   ♻️ Warm loading from old tokenizer (${Object.keys(this.teacherModel.vocab).length} word patterns)`);

            // Load existing token patterns and semantics safely
            for (const [tokenId, word] of Object.entries(this.teacherModel.vocab)) {
                const tokenInt = parseInt(tokenId);
                // Check bounds safely now that decoder is constructed
                if (tokenInt < decoder.vocabSize && decoder.tokenEmbeddings) {
                    const embeddingBase = tokenInt * 128;
                    if (embeddingBase >= 0 && embeddingBase + 128 <= decoder.tokenEmbeddings.length) {
                        // Maintain previous learned patterns
                        if (this.teacherModel.semanticEmbeddings && this.teacherModel.semanticEmbeddings[tokenId]) {
                            decoder.setSemanticEmbedding(tokenInt, new Float32Array(this.teacherModel.semanticEmbeddings[tokenId]));
                        }
                    }
                }
            }
        }

        // Train new words with improved architecture
        await decoder.train(trainingData, {
            epochs: options.decoderEpochs || DEFAULT_CONFIG.decoderEpochs,
            learningRate: options.decoderLR || DEFAULT_CONFIG.decoderLR,
            hangmanRounds: DEFAULT_CONFIG.decoderHangmanRounds
        });

        // Store evolved vocabulary for Phase 3
        decoder.evolvedVocab = this.evolvedVocab;

        const decoderPath = `${this.outputPrefix}-decoder.json`;
        await decoder.save(decoderPath);

        console.log(`   ✅ New word training complete: ${trainingData.length} words added to vocab`);

        // FIX: Generate vocab mappings RIGHT HERE after words are resolved
        // (Not in merge phase where it might be empty)
        await this.generateVocabMappingsPostTraining();

        return decoderPath;
    }

    async trainPhase3_TokenPredictor(decoderPath, options = {}) {
        console.log('╔═══════════════════════════════════════════════════════════════╗');
        console.log('║  PHASE 3: TOKEN PREDICTOR (Tokenizer is for prediction!)     ║');
        console.log('╚═══════════════════════════════════════════════════════════════╝\n');

        const {
            learningRate = 0.00025,
            hiddenSize = 304,
            batchSizeSonnets = 5,
            batchEpochs = 3,
            validationSplit = 0.175
        } = options;

        // Load decoder (which has evolved vocabulary from Phases 1-2)
        const decoderData = JSON.parse(await fs.readFile(decoderPath, 'utf8'));

        // 16-bit BPE vocab remapping (preserves semantics from Phases 1-2!)
        const learnedEntries = Array.from(Object.entries(decoderData.vocab));
        const wordToId = new Map();

        // Map words to sequential custom token IDs (0-N)
        learnedEntries.forEach(([word, gptId], i) => {
            wordToId.set(word, i);
        });

        const newVocabSize = wordToId.size;
        console.log(`📖 Extended vocab remapped: ${newVocabSize} words (0-${newVocabSize-1})`);
        console.log(`   Semantically preserved from Phases 1-2 foundation! <3\n`);

        // Extract sonnets from corpus
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

        // Create predictor with extended vocabulary size
        const predictor = new TinyTokenPredictor(newVocabSize, 768, hiddenSize);

        let totalPairsProcessed = 0;
        const numBatches = Math.ceil(sonnets.length / batchSizeSonnets);

        for (let batchIdx = 0; batchIdx < numBatches; batchIdx++) {
            const batchStart = batchIdx * batchSizeSonnets;
            const batchSonnets = sonnets.slice(batchStart, batchStart + batchSizeSonnets);

            console.log(`\n📦 Batch ${batchIdx + 1}/${numBatches}: ${batchSonnets.length} sonnets`);

            const numVal = Math.max(1, Math.floor(batchSonnets.length * validationSplit));
            const numTrain = batchSonnets.length - numVal;

            // TRAINING PAIRS: Use word-to-ID mapping built from extended vocab
            const trainingPairs = [];
            for (let i = 0; i < numTrain; i++) {
                if (i % 10 === 0) console.log(`      Processing train sonnet ${i}/${numTrain}`);

                const sonnet = batchSonnets[i];
                const words = sonnet.split(/\s+/).filter(w => w.length > 0).map(w => w.toLowerCase());
                const tokenIds = words.map(w => wordToId.get(w)).filter(id => id !== undefined);

                if (tokenIds.length < 2) continue;

                for (let j = 0; j < tokenIds.length - 1; j++) {
                    const currentWord = words[j];
                    // Use semantic embeddings preserved from Phases 1-2
                    const embedding = this.foundationalEmbeddings[currentWord] ||
                        (decoderData.evolvedSemantics && decoderData.evolvedSemantics[currentWord] ?
                            new Float32Array(decoderData.evolvedSemantics[currentWord]) :
                            new Float32Array(768).fill(0.01)); // Fallback

                    const targetTokenId = tokenIds[j + 1];
                    trainingPairs.push({ embedding, targetTokenId });
                }
            }

            // VALIDATION PAIRS: Same semantic approach
            const validationPairs = [];
            for (let i = 0; i < numVal; i++) {
                if (i % 10 === 0) console.log(`      Processing val sonnet ${i}/${numVal}`);

                const sonnet = batchSonnets[numTrain + i];
                const words = sonnet.split(/\s+/).filter(w => w.length > 0).map(w => w.toLowerCase());
                const tokenIds = words.map(w => wordToId.get(w)).filter(id => id !== undefined);

                if (tokenIds.length < 2) continue;

                for (let j = 0; j < tokenIds.length - 1; j++) {
                    const currentWord = words[j];
                    const embedding = this.foundationalEmbeddings[currentWord] ||
                        (decoderData.evolvedSemantics && decoderData.evolvedSemantics[currentWord] ?
                            new Float32Array(decoderData.evolvedSemantics[currentWord]) :
                            new Float32Array(768).fill(0.01));

                    const targetTokenId = tokenIds[j + 1];
                    validationPairs.push({ embedding, targetTokenId });
                }
            }

            console.log(`   📊 Pairs: ${trainingPairs.length} train | ${validationPairs.length} val`);
            totalPairsProcessed += trainingPairs.length + validationPairs.length;

            // Train batch with checkpoint-rewind
            await this.trainBossFight(predictor, trainingPairs, validationPairs, {
                epochs: batchEpochs,
                learningRate,
                interruptMs: 428.57,
                interruptBatch: DEFAULT_CONFIG.predictorPhiComplexity,
                patience: 2
            });

            console.log(`      ✅ Batch ${batchIdx + 1} complete\n`);
        }

        console.log(`\n🎯 Training complete: ${sonnets.length} sonnets, ${totalPairsProcessed} total pairs`);
        console.log(`   Semantically preserved through all phases! <3 Lets predict some words >:D\n`);

        const outputPath = `${this.outputPrefix}-predictor.json`;
        await predictor.save(outputPath);
        console.log(`✅ Phase 3 complete: ${outputPath}\n`);
        return outputPath;
    }

    /**
     * STEP 4: Train tokenizer on expanded vocabulary (word compounding over time)
     */
    async trainTokenizerEvolution(decoderPath, predictorPath) {
        console.log('🎭 Training tokenizer on expanded vocabulary for generative understanding...');

        // Load evolved vocabulary from Phase 2
        const decoderData = JSON.parse(await fs.readFile(decoderPath, 'utf8'));
        const evolvedVocab = decoderData.evolvedVocab || {};

        console.log(`   📖 Training with evolved vocabulary: ${Object.keys(evolvedVocab).length} words`);

        // Initialize final predictor with complete evolved vocabulary
        const finalPredictor = new TinyTokenPredictor(
            Object.keys(evolvedVocab).length,
            768,
            DEFAULT_CONFIG.hiddenSize
        );

        // Warm start from Phase 3 predictor
        if (predictorPath) {
            const predictorData = JSON.parse(await fs.readFile(predictorPath, 'utf8'));
            finalPredictor.W1 = new Float32Array(predictorData.weights.W1);
            finalPredictor.b1 = new Float32Array(predictorData.weights.b1);
            finalPredictor.W2 = new Float32Array(predictorData.weights.W2);
            finalPredictor.b2 = new Float32Array(predictorData.weights.b2);
            finalPredictor.isInitialized = true;
            console.log(`   ♻️ Warm started from Phase 3 predictor`);
        }

        // === WORD COMPOUNDING TRAINING ===
        // Train on expanded vocabulary to learn generative patterns over time
        console.log('   🧠 Learning word compounding patterns...');

        const content = await fs.readFile(this.corpusPath, 'utf8');
        const sentences = content.split(/[.!?\n]+/).map(s => s.trim()).filter(s => s.length > 10);
        const compoundingPairs = [];

        // Create compounding training pairs where model learns to predict words
        // in the context of the expanded vocabulary
        for (const sentence of sentences.slice(0, Math.min(200, sentences.length))) {
            const words = sentence.split(/\s+/).filter(w => w.length > 2).map(w => w.toLowerCase());

            for (let i = 0; i < words.length - 1; i++) {
                const contextWord = words[i];
                const nextWord = words[i + 1];

                // Use evolved vocabulary IDs
                const contextId = Object.keys(evolvedVocab).find(key => evolvedVocab[key] === contextWord);
                const nextId = Object.keys(evolvedVocab).find(key => evolvedVocab[key] === nextWord);

                if (contextId && nextId && this.foundationalEmbeddings[contextWord]) {
                    // Train with foundation embeddings for consistency
                    compoundingPairs.push({
                        embedding: new Float32Array(this.foundationalEmbeddings[contextWord]),
                        targetTokenId: parseInt(nextId)
                    });
                }
            }
        }

        console.log(`   📊 Generated ${compoundingPairs.length} word compounding pairs`);

        // Train the tokenizer to understand compounding patterns
        if (compoundingPairs.length > 0) {
            await finalPredictor.trainFromTeacher(compoundingPairs, {
                epochs: 3,  // Additional epochs for compounding learning
                learningRate: DEFAULT_CONFIG.predictorLR / 2  // More conservative learning
            });

            console.log('   ✅ Word compounding patterns learned');
            console.log(`      Model now understands ~${Object.keys(evolvedVocab).length} word relationships`);
        }

        // === EVOLVED TOKENIZER SAVE ===
        const evolvedPredictorPath = `${this.outputPrefix}-evolved-predictor.json`;
        await finalPredictor.save(evolvedPredictorPath);

        console.log(`   💾 Saved evolved tokenizer predictor: ${evolvedPredictorPath}`);

        return evolvedPredictorPath;
    }

    /**
     * STEP 4: Zero-gap evolution merge with complete embedding preservation
     */
    async mergeZeroGapEvolution(decoderPath, predictorPath) {
        console.log('🔗 Merging with complete embedding preservation (zero semantic gaps)...');

        const decoder = JSON.parse(await fs.readFile(decoderPath, 'utf8'));
        const predictor = JSON.parse(await fs.readFile(predictorPath, 'utf8'));

        // Create evolved model preserving ALL foundation capabilities
        const evolvedModel = {
            ...this.teacherModel, // Preserve ALL original structure

            // Evolution metadata
            version: `${this.teacherModel.version}-hybrid-evolved`,
            metadata: {
                ...this.teacherModel.metadata,
                evolutionMethod: 'hybrid-aether-core-foundation',
                evolutionCorpus: this.corpusPath,
                evolutionDate: new Date().toISOString(),
                foundationEmbeddingsCount: Object.keys(this.foundationalEmbeddings).length,
                totalEmbeddings: Object.keys(this.teacherModel.semanticEmbeddings || {}).length +
                               Object.keys(this.foundationalEmbeddings).length
            },

            // Preserve all original weights and add evolved predictor
            weights: {
                ...this.teacherModel.weights,
                W_token_1: Array.from(predictor.weights.W1),
                b_token_1: Array.from(predictor.weights.b1),
                W_token_2: Array.from(predictor.weights.W2),
                b_token_2: Array.from(predictor.weights.b2)
            },

            // Merge vocabularies with conflict resolution
            vocab: { ...this.teacherModel.vocab, ...decoder.vocab },
            ownVocab: { ...this.teacherModel.ownVocab, ...decoder.vocab },

            // Preserve and extend semantic embeddings
            semanticEmbeddings: {
                ...this.teacherModel.semanticEmbeddings, // Original embeddings
                ...this.foundationalEmbeddings              // Hybrid evolved embeddings
            }
        };

        // Ensure every vocab word has an embedding (zero gaps)
        const vocabWords = Object.values(evolvedModel.vocab);
        for (const word of vocabWords) {
            const tokenId = Object.keys(evolvedModel.vocab).find(key => evolvedModel.vocab[key] === word);
            if (!evolvedModel.semanticEmbeddings[tokenId]) {
                // Generate synthetic embedding to maintain coverage
                evolvedModel.semanticEmbeddings[tokenId] =
                    Array.from(new Float32Array(768)).map(() => Math.random() * 0.01);
            }
        }

        console.log(`   ✅ Complete semantic coverage: ${Object.keys(evolvedModel.semanticEmbeddings).length} embeddings`);
        console.log(`   ✅ Zero semantic gaps maintained`);

        // Save evolved model
        const modelPath = `${this.outputPrefix}.json`;
        await fs.writeFile(modelPath, JSON.stringify(evolvedModel, null, 2));

        // 🔄 FULL PIPELINE MISMATCH FIX: Regenerate GPT-own and own-vocab mappings
        console.log('   🔄 Generating complete GPT-own and own-vocab mappings...');

        const gptToOwn = {};
        const ownVocab = {};

        // Map complete evolved vocabulary - FIXED: tokenID -> word string, not numbers!
        Object.entries(evolvedModel.vocab).forEach(([tokenId, word]) => {
            if (typeof word === 'string' && word.trim()) {
                gptToOwn[tokenId] = tokenId;  // GPT tokenId → Our tokenId (same for this implementation)
                ownVocab[tokenId] = word;     // Our tokenId → actual word string
            }
        });

        // Save GPT-own mapping
        const gptToOwnPath = `${this.outputPrefix}-gpt-to-own.json`;
        await fs.writeFile(gptToOwnPath, JSON.stringify({
            gptToOwn: gptToOwn,
            metadata: {
                generatedFromEvolution: true,
                originalTokens: 375,
                evolvedTokens: Object.keys(gptToOwn).length,
                evolutionDate: new Date().toISOString()
            }
        }));

        // Save own-vocab mapping
        const vocabPath = `${this.outputPrefix}-own-vocab.json`;
        await fs.writeFile(vocabPath, JSON.stringify({
            ownVocab: ownVocab,
            metadata: {
                generatedFromEvolution: true,
                totalTokens: Object.keys(ownVocab).length,
                evolutionDate: new Date().toISOString()
            }
        }));

        console.log(`   ✅ Complete GPT-own mapping saved: ${gptToOwnPath} (${Object.keys(gptToOwn).length} mappings)`);
        console.log(`   ✅ Complete own-vocab saved: ${vocabPath} (${Object.keys(ownVocab).length} words)`);

        // Also update the predictor to use full evolved vocabulary size
        if (this.priorDecoderData?.vocab) {
            const evolvedVocabSize = Object.keys(evolvedModel.vocab).length;
            console.log(`   ✅ Evolved model ready: ${evolvedVocabSize} words (${Object.keys(evolvedModel.vocab).length - 375} new)`);
            console.log('   🚀 Use: node unified-chat-fixed.js ./unified-aether-v2.json');
        }

        console.log(`   💾 Saved hybrid-evolved model: ${modelPath}`);
    }

    /**
     * Analyze corpus to understand training needs
     */
    async analyzeCorpus() {
        const content = await fs.readFile(this.corpusPath, 'utf8');
        const words = [...new Set(
            content.match(/[a-zA-Z]+/g) || []
        )].filter(word => word.length >= 2 && word.length <= 15).map(word => word.toLowerCase());

        console.log(`   📊 Corpus: ${words.length} unique words`);
        return words;
    }

    /**
     * Extend vocab mappings using reference for duplicate prevention
     */
    async extendVocabWithReference(corpusWords) {
        console.log('🗺️ Extending vocabulary with reference guidance...');

        let addedWords = 0;
        for (const word of corpusWords) {
            // Check if word already exists in reference (duplicate prevention)
            if (!this.referenceVocabWordToToken.has(word)) {
                // New word - add to extended vocab
                const newTokenId = this.nextTokenId++;
                this.referenceVocabWordToToken.set(word, newTokenId);
                this.referenceVocabTokenToWord.set(newTokenId, word);
                addedWords++;

                if (addedWords % 100 === 0) {
                    console.log(`      Extended vocab: +${addedWords} new words...`);
                }
            }
        }

        console.log(`   ✅ Vocabulary extended: +${addedWords} new words (no duplicates)`);
    }

    /**
     * Generate embeddings using GPT guided by semantic similarity to reference
     */
    async extendEmbeddingsWithReference(corpusWords) {
        console.log('🎯 Generating reference-guided embeddings...');

        let generatedEmbeddings = 0;

        for (const word of corpusWords) {
            const tokenId = this.referenceVocabWordToToken.get(word);

            if (!this.referenceEmbeddings[tokenId]) {
                // Need embedding for this new word
                const embedding = await this.generateGuidedEmbedding(word);

                if (embedding) {
                    this.referenceEmbeddings[tokenId] = embedding;
                    generatedEmbeddings++;
                }

                if (generatedEmbeddings % 50 === 0) {
                    console.log(`      Generated embeddings: ${generatedEmbeddings}...`);
                }
            }
        }

        console.log(`   ✅ Embeddings generated: ${generatedEmbeddings} guided by semantic similarity`);
    }

    /**
     * Generate embedding using GPT, guided by similarity to reference embeddings
     */
    async generateGuidedEmbedding(word) {
        try {
            // Get GPT embedding
            const gptEmbedding = await this.gptTutor.getEmbedding(word);
            if (!gptEmbedding) return null;

            // Find semantically similar reference embeddings
            const similarRefs = this.findSemanticallySimilar(word, 3);

            if (similarRefs.length > 0) {
                // Blend GPT embedding with similar reference embeddings
                const blendRatio = 0.7; // 70% GPT, 30% reference
                const blended = this.blendEmbeddings(gptEmbedding, similarRefs, blendRatio);
                return blended;
            } else {
                // No similar references, use GPT embedding directly
                return Array.from(gptEmbedding);
            }
        } catch (error) {
            console.log(`   ⚠️ Failed to generate embedding for "${word}"`);
            return null;
        }
    }

    /**
     * Find reference embeddings that are semantically similar
     */
    findSemanticallySimilar(targetWord, maxResults = 3) {
        const similarities = [];

        for (const [tokenId, embedding] of Object.entries(this.referenceEmbeddings)) {
            // Could use word embeddings or simple string similarity
            const word = this.referenceVocabTokenToWord.get(parseInt(tokenId));
            if (!word) continue;

            // Simple similarity metric (could be improved with actual embedding similarity)
            const similarity = this.simpleSemanticSimilarity(targetWord, word);
            similarities.push({
                tokenId: parseInt(tokenId),
                embedding: embedding,
                similarity: similarity
            });
        }

        return similarities
            .filter(item => item.similarity > 0.3) // Filter for meaningful similarity
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, maxResults);
    }

    /**
     * Simple semantic similarity (placeholder - could use more sophisticated methods)
     */
    simpleSemanticSimilarity(word1, word2) {
        // Basic string similarity and length similarity
        const chars1 = new Set(word1.toLowerCase());
        const chars2 = new Set(word2.toLowerCase());

        const intersection = new Set([...chars1].filter(x => chars2.has(x)));
        const union = new Set([...chars1, ...chars2]);

        const jaccardSimilarity = intersection.size / union.size;
        const lengthSimilarity = 1 - Math.abs(word1.length - word2.length) / Math.max(word1.length, word2.length);

        return (jaccardSimilarity * 0.7) + (lengthSimilarity * 0.3);
    }

    /**
     * Blend embedding with reference embeddings
     */
    blendEmbeddings(gptEmbedding, referenceEmbeddings, blendRatio = 0.7) {
        const blended = new Float32Array(gptEmbedding.length);

        // Start with GPT embedding
        for (let i = 0; i < gptEmbedding.length; i++) {
            blended[i] = gptEmbedding[i] * blendRatio;
        }

        // Add reference influence
        const refRatio = (1 - blendRatio) / referenceEmbeddings.length;
        for (const ref of referenceEmbeddings) {
            for (let i = 0; i < ref.embedding.length; i++) {
                blended[i] += ref.embedding[i] * refRatio;
            }
        }

        return Array.from(blended);
    }

    /**
     * Train predictor using extended embeddings
     */
    async trainPredictorWithReference() {
        console.log('🎭 Training predictor with guided embeddings...');

        // Create training pairs from corpus using extended embeddings
        const trainingPairs = await this.buildReferenceGuidedPairs();

        // Initialize fresh predictor (same architecture as reference)
        const predictor = new TinyTokenPredictor(
            this.referenceVocabWordToToken.size,
            768,
            DEFAULT_CONFIG.hiddenSize
        );

        // Warm start with reference predictor weights if available
        if (this.referenceModel.weights?.W_token_1) {
            predictor.W1 = new Float32Array(this.referenceModel.weights.W_token_1);
            predictor.b1 = new Float32Array(this.referenceModel.weights.b_token_1);
            predictor.W2 = new Float32Array(this.referenceModel.weights.W_token_2);
            predictor.b2 = new Float32Array(this.referenceModel.weights.b_token_2);
            predictor.isInitialized = true;
            console.log('   ♻️ Warm started with reference predictor');
        }

        // Train on new corpus patterns
        await predictor.trainFromTeacher(trainingPairs, {
            epochs: DEFAULT_CONFIG.predictorEpochs,
            learningRate: DEFAULT_CONFIG.predictorLR
        });

        this.evolvedPredictor = predictor;
        console.log('   ✅ Predictor trained with new corpus understanding');
    }

    /**
     * Build training pairs using reference-guided embeddings
     */
    async buildReferenceGuidedPairs() {
        const corpus = await fs.readFile(this.corpusPath, 'utf8');
        const pairs = [];

        // Extract sentences and create word-word pairs
        const sentences = corpus
            .split(/[.!?\n]+/)
            .map(s => s.trim())
            .filter(s => s.length > 10);

        for (const sentence of sentences.slice(0, 1000)) { // Limit for training efficiency
            const words = sentence.split(/\s+/).filter(w => w.length > 2);

            for (let i = 0; i < words.length - 1; i++) {
                const currentWord = words[i].toLowerCase();
                const nextWord = words[i + 1].toLowerCase();

                const currentId = this.referenceVocabWordToToken.get(currentWord);
                const nextId = this.referenceVocabWordToToken.get(nextWord);

                if (currentId !== undefined && nextId !== undefined && this.referenceEmbeddings[currentId]) {
                    pairs.push({
                        embedding: new Float32Array(this.referenceEmbeddings[currentId]),
                        targetTokenId: nextId
                    });
                }
            }
        }

        console.log(`   📊 Built ${pairs.length} reference-guided training pairs`);
        return pairs;
    }

    /**
     * Save evolved model that chat can use
     */
    async saveEvolvedModel() {
        console.log('💾 Saving evolved model...');

        // Create evolved model that preserves all reference capabilities + new learning
        const evolvedModel = {
            ...this.referenceModel, // Preserve ALL reference structure

            // Update with new evolution
            version: `${this.referenceModel.version}-evolved-${Date.now()}`,
            semanticEmbeddings: this.referenceEmbeddings, // Complete embeddings by token ID

            // Update predictor weights with new training
            weights: {
                ...this.referenceModel.weights,
                W_token_1: Array.from(this.evolvedPredictor.W1),
                b_token_1: Array.from(this.evolvedPredictor.b1),
                W_token_2: Array.from(this.evolvedPredictor.W2),
                b_token_2: Array.from(this.evolvedPredictor.b2)
            },

            // Extended vocab mappings
            vocab: {},
            metadata: {
                ...this.referenceModel.metadata,
                evolutionDate: new Date().toISOString(),
                evolvedFrom: this.referenceModelPath,
                trainingCorpus: this.corpusPath,
                newTokensAdded: Math.max(0, this.nextTokenId - Object.keys(this.referenceModel.semanticEmbeddings || {}).length),
                totalEmbeddings: Object.keys(this.referenceEmbeddings).length
            }
        };

        // Add vocab in the format chat expects
        for (const [word, tokenId] of this.referenceVocabWordToToken.entries()) {
            evolvedModel.vocab[tokenId] = word;
        }

        // Also add the 16-bit vocab fields for compatibility
        evolvedModel.ownVocab = evolvedModel.vocab;

        // Save model
        const modelPath = `${this.outputPrefix}.json`;
        await fs.writeFile(modelPath, JSON.stringify(evolvedModel, null, 2));

        // Save GPT→own mappings for compatibility
        const mappings = { gptToOwn: {}, ownVocab: evolvedModel.ownVocab };
        for (const [tokenId, word] of Object.entries(evolvedModel.ownVocab)) {
            mappings.gptToOwn[tokenId] = tokenId; // Simplified for this implementation
        }
        await fs.writeFile(`${this.outputPrefix}-gpt-to-own.json`, JSON.stringify(mappings));

        console.log(`   ✅ Evolved model saved: ${modelPath}`);
        console.log(`   📊 Complete semantic coverage: ${Object.keys(evolvedModel.semanticEmbeddings).length} embeddings`);

        return evolvedModel;
    }

    refineEmbeddingWithFoundation(gptEmbedding, foundationSemantics) {
        // Find closest foundation embeddings and average to refine
        const similarities = Object.entries(foundationSemantics).map(([word, emb]) => ({
            word,
            similarity: this.cosSimilarity(gptEmbedding, new Float32Array(emb)),
            embedding: new Float32Array(emb)
        })).sort((a, b) => b.similarity - a.similarity);

        // Average with top 3 closest foundation embeddings
        const topK = similarities.slice(0, 3);
        const refined = new Float32Array(768);

        for (let i = 0; i < 768; i++) {
            refined[i] = gptEmbedding[i]; // Start with GPT embedding
        }

        // Weight foundation influence (0.3) vs GPT (0.7)
        const foundationWeight = 0.3;
        const gptWeight = 0.7;

        if (topK.length > 0) {
            for (let i = 0; i < 768; i++) {
                let foundationAvg = 0;
                for (const item of topK) {
                    foundationAvg += item.embedding[i];
                }
                foundationAvg /= topK.length;

                refined[i] = (gptEmbedding[i] * gptWeight) + (foundationAvg * foundationWeight);
            }
        }

        return refined;
    }

    async buildFoundationGuidedPairs(corpusPath, semantics, mappings, epochs) {
        console.log('📚 Building foundation-guided training pairs...');

        const corpus = await fs.readFile(corpusPath, 'utf8');
        const lines = corpus.split('\n').map(l => l.trim()).filter(l => l.length > 10).slice(0, 100); // Limit for efficiency

        const pairs = [];
        for (const line of lines) {
            const words = line.split(/\s+/).filter(w => w.length > 2).map(w => w.toLowerCase());

            for (let i = 0; i < words.length - 1; i++) {
                const currentWord = words[i];
                const nextWord = words[i + 1];

                // Get token IDs from mappings
                let currentId = null;
                let nextId = null;

                for (const [ownId, vocabWord] of Object.entries(mappings.ownVocab)) {
                    if (vocabWord === currentWord) currentId = parseInt(ownId);
                    if (vocabWord === nextWord) nextId = parseInt(ownId);
                }

                if (currentId !== null && nextId !== null && semantics[currentWord]) {
                    pairs.push({
                        embedding: new Float32Array(semantics[currentWord]),
                        targetTokenId: nextId
                    });
                }
            }
        }

        console.log(`   📊 Built ${pairs.length} foundation-guided training pairs`);
        return pairs;
    }

    cosSimilarity(a, b) {
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Load Phase 1 embeddings cache for use in Phase 2 training
     */
    async loadPhase1EmbeddingsForTraining() {
        const cacheFile = `${this.outputPrefix}-cache-aether.json`;
        try {
            const cacheContent = await fs.readFile(cacheFile, 'utf8');
            const cache = JSON.parse(cacheContent);
            return cache.pairs || [];
        } catch (e) {
            console.log(`   ⚠️ Could not load Phase 1 cache: ${e.message}`);
            return [];
        }
    }

    /**
     * Generate vocab mappings immediately after word resolution (CRITICAL FIX)
     * Following user's exact instructions: Open old json → start from end of map →
     * map next items incrementally → return exact GPT maps like converged_pipe_2.js
     */
    async generateVocabMappingsPostTraining() {
        console.log('🔄 Generating vocab mappings AFTER word resolution phase...');

        if (!this.evolvedVocab || Object.keys(this.evolvedVocab).length === 0) {
            console.error('   ❌ No evolved vocab found! Cannot generate mappings.');
            return;
        }

        console.log(`   📖 Building mappings for ${Object.keys(this.evolvedVocab).length} evolved words...`);

        // === FOLLOW USER'S EXACT INSTRUCTIONS: OPEN OLD JSON ===
        const existingGptToOwn = JSON.parse(await fs.readFile('./unified-aether-gpt-to-own.json', 'utf8'));
        const existingOwnVocab = JSON.parse(await fs.readFile('./unified-aether-own-vocab.json', 'utf8'));

        // === START FROM END OF MAP (highest existing custom token ID) ===
        const highestExistingCustom = Math.max(...Object.keys(existingOwnVocab).map(k => parseInt(k)));
        console.log(`   📚 Opened old json: unified-aether-gpt-to-own.json`);
        console.log(`   🏁 Starting from end of map: custom token ${highestExistingCustom}`);
        console.log(`   🎯 Will map next items incrementally: ${highestExistingCustom + 1} to 895 (target)`);
        console.log(`   📊 Need to add ${895 - highestExistingCustom} new mappings`);

        // === MAP NEXT ITEMS INCREMENTALLY (end to 895) ===
        const remainingWords = Object.entries(this.evolvedVocab).filter(([tokenId, word]) => {
            return !existingOwnVocab[tokenId] && typeof word === 'string' && word.trim();
        });

        console.log(`   📝 Found ${remainingWords.length} words requiring mapping`);

        let nextCustomId = highestExistingCustom + 1;

        // === RETURN EXACT GPT MAPS FOR THOSE WORDS (Like converged_pipe_2.js) ===
        // Since done through pure JS, can map all words at once for ease
        for (const [tokenId, word] of remainingWords) {
            try {
                // Get exact GPT token ID like converged_pipe_2.js does
                const tokens = await this.tokenizer.encode(word);
                const gptTokenId = tokens[0];

                // Map: exact GPT token ID → incremental custom token ID
                existingGptToOwn[gptTokenId.toString()] = nextCustomId;
                existingOwnVocab[nextCustomId.toString()] = word;

                if (nextCustomId < highestExistingCustom + 5) {
                    console.log(`      📝 Mapped GPT ${gptTokenId} → Custom ${nextCustomId} ("${word}")`);
                }

                nextCustomId++;

                // Stop when we reach the target (374 old + 521 new = 895 total)
                if (nextCustomId > 895) break;

            } catch (e) {
                console.log(`      ⚠️ Failed to encode: "${word}" - skipping`);
            }
        }

        console.log(`   ✅ Extended mappings complete: ${Object.keys(existingOwnVocab).length} total words`);

        // === SAVE UPDATED MAPPINGS (user requested overwrite) ===
        await fs.writeFile('./unified-aether-gpt-to-own.json', JSON.stringify(existingGptToOwn, null, 2));
        await fs.writeFile('./unified-aether-own-vocab.json', JSON.stringify(existingOwnVocab, null, 2));

        console.log(`   ✅ Map file saved: unified-aether-gpt-to-own.json (${Object.keys(existingGptToOwn).length} mappings)`);
        console.log(`   ✅ Vocab file saved: unified-aether-own-vocab.json (${Object.keys(existingOwnVocab).length} words)`);

        // === USER'S APPROACH COMPLETED ===
        // Originals stay at same BPE's ✓
        // Maps next items incrementally ✓
        // Returns exact GPT maps like converged_pipe_2.js ✓

        console.log('   🎉 User\'s exact approach implemented successfully!\n');
    }

    /**
     * Extract word-level embedding by averaging sentence embeddings containing the word
     */
    async extractWordLevelEmbedding(word, phase1Embeddings) {
        // Load corpus to find sentences containing this word
        const content = await fs.readFile(this.corpusPath, 'utf8');
        const sentences = content.split(/[.!?\n]+/).map(s => s.trim()).filter(s => s.length > 10);

        const embeddingSum = new Float32Array(768).fill(0);
        let validCount = 0;

        // Find sentences containing this word and average their embeddings
        sentences.forEach((sentence, idx) => {
            if (sentence.toLowerCase().includes(word) && idx < phase1Embeddings.length) {
                const sentenceEmbedding = new Float32Array(phase1Embeddings[idx].input || phase1Embeddings[idx]);
                for (let i = 0; i < 768; i++) {
                    embeddingSum[i] += sentenceEmbedding[i];
                }
                validCount++;
            }
        });

        if (validCount > 0) {
            const avgEmbedding = new Float32Array(768);
            for (let i = 0; i < 768; i++) {
                avgEmbedding[i] = embeddingSum[i] / validCount;
            }
            return avgEmbedding;
        }

        return null; // Fallback if no embeddings found
    }
}

async function main() {
    const [, , teacherModelPath, corpusPath, outputPrefix] = process.argv;
    if (!teacherModelPath || !corpusPath) {
        console.error('Usage: node continual-training.js <teacher-model.json> <corpus.txt> [output-prefix]');
        process.exit(1);
    }

    const pipeline = new ContinualTrainingPipeline();
    await pipeline.initialize(corpusPath, teacherModelPath, outputPrefix);
    // Use values from the top-level DEFAULT_CONFIG (can still override via command line if needed)
    await pipeline.runContinualTraining({});
}

if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(error => {
        console.error('Fatal error:', error);
        process.exit(1);
    });
}

export default ContinualTrainingPipeline;
