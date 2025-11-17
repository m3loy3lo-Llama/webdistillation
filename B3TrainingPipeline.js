/**
 * B3 TRAINING PIPELINE
 * Complete system for training the tiny mystical model from teacher
 *
 * Usage:
 * 1. Drop mystical_corpus.txt in project root (one sentence per line)
 * 2. Run: node trainTinyModel.js
 * 3. Let it learn for hours/days
 * 4. Get tiny-mystical-model.json (~800KB) that mimics DistilGPT-2
 */


import { env, pipeline } from "@huggingface/transformers";
        // Dynamic imports for transformers
// use ES6 style import syntax (recommended)
//import * as ort from '@xenova/ort';
env.backends.onnx.executionProviders = ['cpu'];

// import * as ort from 'onnxruntime-node';
import TinyMysticalModel from './B3TinyMysticalModel.js';
import B3EmbeddingExtractor from './B3EmbeddingExtractor.js';
import fs from 'fs/promises';
import path from 'path';

// Configure environment for local transformers use (same as xenovaLearningBot)
env.localModelPath = process.cwd();
env.allowRemoteModels = false;
env.backends.onnx.wasm.wasmPaths = "./Model/model.onnx"


class B3TrainingPipeline {
    constructor() {
        this.teacherPipeline = null;
        this.extractor = null;
        this.tinyModel = null;
        this.trainingData = [];
    }

    /**
     * Initialize teacher model (Mia-1B)
     */
    async initializeTeacher() {
        console.log('ðŸŽ“ Loading teacher model (Mia-1B)...');

        // Model loads with CPU device configuration
        this.teacherPipeline = await pipeline(
            'text-generation',
            'Model',
            {
                device: 'cpu'
            }
        );

        this.extractor = new B3EmbeddingExtractor(this.teacherPipeline);

        console.log('âœ… Teacher model loaded');
    }

    /**
     * Load training corpus from file
     * Expected format: One sentence per line
     */
    async loadCorpus(filepath) {
        console.log(`ðŸ“š Loading training corpus from ${filepath}...`);

        const content = await fs.readFile(filepath, 'utf8');
        const sentences = content
            .split('\n')
            .map(s => s.trim())
            .filter(s => s.length > 10) // Filter very short lines
            .filter(s => !s.startsWith('#')) // Allow comments
            .filter(s => !s.startsWith('//')); // Allow comments

        console.log(`   Loaded ${sentences.length} sentences`);

        return sentences;
    }

    /**
     * Prepare training data: extract embeddings from teacher
     */
    async prepareTrainingData(sentences, options = {}) {
        const {
            batchSize = 8,
            saveIntermediateAt = 100,
            cacheFile = './training_cache.json'
        } = options;

        console.log(`ðŸ”¬ Extracting embeddings from teacher model...`);
        console.log(`   Processing ${sentences.length} sentences in batches of ${batchSize}`);

        const embeddingPairs = [];

        for (let i = 0; i < sentences.length; i += batchSize) {
            const batch = sentences.slice(i, Math.min(i + batchSize, sentences.length));
            const batchNum = Math.floor(i / batchSize) + 1;
            const totalBatches = Math.ceil(sentences.length / batchSize);

            console.log(`   Batch ${batchNum}/${totalBatches}: Processing ${batch.length} sentences...`);

            for (const sentence of batch) {
                try {
                    // Get teacher's embedding for this sentence
                    const embedding = await this.extractor.getEmbedding(sentence);

                    // Training pair: input = embedding, target = embedding
                    // (Autoencoder-style: learn to reconstruct teacher's output)
                    embeddingPairs.push({
                        input: embedding,
                        target: embedding,
                        sentence: sentence // Keep for debugging
                    });
                } catch (error) {
                    console.warn(`   âš ï¸ Failed to process: "${sentence.substring(0, 50)}..."`);
                }
            }

            // Save intermediate results periodically
            if (embeddingPairs.length % saveIntermediateAt === 0) {
                console.log(`   ðŸ’¾ Saving intermediate cache (${embeddingPairs.length} pairs)...`);
                await this.saveTrainingCache(embeddingPairs, cacheFile);
            }

            // Small delay between batches
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        console.log(`âœ… Extracted ${embeddingPairs.length} embedding pairs`);

        // Save final cache
        await this.saveTrainingCache(embeddingPairs, cacheFile);

        this.trainingData = embeddingPairs;
        return embeddingPairs;
    }

    /**
     * Save training cache to resume later
     */
    async saveTrainingCache(embeddingPairs, filepath) {
        const cache = {
            version: '1.0',
            timestamp: new Date().toISOString(),
            count: embeddingPairs.length,
            // Only save embeddings, not sentences (save space)
            pairs: embeddingPairs.map(p => ({
                input: Array.from(p.input),
                target: Array.from(p.target)
            }))
        };

        await fs.writeFile(filepath, JSON.stringify(cache));
        const stats = await fs.stat(filepath);
        console.log(`   Cache saved: ${(stats.size / 1024 / 1024).toFixed(1)}MB`);
    }

    /**
     * Load training cache to skip re-extraction
     */
    async loadTrainingCache(filepath) {
        try {
            console.log(`ðŸ“‚ Loading training cache from ${filepath}...`);

            const cache = JSON.parse(await fs.readFile(filepath, 'utf8'));

            const embeddingPairs = cache.pairs.map(p => ({
                input: new Float32Array(p.input),
                target: new Float32Array(p.target)
            }));

            console.log(`âœ… Loaded ${embeddingPairs.length} cached embedding pairs`);

            this.trainingData = embeddingPairs;
            return embeddingPairs;

        } catch (error) {
            console.warn(`âš ï¸ Could not load cache: ${error.message}`);
            return null;
        }
    }

    /**
     * Train the tiny model on prepared data
     */
    async trainTinyModel(embeddingPairs, options = {}) {
        const {
            embeddingSize = 768, // Match GPT-2's hidden size
            hiddenSize = 304,
            epochs = 1000,
            learningRate = 0.002,
            validationSplit = 0.1,
            logInterval = 1,
            saveCheckpointsAt = './checkpoints'
        } = options;

        console.log(`ðŸ§  Training tiny mystical model...`);

        // Create model with appropriate embedding size
        this.tinyModel = new TinyMysticalModel(embeddingSize, hiddenSize);

        // Train
        const history = await this.tinyModel.trainFromEmbeddings(embeddingPairs, {
            epochs,
            learningRate,
            validationSplit,
            logInterval
        });

        console.log(`âœ… Training complete!`);
        console.log(`   Final train loss: ${history[history.length - 1].trainLoss.toFixed(6)}`);
        console.log(`   Final val loss: ${history[history.length - 1].valLoss.toFixed(6)}`);

        return this.tinyModel;
    }

    /**
     * Full training pipeline: corpus â†’ embeddings â†’ training â†’ save
     */
    async runFullPipeline(corpusPath, outputPath, options = {}) {
        const {
            useCachedEmbeddings = true,
            cacheFile = './training_cache.json',
            hiddenSize = 128,
            epochs = 100,
            learningRate = 0.001,
            batchSize = 8
        } = options;

        console.log('ðŸš€ FULL TRAINING PIPELINE START');
        console.log('================================\n');

        // Step 1: Initialize teacher
        if (!this.teacherPipeline) {
            await this.initializeTeacher();
        }

        // Step 2: Prepare training data
        let embeddingPairs;

        if (useCachedEmbeddings) {
            embeddingPairs = await this.loadTrainingCache(cacheFile);
        }

        if (!embeddingPairs) {
            const sentences = await this.loadCorpus(corpusPath);
            embeddingPairs = await this.prepareTrainingData(sentences, {
                batchSize,
                cacheFile
            });
        }

        // Step 3: Train tiny model
        const tinyModel = await this.trainTinyModel(embeddingPairs, {
            hiddenSize,
            epochs,
            learningRate
        });

        // Step 4: Save trained model
        await tinyModel.save(outputPath);

        console.log('\n================================');
        console.log('ðŸŽ‰ TRAINING PIPELINE COMPLETE!');
        console.log(`   Model saved to: ${outputPath}`);
        console.log(`   Parameters: ${tinyModel.countParams().toLocaleString()}`);

        const stats = await fs.stat(outputPath);
        console.log(`   File size: ${(stats.size / 1024).toFixed(1)}KB`);
        console.log('\nðŸ§™â€â™‚ï¸ Your tiny mystical model is ready!');

        return tinyModel;
    }

    /**
     * Test the trained model
     */
    async testModel(modelPath, testSentences) {
        console.log(`ðŸ§ª Testing model from ${modelPath}...`);

        // Load model
        const model = new TinyMysticalModel();
        await model.load(modelPath);

        // Test on sentences
        for (const sentence of testSentences) {
            const teacherEmbedding = await this.extractor.getEmbedding(sentence);
            const { output: studentOutput } = model.forward(teacherEmbedding);

            // Calculate similarity (cosine similarity)
            let dotProduct = 0;
            let teacherMag = 0;
            let studentMag = 0;

            for (let i = 0; i < teacherEmbedding.length; i++) {
                dotProduct += teacherEmbedding[i] * studentOutput[i];
                teacherMag += teacherEmbedding[i] * teacherEmbedding[i];
                studentMag += studentOutput[i] * studentOutput[i];
            }

            const similarity = dotProduct / (Math.sqrt(teacherMag) * Math.sqrt(studentMag));

            console.log(`   "${sentence.substring(0, 50)}..."`);
            console.log(`   Similarity: ${(similarity * 100).toFixed(2)}%`);
        }
    }
}

export default B3TrainingPipeline;

// Standalone training script
if (import.meta.url === `file://${process.argv[1]}`) {
    const pipeline = new B3TrainingPipeline();

    await pipeline.runFullPipeline(
        './mystical_corpus.txt',
        './tiny-mystical-model.json',
        {
            epochs: 1000,
            learningRate: 0.02,
            batchSize: 1,
            hiddenSize: 304
        }
    );
}
