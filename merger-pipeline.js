#!/usr/bin/env node
import fs from 'fs/promises';
import path from 'path';
import { mergePriorVersions, mergeSimpleCheckpoints } from './model-merger.js';
import { env, pipeline } from '@huggingface/transformers';
import TinyTokenPredictor from './TinyTokenPredictor.js';
import GrammarInductor from './GrammarPipeline.js';
import TinyLatentProcessor from './TinyLatentProcessor.js';
import B3EmbeddingExtractor from './B3EmbeddingExtractor.js';
import { AetherVocab } from './vocab-resolver.js';

env.localModelPath = process.cwd();
env.allowRemoteModels = false;

async function main() {
    const [, , outputPrefix, corpusPath, numCpStr, inputPrefix] = process.argv;
    const numCheckpoints = parseInt(numCpStr) || 5;
    const searchPrefix = inputPrefix || 'unified-aether-v';

    if (!outputPrefix || !corpusPath) {
        console.log('Usage: node merger-pipeline.js <output-prefix> <corpus-path> [num-checkpoints] [input-prefix]');
        console.log('Example: node merger-pipeline.js merged-aether-v1 corpus.txt 5 unified-aether-v');
        process.exit(1);
    }

    console.log('--- Æther Merger Pipeline ---');
    console.log(`Target: ${outputPrefix}`);
    console.log(`Source Pattern: ${searchPrefix}*.json`);
    console.log(`Corpus: ${corpusPath}`);
    console.log(`Merging last ${numCheckpoints} checkpoints...\n`);

    // --- 1. Auto-detect Checkpoints ---
    const files = await fs.readdir('.');
    const checkpoints = files
        .filter(f => f.startsWith(searchPrefix) && f.endsWith('.json') && f !== `${outputPrefix}.json`)
        .sort((a, b) => {
            // Extract the LAST number found in the filename (usually the version)
            const numbersA = a.match(/\d+/g);
            const numbersB = b.match(/\d+/g);
            const vA = numbersA ? parseInt(numbersA[numbersA.length - 1]) : 0;
            const vB = numbersB ? parseInt(numbersB[numbersB.length - 1]) : 0;
            return vB - vA; // Newest first
        })
        .slice(0, numCheckpoints)
        .reverse(); // Standard order for merge (template is latest)

    if (checkpoints.length === 0) {
        console.error(`❌ No checkpoints found matching prefix: ${searchPrefix}`);
        process.exit(1);
    }

    console.log('Found checkpoints:');
    checkpoints.forEach(cp => console.log(`  - ${cp}`));

    // --- 2. Perform Merges ---
    console.log('\nMerging versions...');
    const merged = await mergePriorVersions(checkpoints, { allowVocabExpansion: true });

    // Merging Sidecars (Latent, Grammar, TextGen, etc.)
    const sidecarTypes = ['latent', 'grammar', 'textgen', 'predictor', 'aether-core', 'decoder'];
    const mergedSidecars = {};

    for (const type of sidecarTypes) {
        const sidecarPaths = checkpoints.map(cp => cp.replace('.json', `-${type}.json`));
        console.log(`Merging ${type} sidecars...`);
        const mergedSidecar = await mergeSimpleCheckpoints(sidecarPaths);
        if (mergedSidecar) {
            mergedSidecars[type] = mergedSidecar;
        }
    }

    // --- 3. Fine-Tune on Mirror Samples ---
    console.log('\n--- Fine-Tuning for Semantic Alignment ---');
    console.log('🔧 Loading GPT model and extractor...');
    const gen = await pipeline('text-generation', 'Models', { device: 'cpu' });
    const tokenizer = gen.tokenizer;
    const extractor = new B3EmbeddingExtractor(gen);
    console.log('   ✓ GPT Model & Tokenizer ready');

    console.log('Preparing mirror-samples (GPT Grounded)...');

    // Load Grammar for training pairs
    const grammarPath = checkpoints[checkpoints.length - 1].replace('.json', '-grammar.json');
    const grammar = new GrammarInductor();
    try {
        await grammar.load(grammarPath);
    } catch (e) {
        console.warn('⚠️  Could not load grammar, using zeros for context.');
    }

    // 🧠 LATENT GROUNDING: Load Latent Processor for embedding refinement
    const latentPath = checkpoints[checkpoints.length - 1].replace('.json', '-latent.json');
    const latentProc = new TinyLatentProcessor(1280, 256);
    try {
        await latentProc.load(latentPath);
        console.log(`   🧠 Latent Processor loaded for grounding (${latentPath})`);
    } catch (e) {
        console.warn(`   ⚠️ Could not load latent processor: ${e.message}`);
    }

    // Load merged model into Predictor
    const predictor = new TinyTokenPredictor(
        merged.architecture.vocabSize,
        merged.architecture.embeddingSize + grammar.numClusters,
        merged.architecture.tokenHiddenSize || 512
    );
    predictor.W1 = new Float32Array(merged.weights.W_token_1);
    predictor.b1 = new Float32Array(merged.weights.b_token_1);
    predictor.W2 = new Float32Array(merged.weights.W_token_2);
    predictor.b2 = new Float32Array(merged.weights.b_token_2);

    // Generate Mirror Training Pairs
    const corpus = await fs.readFile(corpusPath, 'utf8');
    const lines = corpus.split('\n').filter(l => l.trim().length > 10).slice(0, 500);
    const trainingPairs = [];
    const wordToId = new Map(Object.entries(merged.ownVocab).map(([id, w]) => [w, parseInt(id)]));

    for (const line of lines) {
        const tokenIds = tokenizer.encode(line);
        if (tokenIds.length < 2) continue;

        // Map GPT tokens to our 16-bit ownVocab via merged.vocab
        const internalIds = tokenIds.map(gptId => merged.vocab[gptId]).filter(id => id !== undefined);
        if (internalIds.length < 2) continue;

        // Extract REAL GPT hidden state for grounding (story space alignment)
        const groundTruthEmbedding = await extractor.getEmbedding(line, true); // 1280 dim

        // Process reversed sequence (Mirror Mode)
        const reversedIds = [...internalIds].reverse();

        for (let i = 0; i < reversedIds.length - 1; i++) {
            const currentId = reversedIds[i];
            const nextId = reversedIds[i + 1];

            const currentWord = merged.ownVocab[currentId];

            if (currentId !== undefined && nextId !== undefined) {
                // Ground through Latent if available
                let groundedEmbedding = groundTruthEmbedding;
                if (latentProc.initialized) {
                    const thought = latentProc.forward(groundTruthEmbedding, 0, 0.05);
                    groundedEmbedding = thought.reconstruction;
                }

                const currentCluster = grammar.getClusterForWord(currentWord) || 0;
                const grammarContext = grammar.getNextClusterProbs(currentCluster);

                // Input size = 1280 (Semantic/Latent) + 64 (Grammar) = 1344
                const combinedInput = new Float32Array(1344);
                combinedInput.set(groundedEmbedding, 0);
                combinedInput.set(grammarContext, 1280);

                // TEACHER: Distribution from current predictor
                const teacherOutput = predictor.forward(combinedInput);
                const softTarget = new Float32Array(merged.architecture.vocabSize);

                // Mirror Mix: 40% Predicted + 60% Ground-Truth (GPT)
                const phase3Weight = 0.4;
                const gptWeight = 0.6;

                for (let k = 0; k < merged.architecture.vocabSize; k++) {
                    softTarget[k] = teacherOutput.probs[k] * phase3Weight;
                }
                softTarget[nextId] += gptWeight;

                // Normalize for Soft-Target entropy stability
                let sum = 0;
                for (let k = 0; k < merged.architecture.vocabSize; k++) sum += softTarget[k];
                if (sum > 0) {
                    for (let k = 0; k < merged.architecture.vocabSize; k++) softTarget[k] /= sum;
                }

                trainingPairs.push({
                    embedding: combinedInput,
                    targetTokenId: nextId,
                    softTarget: softTarget
                });
            }
        }
    }

    if (trainingPairs.length > 0) {
        console.log(`\nGenerated ${trainingPairs.length} pairs. Training Classifier...`);
        await predictor.trainWithSoftTargets(trainingPairs, {
            epochs: 5,
            learningRate: 0.0003,
            validationSplit: 0.1
        });

        // Train TextGen if available
        if (merged.weights.W_textgen_1) {
            console.log('Training TextGen weights...');
            const textgen = new TinyTokenPredictor(merged.architecture.vocabSize, 1344, 512);
            textgen.W1 = new Float32Array(merged.weights.W_textgen_1);
            textgen.b1 = new Float32Array(merged.weights.b_textgen_1);
            textgen.W2 = new Float32Array(merged.weights.W_textgen_2);
            textgen.b2 = new Float32Array(merged.weights.b_textgen_2);

            await textgen.trainWithSoftTargets(trainingPairs, {
                epochs: 5,
                learningRate: 0.0003,
                validationSplit: 0.1
            });

            merged.weights.W_textgen_1 = Array.from(textgen.W1);
            merged.weights.b_textgen_1 = Array.from(textgen.b1);
            merged.weights.W_textgen_2 = Array.from(textgen.W2);
            merged.weights.b_textgen_2 = Array.from(textgen.b2);
        }
    }

    // Update weights in merged object
    merged.weights.W_token_1 = Array.from(predictor.W1);
    merged.weights.b_token_1 = Array.from(predictor.b1);
    merged.weights.W_token_2 = Array.from(predictor.W2);
    merged.weights.b_token_2 = Array.from(predictor.b2);

    // --- 4. Save Final Merged Model & Sidecars ---
    const finalPath = `${outputPrefix}.json`;
    await fs.writeFile(finalPath, JSON.stringify(merged, null, 2));

    // Determine sidecar names based on chat script expectations:
    // Chat expects: ./unified-aether-vN-own-vocab.json
    // We'll save to both the specific prefix-based name AND the unified-aether-vN name if possible.
    const versionMatch = outputPrefix.match(/v(\d+)/);
    const versionSuffix = versionMatch ? `-v${versionMatch[1]}` : '';

    const vocabSidecarPath = `${outputPrefix}-own-vocab.json`;
    const compatVocabPath = `./unified-aether${versionSuffix}-own-vocab.json`;

    const vocabData = JSON.stringify(merged.ownVocab, null, 2);
    await fs.writeFile(vocabSidecarPath, vocabData);
    if (vocabSidecarPath !== compatVocabPath) {
        await fs.writeFile(compatVocabPath, vocabData);
        console.log(`✅ Vocab saved (with chat-compatibility): ${compatVocabPath}`);
    } else {
        console.log(`✅ Vocab saved: ${vocabSidecarPath}`);
    }

    // Save grammar sidecar if available
    const finalGrammar = mergedSidecars.grammar || (merged.weights.grammar_transitions ? {
        centroids: merged.weights.grammar_centroids,
        transitions: merged.weights.grammar_transitions,
        wordToCluster: merged.weights.grammar_word_map
    } : null);

    if (finalGrammar) {
        const grammarSidecarPath = `${outputPrefix}-grammar.json`;
        await fs.writeFile(grammarSidecarPath, JSON.stringify(finalGrammar, null, 2));
        console.log(`✅ Grammar sidecar saved: ${grammarSidecarPath}`);
    }

    // Save other sidecars
    for (const [type, data] of Object.entries(mergedSidecars)) {
        if (type === 'grammar') continue; // handled above
        const sidecarPath = `${outputPrefix}-${type}.json`;
        await fs.writeFile(sidecarPath, JSON.stringify(data, null, 2));
        console.log(`✅ Sidecar saved: ${sidecarPath}`);
    }

    console.log(`\n✅ Pipeline Complete! Merged model saved to: ${finalPath}`);
    console.log(`🚀 Try it: node unified_chat_fixed.js ${finalPath}`);
}

main().catch(err => {
    console.error('💥 Pipeline Failure:', err);
    process.exit(1);
});
