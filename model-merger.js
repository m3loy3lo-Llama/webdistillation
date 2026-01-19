/**
 * Æther Model Merger Utility
 * Purpose: Merges multiple checkpoints to retain semantic history and stabilize weights.
 * Handles vocabulary re-mapping to prevent ID collisions from restarted vocabularies.
 */

import fs from 'fs/promises';

/**
 * Robustly extracts Word -> ID mapping regardless of vocab orientation.
 * Some files use {ID: Word}, others use {Word: ID}.
 */
function getNormalizedVocab(ownVocab) {
    if (!ownVocab) return new Map();
    const entries = Object.entries(ownVocab);
    if (entries.length === 0) return new Map();

    const [firstKey, firstVal] = entries[0];
    const wordToId = new Map();

    if (typeof firstVal === 'string') {
        // Format is { ID: Word }
        entries.forEach(([id, word]) => wordToId.set(word, parseInt(id)));
    } else {
        // Format is { Word: ID }
        entries.forEach(([word, id]) => wordToId.set(word, parseInt(id)));
    }
    return wordToId;
}

/**
 * Merges prior versions by averaging weights and semantic embeddings.
 * Supports Global Vocab Remapping to handle restarted vocabularies.
 */
export async function mergePriorVersions(checkpoints, options = { allowVocabExpansion: true }) {
    const loadedCheckpoints = await Promise.all(
        checkpoints.map(async (cp) => {
            if (typeof cp === 'string') {
                const data = await fs.readFile(cp, 'utf8');
                return JSON.parse(data);
            }
            return cp;
        })
    );

    if (loadedCheckpoints.length === 0) throw new Error("No checkpoints to merge");

    // --- 1. Global Vocab Reconstruction ---
    console.log("Reconstructing global vocabulary...");
    const allWords = new Set();
    const checkpointVocabs = loadedCheckpoints.map(cp => getNormalizedVocab(cp.ownVocab));

    for (const wordToId of checkpointVocabs) {
        for (const word of wordToId.keys()) {
            allWords.add(word);
        }
    }

    const globalOwnVocab = {}; // Output format: { ID: Word }
    const wordToGlobalId = new Map();
    const sortedWords = Array.from(allWords).sort(); // Deterministic sorting

    sortedWords.forEach((word, idx) => {
        globalOwnVocab[idx] = word;
        wordToGlobalId.set(word, idx);
    });

    const globalVocabSize = sortedWords.length;
    console.log(`Unified vocabulary: ${globalVocabSize} words.`);

    // Use the latest checkpoint as the template for architecture and metadata
    const latest = loadedCheckpoints[loadedCheckpoints.length - 1];
    const merged = {
        ...JSON.parse(JSON.stringify(latest)),
        ownVocab: globalOwnVocab,
        architecture: {
            ...latest.architecture,
            vocabSize: globalVocabSize
        },
        metadata: {
            ...latest.metadata,
            mergeDate: new Date().toISOString(),
            versionsMerged: loadedCheckpoints.length,
            method: "Global Vocab Remapping & Symmetric Averaging"
        },
        vocab: {} // To be rebuilt
    };

    // --- 2. Rebuild GPT-to-Own Mapping (vocab) ---
    console.log("Rebuilding GPT-to-Own mappings...");
    const mergedGptMap = {};
    for (let c = 0; c < loadedCheckpoints.length; c++) {
        const cp = loadedCheckpoints[c];
        if (cp.vocab && cp.ownVocab) {
            const localWordToId = checkpointVocabs[c];
            const localIdToWord = new Map(Object.entries(cp.ownVocab).map(([id, w]) => [parseInt(id), w]));

            Object.entries(cp.vocab).forEach(([gptId, localId]) => {
                const word = localIdToWord.get(localId);
                if (word) {
                    const globalId = wordToGlobalId.get(word);
                    if (globalId !== undefined) {
                        mergedGptMap[gptId] = globalId;
                    }
                }
            });
        }
    }
    merged.vocab = mergedGptMap;

    // --- 3. Merge & Average Weights ---
    console.log("Merging weights with scattering...");
    const weightKeys = Object.keys(latest.weights);
    const hiddenSize = latest.architecture.tokenHiddenSize || 512;

    for (const key of weightKeys) {
        if (key.includes('grammar') || key.includes('map')) {
            merged.weights[key] = latest.weights[key];
            continue;
        }

        const isOutputLayer = key.includes('_2'); // W_token_2, b_token_2, etc. (hidden -> vocab)
        const isBias = key.startsWith('b_');

        if (isOutputLayer) {
            // SCATTER MODE
            if (isBias) {
                const globalBias = new Float32Array(globalVocabSize).fill(0);
                for (let gId = 0; gId < globalVocabSize; gId++) {
                    const word = globalOwnVocab[gId];
                    let sum = 0, count = 0;
                    for (let c = 0; c < loadedCheckpoints.length; c++) {
                        const localWordToId = checkpointVocabs[c];
                        const localId = localWordToId.get(word);
                        if (localId !== undefined && loadedCheckpoints[c].weights[key]) {
                            sum += loadedCheckpoints[c].weights[key][localId];
                            count++;
                        }
                    }
                    globalBias[gId] = count > 0 ? sum / count : 0;
                }
                merged.weights[key] = Array.from(globalBias);
            } else {
                // W [hidden x vocab] -> col-major: col = id, row = hidden
                const globalW = new Float32Array(hiddenSize * globalVocabSize).fill(0);
                for (let gId = 0; gId < globalVocabSize; gId++) {
                    const word = globalOwnVocab[gId];
                    let count = 0;
                    const accumulatedCol = new Float32Array(hiddenSize).fill(0);

                    for (let c = 0; c < loadedCheckpoints.length; c++) {
                        const localWordToId = checkpointVocabs[c];
                        const localId = localWordToId.get(word);
                        if (localId !== undefined && loadedCheckpoints[c].weights[key]) {
                            const cpVocabSize = loadedCheckpoints[c].architecture.vocabSize || (loadedCheckpoints[c].weights[key].length / hiddenSize);
                            for (let h = 0; h < hiddenSize; h++) {
                                accumulatedCol[h] += loadedCheckpoints[c].weights[key][h * cpVocabSize + localId];
                            }
                            count++;
                        }
                    }
                    if (count > 0) {
                        for (let h = 0; h < hiddenSize; h++) {
                            globalW[h * globalVocabSize + gId] = accumulatedCol[h] / count;
                        }
                    }
                }
                merged.weights[key] = Array.from(globalW);
            }
        } else {
            // STANDARD AVERAGE MODE (Input layers W1, b1)
            const firstWeight = latest.weights[key];
            if (Array.isArray(firstWeight)) {
                const length = firstWeight.length;
                const averaged = new Float32Array(length);
                for (let i = 0; i < length; i++) {
                    let sum = 0, count = 0;
                    for (const cp of loadedCheckpoints) {
                        if (cp.weights && cp.weights[key] && cp.weights[key][i] !== undefined) {
                            sum += cp.weights[key][i];
                            count++;
                        }
                    }
                    averaged[i] = count > 0 ? sum / count : 0;
                }
                merged.weights[key] = Array.from(averaged);
            }
        }
    }

    // --- 3. Merge & Average Semantic Embeddings ---
    console.log("Merging semantic embeddings (string-aligned)...");
    const globalEmbeddings = {};
    for (let gId = 0; gId < globalVocabSize; gId++) {
        const word = globalOwnVocab[gId];
        let sumVector = null;
        let count = 0;

        for (let c = 0; c < loadedCheckpoints.length; c++) {
            const localWordToId = checkpointVocabs[c];
            const localId = localWordToId.get(word);
            const embed = loadedCheckpoints[c].semanticEmbeddings?.[localId];

            if (embed && Array.isArray(embed)) {
                if (!sumVector) sumVector = new Float32Array(embed.length);
                for (let i = 0; i < embed.length; i++) sumVector[i] += embed[i];
                count++;
            }
        }

        if (sumVector && count > 0) {
            for (let i = 0; i < sumVector.length; i++) sumVector[i] /= count;
            globalEmbeddings[gId] = Array.from(sumVector);
        }
    }
    merged.semanticEmbeddings = globalEmbeddings;

    return merged;
}

/**
 * Generic merge for simple weight-based checkpoints (Latent, Grammar, etc.)
 */
export async function mergeSimpleCheckpoints(checkpoints) {
    const loaded = await Promise.all(
        checkpoints.map(async (cp) => {
            if (typeof cp === 'string') {
                try {
                    const data = await fs.readFile(cp, 'utf8');
                    return JSON.parse(data);
                } catch (e) { return null; }
            }
            return cp;
        })
    );

    const valid = loaded.filter(cp => cp !== null);
    if (valid.length === 0) return null;

    const template = valid[valid.length - 1];
    const merged = JSON.parse(JSON.stringify(template));

    const averageRecursive = (target, sources) => {
        for (const key in target) {
            if (key.toLowerCase().includes('id') || key.includes('wordToCluster') || key.includes('map')) {
                continue;
            }

            if (typeof target[key] === 'number') {
                let sum = 0, count = 0;
                for (const s of sources) {
                    if (typeof s[key] === 'number') {
                        sum += s[key];
                        count++;
                    }
                }
                target[key] = count > 0 ? sum / count : (target[key] || 0);
            } else if (Array.isArray(target[key])) {
                const len = target[key].length;
                for (let i = 0; i < len; i++) {
                    let sum = 0, count = 0;
                    for (const s of sources) {
                        if (Array.isArray(s[key]) && s[key][i] !== undefined) {
                            sum += s[key][i];
                            count++;
                        }
                    }
                    if (count > 0) target[key][i] = sum / count;
                }
            } else if (typeof target[key] === 'object' && target[key] !== null) {
                averageRecursive(target[key], sources.map(s => s[key]).filter(s => s !== undefined));
            }
        }
    };

    averageRecursive(merged, valid);
    return merged;
}
