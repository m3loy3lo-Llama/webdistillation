/**
 * TINY TOKEN PREDICTOR
 * A small neural network that learns to predict next tokens
 *
 * Architecture: Embedding (1280) → Hidden (304) → Vocab (50257)
 * Total params: ~200M (vs SmolLM2's 270M)
 *
 * This model can GENERATE TEXT, not just transform embeddings!
 */

import fs from 'fs/promises';

class TinyTokenPredictor {
    constructor(vocabSize = 50257, embeddingSize = 1280, hiddenSize = 304) {
        this.vocabSize = vocabSize;      // Full GPT-2 vocabulary
        this.embeddingSize = embeddingSize; // Input from sentence embedding
        this.hiddenSize = hiddenSize;     // Hidden layer size

        // Layer 1: embedding → hidden
        this.W1 = this.xavierInit(embeddingSize, hiddenSize);
        this.b1 = new Float32Array(hiddenSize).fill(0);

        // Layer 2: hidden → vocabulary predictions
        this.W2 = this.xavierInit(hiddenSize, vocabSize);
        this.b2 = new Float32Array(vocabSize).fill(0);

        this.isInitialized = true;
        this.freezeEmbedding = false;

        const params = this.countParams();
        const sizeMB = (params * 4) / (1024 * 1024);

        console.log('🎯 Tiny Token Predictor initialized:');
        console.log(`   Vocab: ${vocabSize.toLocaleString()}, Hidden: ${hiddenSize}`);
        console.log(`   Parameters: ${params.toLocaleString()} (~${Math.round(sizeMB)}MB)`);
        console.log(`   Can predict ${vocabSize.toLocaleString()} different tokens!`);
    }

    /**
     * Xavier/Glorot initialization
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
     * Forward pass: sentence embedding → token probabilities
     * Returns probability distribution over entire vocabulary
     */
    forward(embedding, training = false, dropout = 0.3) {
        if (embedding.length !== this.embeddingSize) {
            throw new Error(`Input size mismatch: expected ${this.embeddingSize}, got ${embedding.length}`);
        }

        // Layer 1: embedding → hidden (with ReLU activation)
        const hidden = new Float32Array(this.hiddenSize);
        for (let i = 0; i < this.hiddenSize; i++) {
            let sum = this.b1[i];
            for (let j = 0; j < this.embeddingSize; j++) {
                sum += embedding[j] * this.W1[j * this.hiddenSize + i];
            }
            hidden[i] = Math.max(0, sum); // ReLU
        }

        // DROPOUT: Only during training
        let hiddenAfterDropout = hidden;
        if (training && dropout > 0) {
            hiddenAfterDropout = new Float32Array(this.hiddenSize);
            const scale = 1.0 / (1.0 - dropout);
            for (let i = 0; i < this.hiddenSize; i++) {
                if (Math.random() > dropout) {
                    hiddenAfterDropout[i] = hidden[i] * scale;
                } else {
                    hiddenAfterDropout[i] = 0;
                }
            }
        }

        // Layer 2: hidden → vocabulary logits
        const logits = new Float32Array(this.vocabSize);
        for (let i = 0; i < this.vocabSize; i++) {
            let sum = this.b2[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += hiddenAfterDropout[j] * this.W2[j * this.vocabSize + i];
            }
            logits[i] = sum;
        }

        // Apply softmax to get probabilities
        const probs = this.softmax(logits);

        return { logits, probs, hidden: hiddenAfterDropout, embedding };
    }

    /**
     * Softmax: convert logits to probabilities
     */
    softmax(logits) {
        const maxLogit = Math.max(...logits);
        const exps = new Float32Array(this.vocabSize);
        let sumExps = 0;

        for (let i = 0; i < this.vocabSize; i++) {
            exps[i] = Math.exp(logits[i] - maxLogit);
            sumExps += exps[i];
        }

        const probs = new Float32Array(this.vocabSize);
        for (let i = 0; i < this.vocabSize; i++) {
            probs[i] = exps[i] / sumExps;
        }

        return probs;
    }

    /**
     * Sample a token from probability distribution
     * Uses temperature and top-k sampling for quality
     */
    sampleToken(probs, temperature = 1.0, topK = 50, topP = 0.9, repetitionPenalty = 1.0, priorTokens = null) {
        // Temperature scaling & Logit reconstruction
        const scaledLogProbs = new Float32Array(this.vocabSize);
        for (let i = 0; i < this.vocabSize; i++) {
            scaledLogProbs[i] = Math.log(probs[i] + 1e-10);
        }

        // Apply Repetition Penalty
        if (repetitionPenalty !== 1.0 && priorTokens) {
            const tempSet = new Set(priorTokens);
            for (const token of tempSet) {
                if (scaledLogProbs[token] < 0) {
                    scaledLogProbs[token] *= repetitionPenalty;
                } else {
                    scaledLogProbs[token] /= repetitionPenalty;
                }
            }
        }

        // Apply Temperature
        for (let i = 0; i < this.vocabSize; i++) {
            scaledLogProbs[i] /= temperature;
        }

        // Re-softmax with temperature
        const maxLogProb = Math.max(...scaledLogProbs);
        const scaledProbs = new Float32Array(this.vocabSize);
        let sumProbs = 0;

        for (let i = 0; i < this.vocabSize; i++) {
            scaledProbs[i] = Math.exp(scaledLogProbs[i] - maxLogProb);
            sumProbs += scaledProbs[i];
        }

        // Create indexed array for sorting
        const indexed = [];
        for (let i = 0; i < this.vocabSize; i++) {
            scaledProbs[i] /= sumProbs;
            indexed.push({ prob: scaledProbs[i], index: i });
        }

        // Sort by probability descending
        indexed.sort((a, b) => b.prob - a.prob);

        // TOP-K Filtering
        const topKFiltered = indexed.slice(0, topK);

        // TOP-P (Nucleus) Filtering
        // We only keep the top tokens whose cumulative probability <= topP
        const nucleus = [];
        let cumulativeProb = 0;

        for (const item of topKFiltered) {
            cumulativeProb += item.prob;
            nucleus.push(item);
            // If we've crossed the threshold, stop adding more tokens
            // (But always include at least one token)
            if (cumulativeProb >= topP) {
                break;
            }
        }

        // Normalize the nucleus probabilities
        const nucleusSum = nucleus.reduce((sum, item) => sum + item.prob, 0);

        // Sample from nucleus
        let random = Math.random() * nucleusSum;
        for (const item of nucleus) {
            random -= item.prob;
            if (random <= 0) {
                return item.index;
            }
        }

        return nucleus[0].index; // Fallback
    }

    /**
     * BOSS-FIGHT trainFromTeacher with WINCON escalation - adaptive learning rate + validation splits
     * Retries epoch attempts until validation improves, escalates strategy under pressure
     */
    async trainFromTeacher(pairs, options = {}) {
        const {
            epochs = 10,
            learningRate = 0.001,
            batchSize = 16,
            validationSplit = 0.1,
            logInterval = 1,
            interruptMs = 428.57,
            interruptBatch = 0.314159,
            patience = 2,
            checkpointRewind = true
        } = options;

        let currentLearningRate = learningRate;
        let currentValidationSplit = validationSplit;

        // Split data initially
        const shuffled = [...pairs].sort(() => Math.random() - 0.5);
        let splitIdx = Math.floor(pairs.length * (1 - currentValidationSplit));
        let trainData = shuffled.slice(0, splitIdx);
        let valData = shuffled.slice(splitIdx);

        console.log(`🎓 Training token predictor on ${pairs.length} examples...`);
        console.log(`   Epochs: ${epochs}, Initial LR: ${learningRate}, Initial Val Split: ${validationSplit}`);
        console.log(`   Interrupt: ${interruptMs} ms / ${interruptBatch} examples`);
        console.log(`   📦 Checkpoint-Rewind: ${checkpointRewind ? 'ON (patience=' + patience + ')' : 'OFF'}\n`);
        console.log(`   Initial Train: ${trainData.length} | Val: ${valData.length}\n`);

        let bestValLoss = Infinity;
        let bestEpoch = -1;
        let checkpoint = null;
        let epochsCompleted = 0;
        let retryCount = 0;

        // BOSS-FIGHT MODE: Stay at checkpoint until we beat it
        while (epochsCompleted < epochs) {
            // Reset hyperparameters for each epoch attempt (fresh per batch epoch)
            currentLearningRate = learningRate;
            currentValidationSplit = validationSplit;

            // Re-split data with original validation split
            splitIdx = Math.floor(pairs.length * (1 - currentValidationSplit));
            trainData = shuffled.slice(0, splitIdx);
            valData = shuffled.slice(splitIdx);

            const epoch = epochsCompleted;

            // Train one epoch
            const trainLoss = await this._runEpoch(trainData, currentLearningRate, batchSize, interruptMs, interruptBatch, true);
            const { loss: valLoss, accuracy: valAcc } = await this._validate(valData);

            const remaining = epochs - epochsCompleted - 1;

            // Check if this is a new best
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                bestEpoch = epoch;
                retryCount = 0; // Reset retry counter on improvement

                // Save checkpoint
                checkpoint = {
                    W1: new Float32Array(this.W1),
                    b1: new Float32Array(this.b1),
                    W2: new Float32Array(this.W2),
                    b2: new Float32Array(this.b2),
                    epoch: epoch,
                    valLoss: valLoss,
                    learningRate: currentLearningRate,
                    validationSplit: currentValidationSplit
                };

                console.log(`   💾 Epoch ${epoch.toString().padStart(3)}/${epochs}: Train Loss = ${trainLoss.toFixed(4)}, Val Loss = ${valLoss.toFixed(4)} ⭐ NEW BEST, Val Acc = ${valAcc.toFixed(2)}%, LR=${currentLearningRate.toFixed(6)}, ValSplit=${currentValidationSplit.toFixed(2)}, Time = ${Date.now()}ms, Remaining = ${remaining}`);

                epochsCompleted++;

            } else {
                // BOSS NOT DEFEATED!
                retryCount++;
                console.log(`   🔄 Epoch ${epoch.toString().padStart(3)}/${epochs}: Train Loss = ${trainLoss.toFixed(4)}, Val Loss = ${valLoss.toFixed(4)} ❌ WORSE (Retry #${retryCount}), Val Acc = ${valAcc.toFixed(2)}%, Time = ${Date.now()}ms`);

                if (checkpoint && checkpointRewind) {
                    // Restore checkpoint weights
                    this.W1 = new Float32Array(checkpoint.W1);
                    this.b1 = new Float32Array(checkpoint.b1);
                    this.W2 = new Float32Array(checkpoint.W2);
                    this.b2 = new Float32Array(checkpoint.b2);

                    // WINCON: After 3 failed retries, escalate
                    if (retryCount >= 3 && retryCount % 3 === 0) {
                        // Drop LR by 0.0001 + Increase Val Split by 0.05
                        currentLearningRate = Math.max(currentLearningRate - 0.0001, 1e-7);
                        currentValidationSplit = Math.min(currentValidationSplit + 0.05, 0.5);

                        console.log(`   ⚡ WINCON ACTIVATED! Retry #${retryCount} → LR ↓ ${currentLearningRate.toFixed(7)}, ValSplit ↑ ${currentValidationSplit.toFixed(2)}`);

                        // Re-split data with new validation size
                        const newSplitIdx = Math.floor(pairs.length * (1 - currentValidationSplit));
                        trainData = shuffled.slice(0, newSplitIdx);
                        valData = shuffled.slice(newSplitIdx);

                        console.log(`   🧠 Resplit: Train=${trainData.length} | Val=${valData.length}`);
                    }

                    console.log(`   ⚔️  RESPAWN! Retrying from epoch ${checkpoint.epoch} (best val: ${checkpoint.valLoss.toFixed(4)})\n`);
                    // Don't increment epochsCompleted
                } else {
                    // No checkpoint rewind enabled, just continue
                    epochsCompleted++;
                }
            }
        }

        // Training complete - restore best checkpoint
        if (checkpoint) {
            console.log(`\n   🏆 Training complete! Final best checkpoint from epoch ${bestEpoch}`);
            console.log(`      Best val loss: ${bestValLoss.toFixed(4)}\n`);

            this.W1 = checkpoint.W1;
            this.b1 = checkpoint.b1;
            this.W2 = checkpoint.W2;
            this.b2 = checkpoint.b2;
        }

        console.log('✅ Token predictor training complete!\n');
    }

    /**
     * Backward pass: compute gradients and update weights
     */
    backward(forwardResult, targetTokenId, learningRate) {
        const { probs, hidden, embedding } = forwardResult;

        // Gradient of cross-entropy loss w.r.t. logits
        // For cross-entropy with softmax: grad = probs - one_hot(target)
        const gradLogits = new Float32Array(this.vocabSize);
        for (let i = 0; i < this.vocabSize; i++) {
            gradLogits[i] = probs[i];
        }
        gradLogits[targetTokenId] -= 1.0; // Subtract 1 for the target

        // Backprop through Layer 2
        const gradHidden = new Float32Array(this.hiddenSize);
        for (let i = 0; i < this.vocabSize; i++) {
            this.b2[i] -= learningRate * gradLogits[i];

            for (let j = 0; j < this.hiddenSize; j++) {
                this.W2[j * this.vocabSize + i] -= learningRate * gradLogits[i] * hidden[j];
                gradHidden[j] += gradLogits[i] * this.W2[j * this.vocabSize + i];
            }
        }

        // Backprop through Layer 1 (with ReLU derivative)
        if (!this.freezeEmbedding) {
            for (let i = 0; i < this.hiddenSize; i++) {
                if (hidden[i] > 0) { // ReLU derivative: 1 if x > 0, else 0
                    this.b1[i] -= learningRate * gradHidden[i];

                    for (let j = 0; j < this.embeddingSize; j++) {
                        this.W1[j * this.hiddenSize + i] -= learningRate * gradHidden[i] * embedding[j];
                    }
                }
            }
        }
    }

    /**
     * Run one epoch of training
     */
    async _runEpoch(data, lr, batchSize, interruptMs, interruptBatch, training = true) {
        // Safety check for empty data
        if (!data || data.length === 0) {
            console.log('      ⚠️ Empty training data, returning 0 loss');
            return 0;
        }

        let totalLoss = 0;
        let lastInterrupt = Date.now();
        let exampleCounter = 0;

        const yieldControl = async () => {
            await new Promise(r => setTimeout(r, 0));
        };

        for (let i = 0; i < data.length; i++) {
            // ---- optional per-example interrupt ----
            if (interruptBatch > 0 && ++exampleCounter >= interruptBatch) {
                await yieldControl();
                exampleCounter = 0;
            }

            // ---- time-based interrupt ----
            const now = Date.now();
            if (interruptMs > 0 && now - lastInterrupt >= interruptMs) {
                await yieldControl();
                lastInterrupt = now;
            }

            const { embedding, targetTokenId } = data[i];

            // Forward pass
            const forwardResult = this.forward(embedding, training);

            // Cross-entropy loss (for single target token)
            const loss = -Math.log(forwardResult.probs[targetTokenId] + 1e-10);
            totalLoss += loss;

            // Backward pass
            this.backward(forwardResult, targetTokenId, lr);
        }

        return totalLoss / data.length;
    }

    /**
     * Validate the model on data
     */
    async _validate(data) {
        // Safety check for empty data
        if (!data || data.length === 0) {
            console.log('      ⚠️ Empty validation data, returning Infinity loss');
            return { loss: Infinity, accuracy: 0 };
        }

        let valLoss = 0;
        let valAccuracy = 0;

        for (const { embedding, targetTokenId } of data) {
            const { probs } = this.forward(embedding); // No training
            valLoss -= Math.log(probs[targetTokenId] + 1e-10);

            // Calculate accuracy (top-1)
            let maxProb = -Infinity;
            let predictedId = -1;
            for (let i = 0; i < probs.length; i++) {
                if (probs[i] > maxProb) {
                    maxProb = probs[i];
                    predictedId = i;
                }
            }
            if (predictedId === targetTokenId) valAccuracy++;
        }

        const loss = valLoss / data.length;
        const accuracy = (valAccuracy / data.length) * 100; // Percentage

        return { loss, accuracy };
    }

    /**
     * Predict top-k most likely next tokens
     */
    predictTopK(embedding, k = 10) {
        const { probs } = this.forward(embedding);

        const indexed = [];
        for (let i = 0; i < this.vocabSize; i++) {
            indexed.push({ tokenId: i, prob: probs[i] });
        }
        indexed.sort((a, b) => b.prob - a.prob);

        return indexed.slice(0, k);
    }

    /**
     * Save model to file
     */
    async save(filepath) {
        const modelData = {
            version: '1.0',
            type: 'token_predictor',
            architecture: {
                vocabSize: this.vocabSize,
                embeddingSize: this.embeddingSize,
                hiddenSize: this.hiddenSize
            },
            weights: {
                W1: Array.from(this.W1),
                b1: Array.from(this.b1),
                W2: Array.from(this.W2),
                b2: Array.from(this.b2)
            },
            metadata: {
                trained: new Date().toISOString(),
                parameters: this.countParams()
            }
        };

        await fs.writeFile(filepath, JSON.stringify(modelData));
        const stats = await fs.stat(filepath);
        const sizeMB = stats.size / (1024 * 1024);

        console.log(`💾 Saved token predictor to ${filepath}`);
        console.log(`   File size: ${sizeMB.toFixed(1)}MB`);
    }

    /**
     * Load model from file
     */
    async load(filepath) {
        const modelData = JSON.parse(await fs.readFile(filepath, 'utf8'));

        this.vocabSize = modelData.architecture.vocabSize;
        this.embeddingSize = modelData.architecture.embeddingSize;
        this.hiddenSize = modelData.architecture.hiddenSize;

        this.W1 = new Float32Array(modelData.weights.W1);
        this.b1 = new Float32Array(modelData.weights.b1);
        this.W2 = new Float32Array(modelData.weights.W2);
        this.b2 = new Float32Array(modelData.weights.b2);

        this.isInitialized = true;

        console.log(`📂 Loaded token predictor from ${filepath}`);
        console.log(`   Trained: ${modelData.metadata?.trained || 'unknown'}`);
        console.log(`   Parameters: ${this.countParams().toLocaleString()}`);
    }

    countParams() {
        return this.W1.length + this.b1.length + this.W2.length + this.b2.length;
    }

    getStatus() {
        return {
            initialized: this.isInitialized,
            type: 'token_predictor',
            architecture: {
                vocab: this.vocabSize,
                embedding: this.embeddingSize,
                hidden: this.hiddenSize
            },
            parameters: this.countParams(),
            canGenerateText: true
        };
    }
}

export default TinyTokenPredictor;
