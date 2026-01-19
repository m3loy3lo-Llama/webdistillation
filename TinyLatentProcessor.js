/**
 * TINY LATENT PROCESSOR
 * The "Brain" of Æther.
 * 
 * Compresses 1280-dim embedding -> 256-dim Latent Thought
 * "Thinks" for N steps (Latent -> Latent recurrence)
 * Expands 256-dim Latent -> 1280-dim embedding for prediction
 */

import fs from 'fs/promises';
import wasm from './wasm-ops.js';

class TinyLatentProcessor {
    constructor(inputDim = 1280, latentDim = 256) {
        this.inputDim = inputDim;
        this.latentDim = latentDim;

        // ENCODER: Input -> Latent
        this.W_enc = this.xavierInit(inputDim, latentDim);
        this.b_enc = new Float32Array(latentDim).fill(0);

        // THINKER: Latent -> Latent (The "RNN" part)
        this.W_think = this.xavierInit(latentDim, latentDim);
        this.b_think = new Float32Array(latentDim).fill(0);

        // DECODER: Latent -> Input
        this.W_dec = this.xavierInit(latentDim, inputDim);
        this.b_dec = new Float32Array(inputDim).fill(0);

        this.initialized = true;
    }

    xavierInit(inDim, outDim) {
        const limit = Math.sqrt(6.0 / (inDim + outDim));
        const w = new Float32Array(inDim * outDim);
        for (let i = 0; i < w.length; i++) {
            w[i] = (Math.random() * 2 - 1) * limit;
        }
        return w;
    }

    /**
     * Encode: Embedding (1280) -> Latent (256)
     */
    encode(embedding) {
        const latent = new Float32Array(this.latentDim);
        // Matmul: 1 x 1280 * 1280 x 256 -> 1 x 256
        wasm.matmul(embedding, this.W_enc, latent, 1, this.latentDim, this.inputDim);

        // Bias + Activation (Tanh for bounded latent space)
        for (let i = 0; i < this.latentDim; i++) {
            latent[i] = Math.tanh(latent[i] + this.b_enc[i]);
        }
        return latent;
    }

    /**
     * Decode: Latent (256) -> Embedding (1280)
     */
    decode(latent) {
        const embedding = new Float32Array(this.inputDim);
        // Matmul: 1 x 256 * 256 x 1280 -> 1 x 1280
        wasm.matmul(latent, this.W_dec, embedding, 1, this.inputDim, this.latentDim);

        // Bias (Linear activation for embedding reconstruction)
        for (let i = 0; i < this.inputDim; i++) {
            embedding[i] += this.b_dec[i];
        }
        return embedding;
    }

    /**
     * Think: Evolve latent state for one step
     * z_new = Tanh(W_think * z + b_think) + z (Residual connection!)
     */
    think(latent) {
        const nextThought = new Float32Array(this.latentDim);
        // Matmul: 1 x 256 * 256 x 256 -> 1 x 256
        wasm.matmul(latent, this.W_think, nextThought, 1, this.latentDim, this.latentDim);

        // Activation + Residual
        const result = new Float32Array(this.latentDim);
        for (let i = 0; i < this.latentDim; i++) {
            result[i] = Math.tanh(nextThought[i] + this.b_think[i]) + (0.5 * latent[i]);
            // 0.5 residual scale keeps it stable
        }
        return result;
    }

    /**
     * Full Forward Pass with Thinking
     */
    forward(embedding, steps = 1, noiseScale = 0.0) {
        let z = this.encode(embedding);

        // Add Noise (Creativity/Innovation)
        if (noiseScale > 0) {
            for (let i = 0; i < this.latentDim; i++) {
                z[i] += (Math.random() * 2 - 1) * noiseScale;
            }
        }

        // Thinking Steps
        for (let s = 0; s < steps; s++) {
            z = this.think(z);
        }

        // Decode
        const rec = this.decode(z);
        return { reconstruction: rec, finalLatent: z };
    }

    /**
     * Train Step (Autoencoder style)
     * Target is usually the same as input (reconstruction), 
     * or next sentence embedding (prediction).
     */
    trainStep(input, target, lr = 0.001) {
        // FORWARD
        const z = this.encode(input);
        const out = this.decode(z);

        // BACKPROP (Simple SGD)

        // 1. Output Error (MSE derivative)
        const gradOut = new Float32Array(this.inputDim);
        for (let i = 0; i < this.inputDim; i++) {
            gradOut[i] = out[i] - target[i];
        }

        // 2. Decoder Gradients
        const gradZ = new Float32Array(this.latentDim);
        for (let i = 0; i < this.inputDim; i++) {
            this.b_dec[i] -= lr * gradOut[i]; // Bias up
            for (let j = 0; j < this.latentDim; j++) {
                // W_dec[j*out + i]
                const idx = j * this.inputDim + i;
                this.W_dec[idx] -= lr * gradOut[i] * z[j];
                // Accumulate gradient for Z
                gradZ[j] += gradOut[i] * this.W_dec[idx];
            }
        }

        // 3. Encoder Gradients (Through Tanh)
        // dTanh = 1 - y^2
        const gradEnc = new Float32Array(this.latentDim);
        for (let i = 0; i < this.latentDim; i++) {
            const dAct = 1.0 - (z[i] * z[i]);
            gradEnc[i] = gradZ[i] * dAct;

            this.b_enc[i] -= lr * gradEnc[i];
            for (let j = 0; j < this.inputDim; j++) {
                const idx = j * this.latentDim + i;
                this.W_enc[idx] -= lr * gradEnc[i] * input[j];
            }
        }

        // Calculate Loss (MSE)
        let loss = 0;
        for (let i = 0; i < this.inputDim; i++) {
            loss += (out[i] - target[i]) ** 2;
        }
        return loss / this.inputDim;
    }

    /**
     * Train Thinking Step (Predict NEXT embedding from CURRENT)
     * Input -> Enc -> Z -> Think -> Z_next -> Dec -> Output
     * Target = Next Input
     */
    trainThinkingStep(input, nextInput, lr = 0.001) {
        // 1. Forward Pass
        // Encode
        const z = this.encode(input);

        // Think (z -> z_next)
        // z_next = Tanh(W_think * z + b_think) + 0.5 * z
        const preActThink = new Float32Array(this.latentDim);
        wasm.matmul(z, this.W_think, preActThink, 1, this.latentDim, this.latentDim);

        const z_next = new Float32Array(this.latentDim);
        const thinkAct = new Float32Array(this.latentDim); // Store tanh outputs for gradient

        for (let i = 0; i < this.latentDim; i++) {
            thinkAct[i] = Math.tanh(preActThink[i] + this.b_think[i]);
            z_next[i] = thinkAct[i] + (0.5 * z[i]);
        }

        // Decode (z_next -> output)
        const out = this.decode(z_next);

        // 2. Backpropagation

        // A. Decoder Gradients (from Error)
        const gradOut = new Float32Array(this.inputDim);
        let loss = 0;
        for (let i = 0; i < this.inputDim; i++) {
            gradOut[i] = out[i] - nextInput[i];
            loss += gradOut[i] ** 2;
        }

        const gradZ_next = new Float32Array(this.latentDim);
        // Backprop through Decoder to Z_next
        for (let i = 0; i < this.inputDim; i++) {
            this.b_dec[i] -= lr * gradOut[i];
            for (let j = 0; j < this.latentDim; j++) {
                const idx = j * this.inputDim + i;
                this.W_dec[idx] -= lr * gradOut[i] * z_next[j];
                gradZ_next[j] += gradOut[i] * this.W_dec[idx];
            }
        }

        // B. Thinker Gradients (from Z_next to Z)
        // z_next = tanh(u) + 0.5*z
        // grad_u = gradZ_next * (1 - tanh^2(u))
        // grad_z = (W^T * grad_u) + (0.5 * gradZ_next)

        const gradZ = new Float32Array(this.latentDim);
        const gradThinkPre = new Float32Array(this.latentDim);

        for (let i = 0; i < this.latentDim; i++) {
            // Gradient of Tanh branch
            const dTanh = 1.0 - (thinkAct[i] * thinkAct[i]);
            gradThinkPre[i] = gradZ_next[i] * dTanh;

            // Update Thinker Weights
            this.b_think[i] -= lr * gradThinkPre[i];
            for (let j = 0; j < this.latentDim; j++) {
                const idx = j * this.latentDim + i;
                this.W_think[idx] -= lr * gradThinkPre[i] * z[j]; // dW = grad * input(z)

                // Accumulate gradients for Z (from the matmul part)
                gradZ[j] += gradThinkPre[i] * this.W_think[idx];
            }

            // Add residual gradient (from 0.5*z branch)
            gradZ[i] += 0.5 * gradZ_next[i];
        }

        // C. Encoder Gradients (from Z to Input)
        const gradEnc = new Float32Array(this.latentDim);
        for (let i = 0; i < this.latentDim; i++) {
            // Encoder uses Tanh
            const dAct = 1.0 - (z[i] * z[i]);
            gradEnc[i] = gradZ[i] * dAct;

            this.b_enc[i] -= lr * gradEnc[i];
            for (let j = 0; j < this.inputDim; j++) {
                const idx = j * this.latentDim + i;
                this.W_enc[idx] -= lr * gradEnc[i] * input[j];
            }
        }

        return loss / this.inputDim;
    }

    async save(path) {
        const data = {
            version: '1.0-latent',
            dims: [this.inputDim, this.latentDim],
            W_enc: Array.from(this.W_enc), b_enc: Array.from(this.b_enc),
            W_dec: Array.from(this.W_dec), b_dec: Array.from(this.b_dec),
            W_think: Array.from(this.W_think), b_think: Array.from(this.b_think)
        };
        await fs.writeFile(path, JSON.stringify(data));
    }

    async load(path) {
        const data = JSON.parse(await fs.readFile(path, 'utf8'));
        this.W_enc = new Float32Array(data.W_enc);
        this.b_enc = new Float32Array(data.b_enc);
        this.W_dec = new Float32Array(data.W_dec);
        this.b_dec = new Float32Array(data.b_dec);
        if (data.W_think) {
            this.W_think = new Float32Array(data.W_think);
            this.b_think = new Float32Array(data.b_think);
        }
        console.log("🧠 Latent Processor Loaded.");
    }
}

export default TinyLatentProcessor;
