#!/usr/bin/env node
/**
 * TRAIN LATENT SPACE
 * Trains the TinyLatentProcessor to compress/reconstruct Phase 1 embeddings.
 */

import fs from 'fs/promises';
import TinyLatentProcessor from './TinyLatentProcessor.js';

async function train() {
    console.log("🧠 Initializing Latent Space Training...");

    // 1. Load Data (Phase 1 Cache)
    const cacheFile = './unified-aether-cache-aether.json'; // Adjust based on your actual file
    let pairs = [];
    try {
        const data = JSON.parse(await fs.readFile(cacheFile, 'utf8'));
        pairs = data.pairs;
        console.log(`📦 Loaded ${pairs.length} embedding pairs.`);
    } catch (e) {
        console.error("❌ Could not load cache file! Ensure Phase 1 training involved.");
        console.error("   Looking for: " + cacheFile);
        process.exit(1);
    }

    // 2. Init Model
    const processor = new TinyLatentProcessor(1280, 256);
    const epochs = 42;
    const lr = 0.004;

    console.log(`🚀 Training Autoencoder (1280 -> 256 -> 1280) for ${epochs} epochs...`);

    // 3. Train Loop
    for (let epoch = 0; epoch < epochs; epoch++) {
        let totalLoss = 0;

        // Shuffle
        const shuffled = pairs.sort(() => Math.random() - 0.5);

        for (const p of shuffled) {
            const input = new Float32Array(p.input);
            // Autoencoder target = input (reconstruction)
            // Ideally we could also train "next thought" using p.target? 
            // For now, let's just ground the latent space in reconstruction.
            const loss = processor.trainStep(input, input, lr);
            totalLoss += loss;
        }

        const avgLoss = totalLoss / pairs.length;
        console.log(`   Epoch ${epoch + 1}/${epochs}: Loss = ${avgLoss.toFixed(6)}`);
    }

    // 4. Save
    await processor.save('./unified-aether-latent.json');
    console.log("✅ Latent Processor Trained & Saved!");
}

train();
