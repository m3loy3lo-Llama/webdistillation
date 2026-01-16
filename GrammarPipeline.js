import fs from 'fs/promises';

/**
 * GRAMMAR INDUCTOR
 * Unsupervised discovery of grammatical structure via semantic clustering.
 * 
 * Concept:
 * 1. Words with similar embeddings often share grammatical roles (nouns, verbs, etc.)
 * 2. We cluster embeddings into K groups (Pseudo-POS tags).
 * 3. We learn the transition probabilities between these groups.
 */
class GrammarInductor {
    constructor(numClusters = 64, embeddingDim = 1280) {
        this.numClusters = numClusters;
        this.embeddingDim = embeddingDim;

        this.centroids = []; // Array of Float32Array
        this.wordToCluster = new Map(); // word (string) -> clusterId (int)
        this.transitions = []; // numClusters x numClusters matrix (probabilities)

        this.isInitialized = false;
    }

    /**
     * Initialize centroids randomly from data
     */
    initCentroids(embeddings) {
        this.centroids = [];
        const indices = new Set();
        while (indices.size < this.numClusters && indices.size < embeddings.length) {
            indices.add(Math.floor(Math.random() * embeddings.length));
        }

        for (const idx of indices) {
            this.centroids.push(new Float32Array(embeddings[idx]));
        }
    }

    /**
     * Compute Euclidean distance between two vectors
     */
    distance(v1, v2) {
        let sum = 0;
        for (let i = 0; i < v1.length; i++) {
            const diff = v1[i] - v2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * Run K-Means clustering on word embeddings
     * @param {Map<string, Float32Array>} wordEmbeddings 
     */
    async trainClusters(wordEmbeddings, iterations = 20) {
        console.log(`   🧩 Clustering ${wordEmbeddings.size} words into ${this.numClusters} grammatical categories...`);

        const words = Array.from(wordEmbeddings.keys());
        const embeddings = Array.from(wordEmbeddings.values());

        // 1. Initialize
        this.initCentroids(embeddings);

        // 2. Iterate
        for (let iter = 0; iter < iterations; iter++) {
            const clusters = Array(this.numClusters).fill().map(() => []);
            let changes = 0;

            // Assign words to nearest centroid
            for (let i = 0; i < words.length; i++) {
                const emb = embeddings[i];
                let minDist = Infinity;
                let bestCluster = 0;

                for (let c = 0; c < this.numClusters; c++) {
                    const d = this.distance(emb, this.centroids[c]);
                    if (d < minDist) {
                        minDist = d;
                        bestCluster = c;
                    }
                }

                clusters[bestCluster].push(emb);

                const word = words[i];
                if (this.wordToCluster.get(word) !== bestCluster) {
                    this.wordToCluster.set(word, bestCluster);
                    changes++;
                }
            }

            // Recompute centroids
            for (let c = 0; c < this.numClusters; c++) {
                if (clusters[c].length > 0) {
                    const newCentroid = new Float32Array(this.embeddingDim).fill(0);
                    for (const emb of clusters[c]) {
                        for (let j = 0; j < this.embeddingDim; j++) {
                            newCentroid[j] += emb[j];
                        }
                    }
                    for (let j = 0; j < this.embeddingDim; j++) {
                        newCentroid[j] /= clusters[c].length;
                    }
                    this.centroids[c] = newCentroid;
                }
            }

            if (iter % 5 === 0) {
                console.log(`      Iter ${iter}: ${changes} assignments changed`);
            }

            if (changes === 0) break;
        }

        console.log(`   ✓ Clustering complete. Discovered ${this.numClusters} grammar tags.\n`);
    }

    /**
     * Learn transition probabilities between clusters (Bigram model)
     * @param {string[][]} tokenizedSentences Array of sentences (each is array of words)
     */
    trainTransitions(tokenizedSentences) {
        console.log(`   🔗 Learning grammar flow from ${tokenizedSentences.length} sentences...`);

        // Initialize counts
        const counts = Array(this.numClusters).fill().map(() => new Float32Array(this.numClusters).fill(0));
        const totals = new Float32Array(this.numClusters).fill(0);

        for (const sentence of tokenizedSentences) {
            let prevCluster = -1;

            for (const word of sentence) {
                const cluster = this.wordToCluster.get(word);
                if (cluster !== undefined) {
                    if (prevCluster !== -1) {
                        counts[prevCluster][cluster]++;
                        totals[prevCluster]++;
                    }
                    prevCluster = cluster;
                }
            }
        }

        // Normalize to probabilities
        this.transitions = counts.map((row, i) => {
            const total = totals[i] || 1;
            return row.map(count => count / total);
        });

        console.log(`   ✓ Grammar transition matrix built (${this.numClusters}x${this.numClusters})\n`);
        this.isInitialized = true;
    }

    /**
     * Get the probability distribution for the NEXT cluster given the CURRENT cluster
     */
    getNextClusterProbs(currentClusterId) {
        if (!this.isInitialized || currentClusterId < 0 || currentClusterId >= this.numClusters) {
            return new Float32Array(this.numClusters).fill(1.0 / this.numClusters); // Uniform fallback
        }
        return this.transitions[currentClusterId];
    }

    getClusterForWord(word) {
        return this.wordToCluster.get(word);
    }

    async save(filepath) {
        const data = {
            version: "1.0-grammar",
            numClusters: this.numClusters,
            embeddingDim: this.embeddingDim,
            centroids: this.centroids.map(c => Array.from(c)),
            wordToCluster: Object.fromEntries(this.wordToCluster),
            transitions: this.transitions.map(row => Array.from(row))
        };
        await fs.writeFile(filepath, JSON.stringify(data));
        console.log(`   💾 Grammar model saved to ${filepath}`);
    }

    async load(filepath) {
        const data = JSON.parse(await fs.readFile(filepath, 'utf8'));
        this.numClusters = data.numClusters;
        this.embeddingDim = data.embeddingDim;
        this.centroids = data.centroids.map(c => new Float32Array(c));
        this.wordToCluster = new Map(Object.entries(data.wordToCluster).map(([k, v]) => [k, parseInt(v)]));
        this.transitions = data.transitions.map(row => new Float32Array(row));
        this.isInitialized = true;
        console.log(`   📂 Grammar model loaded (${this.numClusters} clusters)`);
    }
}

export default GrammarInductor;
