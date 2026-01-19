/**
 * Formats an array of tokens into a sonnet-like structure.
 * 
 * Logic:
 * - Lines: ~10 words (8-12 range), or earlier if a sentence ends.
 * - Stanzas: 3 quatrains (4 lines) and 1 couplet (2 lines).
 * - Total: 14 lines.
 * 
 * @param {string[]} tokens - Array of generated token strings.
 * @returns {string} - Formatted sonnet string.
 */
export function formatAsSonnet(tokens) {
    if (!tokens || tokens.length === 0) return "";

    const lines = [];
    let currentLineTokens = [];
    let lineCount = 0;

    for (let i = 0; i < tokens.length; i++) {
        const token = tokens[i];
        currentLineTokens.push(token);

        const isSentenceEnd = /[.!?]$/.test(token);
        const isNaturalPause = /[,;:]$/.test(token);

        // Target 10 words, pull in a bit for sentence ends or pauses
        const shouldBreakLine =
            (isSentenceEnd && currentLineTokens.length >= 7) ||
            (currentLineTokens.length >= 10 && isNaturalPause) ||
            (currentLineTokens.length >= 12) ||
            (i === tokens.length - 1);

        if (shouldBreakLine) {
            lines.push(currentLineTokens.join(" "));
            currentLineTokens = [];
            lineCount++;
            if (lineCount >= 14) break;
        }
    }

    // Group lines into stanzas (4, 4, 4, 2)
    let result = "";
    for (let i = 0; i < lines.length; i++) {
        result += lines[i] + "\n";

        // After lines 4, 8, and 12, add a blank line for stanza separation
        if (i === 3 || i === 7 || i === 11) {
            if (i < lines.length - 1) {
                result += "\n";
            }
        }
    }

    return result.trim();
}

// Support CommonJS if needed (though the project seems to use ESM)
// module.exports = { formatAsSonnet };
