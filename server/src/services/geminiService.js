const axios = require('axios');

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const ANALYSIS_PROMPT = `You are a professional Hollywood script analyst AI. Analyze the following movie script text and return a structured JSON response. Do NOT include markdown code fences or any extra text — output ONLY valid JSON.

The JSON must follow this exact structure:
{
  "plot_summary": "A 3-5 sentence summary of the script's main plot.",
  "tone": {
    "primary": "The dominant tone (e.g., Dark, Light-hearted, Suspenseful, Romantic, Comedic, Inspirational)",
    "secondary": "A secondary tone if applicable, or null",
    "description": "A 1-2 sentence explanation of the tonal quality."
  },
  "pacing": {
    "rating": "Slow / Moderate / Fast / Variable",
    "description": "A 1-2 sentence analysis of the script's pacing."
  },
  "genre_prediction": ["Primary Genre", "Secondary Genre"],
  "themes": ["Theme 1", "Theme 2", "Theme 3"],
  "strengths": ["Strength 1", "Strength 2", "Strength 3"],
  "weaknesses": ["Weakness 1", "Weakness 2"],
  "demographics": {
    "target_age": "e.g., 18-35",
    "target_gender": "e.g., Universal / Male-skewing / Female-skewing",
    "target_interests": ["Interest 1", "Interest 2", "Interest 3"],
    "market_appeal": "Global / Domestic / Niche",
    "comparable_films": ["Film 1", "Film 2", "Film 3"]
  },
  "success_indicators": {
    "commercial_potential": "Low / Medium / High",
    "critical_potential": "Low / Medium / High",
    "audience_engagement": "Low / Medium / High",
    "reasoning": "A 2-3 sentence explanation of the commercial and critical outlook."
  },
  "overall_score": 7.5
}

Here is the script text to analyze:
`;

/** Helper: wait for N milliseconds */
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Call Gemini REST API directly with retry logic for rate limits
 */
const callGeminiREST = async (modelName, prompt, retryCount = 0) => {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${GEMINI_API_KEY}`;

    try {
        const response = await axios.post(url, {
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: {
                temperature: 0.7,
                maxOutputTokens: 4096,
            }
        }, {
            headers: { 'Content-Type': 'application/json' },
            timeout: 90000, // 90 second timeout
        });

        const text = response.data?.candidates?.[0]?.content?.parts?.[0]?.text;
        if (!text) {
            throw new Error('Empty response from Gemini API');
        }
        return text;

    } catch (err) {
        const status = err.response?.status;

        // If rate limited (429) or temporarily overloaded (503), wait and retry up to 2 times
        if ((status === 429 || status === 503) && retryCount < 2) {
            const waitTime = (retryCount + 1) * 15; // 15s, then 30s
            console.log(`⏳ Rate limited on "${modelName}". Waiting ${waitTime}s before retry ${retryCount + 1}/2...`);
            await sleep(waitTime * 1000);
            return callGeminiREST(modelName, prompt, retryCount + 1);
        }

        throw err; // re-throw if retries exhausted or different error
    }
};

/**
 * Analyze script text using Gemini AI with automatic retry + model fallback
 * @param {string} scriptText - The extracted text from the uploaded script
 * @returns {Object} The structured analysis result
 */
const analyzeScript = async (scriptText) => {
    // Truncate to ~15,000 chars to reduce token usage and avoid rate limits
    const truncated = scriptText.length > 15000
        ? scriptText.substring(0, 15000) + '\n\n[...script truncated for analysis...]'
        : scriptText;

    const fullPrompt = ANALYSIS_PROMPT + truncated;

    // Ordered list of models to try
    const modelCandidates = [
        'gemini-2.0-flash',
        'gemini-2.0-flash-lite',
        'gemini-2.5-flash',
    ];

    let lastError = null;

    for (const modelName of modelCandidates) {
        try {
            console.log(`🎬 Script Analysis: Trying model "${modelName}"...`);

            const text = await callGeminiREST(modelName, fullPrompt);

            // Strip any markdown code fences the model might still add
            const cleaned = text
                .replace(/```json\s*/gi, '')
                .replace(/```\s*/g, '')
                .trim();

            const parsed = JSON.parse(cleaned);
            console.log(`✅ Script Analysis succeeded with model "${modelName}".`);
            return parsed;

        } catch (err) {
            lastError = err;
            const status = err.response?.status;
            const errorMsg = err.response?.data?.error?.message || err.message;
            console.warn(`⚠️ Model "${modelName}" failed (HTTP ${status || 'N/A'}): ${errorMsg}`);

            // Retryable errors — move to next model
            if (status === 503 || status === 429 || status === 404 || status === 500) {
                continue;
            }
            if (err instanceof SyntaxError) {
                continue;
            }
            if (err.code === 'ECONNREFUSED' || err.code === 'ETIMEDOUT' || err.code === 'ENOTFOUND') {
                continue;
            }
            // Non-retryable (e.g. 400 bad key) — stop
            throw new Error(`Gemini API Error: ${errorMsg}`);
        }
    }

    console.error('❌ All Gemini models failed. Last error:', lastError?.message);
    throw new Error('All AI models are currently unavailable. Please try again in a few minutes.');
};

module.exports = { analyzeScript };
