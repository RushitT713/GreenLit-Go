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

/**
 * Call Gemini REST API directly (no SDK dependency)
 */
const callGeminiREST = async (modelName, prompt) => {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${GEMINI_API_KEY}`;

    const response = await axios.post(url, {
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: {
            temperature: 0.7,
            maxOutputTokens: 4096,
        }
    }, {
        headers: { 'Content-Type': 'application/json' },
        timeout: 60000, // 60 second timeout
    });

    // Extract the text from Gemini's response structure
    const text = response.data?.candidates?.[0]?.content?.parts?.[0]?.text;
    if (!text) {
        throw new Error('Empty response from Gemini API');
    }
    return text;
};

/**
 * Analyze script text using Gemini AI with automatic model fallback
 * @param {string} scriptText - The extracted text from the uploaded script
 * @returns {Object} The structured analysis result
 */
const analyzeScript = async (scriptText) => {
    // Truncate to ~20,000 chars to stay within free-tier token limits
    const truncated = scriptText.length > 20000
        ? scriptText.substring(0, 20000) + '\n\n[...script truncated for analysis...]'
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
            // Get the HTTP status code from axios error if available
            const status = err.response?.status;
            const errorMsg = err.response?.data?.error?.message || err.message;
            console.warn(`⚠️ Model "${modelName}" failed (HTTP ${status || 'N/A'}): ${errorMsg}`);

            // If it's a retryable error (overloaded, rate limit, not found), try next model
            if (status === 503 || status === 429 || status === 404 || status === 500) {
                continue;
            }
            // JSON parse failure — try next model
            if (err instanceof SyntaxError) {
                continue;
            }
            // For network errors, try next model too
            if (err.code === 'ECONNREFUSED' || err.code === 'ETIMEDOUT' || err.code === 'ENOTFOUND') {
                continue;
            }
            // For any other error (e.g. 400 bad API key), stop
            throw new Error(`Gemini API Error: ${errorMsg}`);
        }
    }

    // If all models failed
    console.error('❌ All Gemini models failed. Last error:', lastError?.message);
    throw new Error('All AI models are currently unavailable. Please try again in a few minutes.');
};

module.exports = { analyzeScript };
