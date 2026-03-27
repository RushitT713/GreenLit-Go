const { GoogleGenerativeAI } = require('@google/generative-ai');

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

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
 * Analyze script text using Gemini AI
 * @param {string} scriptText - The extracted text from the uploaded script
 * @returns {Object} The structured analysis result
 */
const analyzeScript = async (scriptText) => {
    // Truncate to ~20,000 chars to stay within free-tier token limits (approx 5k tokens)
    const truncated = scriptText.length > 20000
        ? scriptText.substring(0, 20000) + '\n\n[...script truncated for analysis...]'
        : scriptText;

    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

    const result = await model.generateContent(ANALYSIS_PROMPT + truncated);
    const response = result.response;
    const text = response.text();

    // Strip any markdown code fences the model might still add
    const cleaned = text
        .replace(/```json\s*/gi, '')
        .replace(/```\s*/g, '')
        .trim();

    try {
        return JSON.parse(cleaned);
    } catch (parseError) {
        console.error('Gemini returned non-JSON response:', cleaned.substring(0, 200));
        throw new Error('AI returned an invalid response. Please try again.');
    }
};

module.exports = { analyzeScript };
