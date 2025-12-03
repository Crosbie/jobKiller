// aiService.js

const { GoogleGenAI } = require("@google/genai");
const OpenAI = require("openai");
const fs = require('fs');
const path = require('path');
const https = require('https');
const crypto = require('crypto');

// --- Initialization ---

// Explicitly assign API keys from process.env (loaded by dotenv in index.js)
const GEMINI_KEY = process.env.GEMINI_API_KEY;
const OPENAI_KEY = process.env.OPENAI_API_KEY;


if (!GEMINI_KEY || !OPENAI_KEY) {
    console.error("CRITICAL: One or both API keys are missing. Ensure GEMINI_API_KEY and OPENAI_API_KEY are set in your .env file.");
}

// Initialize AI clients, explicitly passing the key
const ai = new GoogleGenAI({ apiKey: GEMINI_KEY }); // Explicit assignment
const openai = new OpenAI({ apiKey: OPENAI_KEY });   // Explicit assignment


// --- Configuration (Remains the same) ---
const GEMINI_MODEL_EXTRACT = "gemini-2.5-flash"; 
const GEMINI_MODEL_GUIDE = "gemini-2.5-pro";    
const CHATGPT_MODEL_EXTRACT = "gpt-4o-mini";    
const CHATGPT_MODEL_GUIDE = "gpt-4o";           


// 1. Define the schema for a single Feature Object
const featureObjectSchema = {
    type: "object",
    properties: {
        featureName: { type: "string", description: "A concise, descriptive title for the product update or feature." },
        featureSummary: { type: "string", description: "A 2-3 sentence technical summary of what the new feature does or how the update works." },
        potentialUseCases: {
            type: "array",
            items: { type: "string" },
            description: "List three distinct, real-world use cases that this feature could address."
        }
    },
    required: ["featureName", "featureSummary", "potentialUseCases"]
};

// 2. Define the TOP-LEVEL schema for Gemini (array)
const geminiExtractionSchema = {
    type: "array",
    items: featureObjectSchema
};

// 3. Define the TOP-LEVEL schema for OpenAI (object wrapper)
const openaiExtractionSchema = {
    type: "object",
    properties: {
        extractedFeatures: { // <-- Key to hold the array
            type: "array",
            description: "A list of all extracted technical features and their associated use cases.",
            items: featureObjectSchema
        }
    },
    required: ["extractedFeatures"]
};


// ------------------------------------------
// Extraction Function Wrapper
// ------------------------------------------

async function extractFeatures(articleText, provider = 'gemini') {
    const prompt = `
        Analyze the following Red Hat news article. Your task is to extract all new product updates, technical features, or significant value-added stories.
        If the article is primarily corporate news, opinion, or non-technical, return an empty array (or a wrapper object with an empty array).
        Otherwise, for each significant technical update, provide a descriptive name, a technical summary, and three distinct, real-world use cases.
        Return ONLY the JSON object that strictly adheres to the provided schema.

        ARTICLE CONTENT:
        ---
        ${articleText}
        ---
    `;

    try {
        let jsonText;
        let finalResult = [];

        if (provider === 'openai') {
            const response = await openai.chat.completions.create({
                model: CHATGPT_MODEL_EXTRACT,
                messages: [{ role: "user", content: prompt }],
                tools: [{
                    type: "function",
                    function: {
                        name: "extract_features",
                        description: "Extracts product features and use cases from a Red Hat announcement article.",
                        parameters: openaiExtractionSchema,
                    }
                }],
                tool_choice: { type: "function", function: { name: "extract_features" } },
                temperature: 0.1,
            });
            
            const callArguments = response.choices[0].message.tool_calls[0].function.arguments;
            jsonText = callArguments; 
            
            const parsedObject = JSON.parse(jsonText);
            finalResult = parsedObject.extractedFeatures || [];

        } else { // 'gemini'
            const response = await ai.models.generateContent({
                model: GEMINI_MODEL_EXTRACT,
                contents: prompt,
                config: {
                    responseMimeType: "application/json",
                    responseSchema: geminiExtractionSchema,
                    temperature: 0.1,
                },
            });
            jsonText = response.text.trim();
            finalResult = JSON.parse(jsonText);
        }

        return finalResult;

    } catch (error) {
        console.error(`[${provider.toUpperCase()}] API Extraction Error:`, error);
        return [];
    }
}


// ------------------------------------------
// Guide Generation Wrapper
// ------------------------------------------

async function generateGuide(feature, useCase, provider = 'gemini', industry = 'General Tech') {
    
    const guidePrompt = `
        You are a technical writer. Write a comprehensive, step-by-step technical guide for a **technical user**.
        The guide must focus on the new Red Hat feature: **${feature.featureName}** (Summary: ${feature.featureSummary}).
        
        The entire guide must be contextualized around the following real-world scenario/use case: **${useCase}**.
        
        ***CRITICAL CONTEXT: The guide must be written specifically for a company operating within the ${industry} industry. Use relevant terminology, compliance considerations (if applicable), and examples typical of that industry.***
        
        The guide must be returned as a complete HTML snippet (excluding <html>, <head>, and <body> tags, but including <h1>, <p>, <h2>, <ul>, and <code> tags).
        It should be professional, instructive, and include:
        1. An introduction relating the feature to the use case in the context of the ${industry} industry.
        2. Prerequisites (e.g., 'RHEL 9', 'OpenShift Cluster access', necessary toolchains, etc.).
        3. A section of at least 3 actionable, technical steps/commands with brief explanations.
        4. A conclusion on the value proposition for an ${industry} business.
    `;

    try {
        let responseText;

        if (provider === 'openai') {
            const response = await openai.chat.completions.create({
                model: CHATGPT_MODEL_GUIDE,
                messages: [{ role: "user", content: guidePrompt }],
                temperature: 0.7,
            });
            responseText = response.choices[0].message.content;

        } else { // 'gemini'
            const response = await ai.models.generateContent({
                model: GEMINI_MODEL_GUIDE,
                contents: guidePrompt,
                config: { temperature: 0.7 }
            });
            responseText = response.text;
        }

        return responseText.replace(/```(html)?/g, '').trim();

    } catch (error) {
        console.error(`[${provider.toUpperCase()}] API Guide Generation Error:`, error);
        return "<h3>Error Generating Guide</h3><p>Could not generate the technical guide using the selected AI provider.</p>";
    }
}

module.exports = { extractFeatures, generateGuide };