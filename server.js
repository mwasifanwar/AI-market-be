const express = require("express");
const dotenv = require("dotenv");
const axios = require("axios");
const cors = require("cors");
const helmet = require("helmet");
const fs = require("fs");

dotenv.config();
const app = express();
const PORT = process.env.PORT || 5000;
const GROQ_API_KEY = process.env.GROQ_API_KEY;

// Validate API Key
if (!GROQ_API_KEY) {
    throw new Error("Missing GROQ_API_KEY in environment variables");
}

app.use(express.json());
app.use(cors({
    origin: ["*"], // Replace with your actual frontend URL
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type", "Authorization"]
}));

app.use(helmet());

// Configure Groq API client
const groqAPI = axios.create({
    baseURL: "https://api.groq.com/openai/v1",
    headers: {
        "Authorization": `Bearer ${GROQ_API_KEY}`,
        "Content-Type": "application/json",
    },
    timeout: 15000 // 15-second timeout
});

app.post("/analyze-market", async (req, res) => {
    try {
        const { industry, region, depth } = req.body;

        if (!industry || !region || !depth) {
            return res.status(400).json({ error: "❌ Missing required parameters: industry, region, depth." });
        }

        // ✅ Corrected API Payload
        const requestBody = {
            model: "llama-3.3-70b-versatile", // ✅ Ensure correct model
            messages: [
                { role: "system", content: "You are a market analyst AI." },
                { role: "user", content: `Perform an in-depth, data-driven market analysis for the ${industry} industry in ${region} with a ${depth} level of detail.
                
                - The data must be **latest and authentic**, backed by real market trends and research.
                - Provide only **genuine** insights; no fabricated or vague estimates.
                - Ensure numbers, statistics, and recommendations are **credible**.
                - Return **strictly** in this JSON format without any additional text:
                
                {
                    "graphData": {
                        "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                        "values": [X1, X2, X3, X4, X5, X6]  
                    },
                    "keyFindings": {
                        "marketGrowth": "XX%",  
                        "competitionLevel": "Low/Medium/High",
                        "entryBarriers": "Low/Medium/High"
                    },
                    "recommendations": [
                        "Recommendation 1",
                        "Recommendation 2",
                        "Recommendation 3"
                    ]
                }` }
            ],
            temperature: 0.3,
            max_tokens: 800
        };

        // ✅ Retry Function (Handles API Failures & Timeout)
        const fetchAnalysis = async (attempt = 1) => {
            try {
                console.log(`⏳ Attempt ${attempt} - Fetching Market Analysis...`);
                const response = await groqAPI.post("/chat/completions", requestBody, { timeout: 60000 }); // ✅ 60s Timeout
                console.log("✅ API Response:", response.data);

                const responseText = response.data.choices[0]?.message?.content || "";
                const jsonMatch = responseText.match(/\{[\s\S]*\}/);
                if (!jsonMatch) throw new Error("❌ No valid JSON found in response.");

                return JSON.parse(jsonMatch[0]); // ✅ Return Parsed JSON

            } catch (error) {
                console.error(`⚠️ Attempt ${attempt} failed:`, error.message);
                if (attempt < 3) {
                    return await fetchAnalysis(attempt + 1); // 🔄 Retry up to 3 times
                }
                throw error; // ❌ Final Failure
            }
        };

        // ✅ Fetch Analysis Data (with retries)
        const analysis = await fetchAnalysis();

        // ✅ Send JSON Response
        return res.json(analysis);

    } catch (error) {
        console.error("❌ Market analysis error:", error.message);
        return res.status(500).json({ error: "Failed to fetch market analysis after multiple attempts." });
    }
});



/* 🔹 AI Market Assistant */
app.post("/ai-market-assistant", async (req, res) => {
    try {
        const { userQuery, industry } = req.body;

        if (!userQuery || !industry) {
            return res.status(400).json({ error: "❌ Missing required parameters: userQuery, industry." });
        }

        const prompt = `Provide a **concise, latest, data-backed response** to the following market-related question.
        
        - **Only return the direct answer. Do not add intros, explanations, or filler text.**
        - **Ensure insights are based on the latest trends, market data, and statistics.**
        - **If the question is unrelated to ${industry}, respond with:** "Please update the inputs on the previous page for new analysis."
        
        User's Question: "${userQuery}"`;

        const requestBody = {
            model: "llama-3.3-70b-versatile",
            messages: [
                { role: "system", content: "You are a strict market analysis AI. Provide only direct, data-backed insights." },
                { role: "user", content: prompt }
            ],
            temperature: 0.3,
            max_tokens: 200
        };

        // ✅ Retry Mechanism for Stability (3 Attempts)
        const fetchAIResponse = async (retries = 3) => {
            for (let i = 0; i < retries; i++) {
                try {
                    const response = await groqAPI.post("/chat/completions", requestBody, { timeout: 60000 }); // ⏳ 60s Timeout
                    console.log("✅ AI Assistant Response:", response.data);

                    // ✅ Extract the assistant's reply
                    let aiResponse = response.data.choices[0]?.message?.content || "No relevant insights found.";

                    return aiResponse;
                } catch (err) {
                    console.error(`⚠️ Attempt ${i + 1} failed: ${err.message}`);
                    if (i === retries - 1) throw err; // Fail after 3 attempts
                }
            }
        };

        // ✅ Get AI Response
        const aiResponse = await fetchAIResponse();

        // ✅ Send JSON Response
        return res.json({ assistantResponse: aiResponse });

    } catch (error) {
        console.error("❌ AI Assistant error:", error.response?.data || error.message);
        return res.status(500).json({ error: "Failed to fetch assistant response after multiple attempts." });
    }
});


/* 🔹 2️⃣ Competitor Analysis */
app.post("/analyze-competitor", async (req, res) => {
    try {
        const { competitorName } = req.body;

        if (!competitorName) {
            return res.status(400).json({ error: "❌ Missing required parameter: competitorName." });
        }

        const prompt = `Perform a strict, **data-backed competitor analysis** for ${competitorName}.
        
        🔹 **STRICT REQUIREMENTS:**
        - **All insights must be the latest and 100% authentic.** No outdated, vague, or fabricated data.
        - **Use real, fact-based industry trends.** No assumptions or estimates.
        - **Return the response in the exact JSON format below.** No extra text or explanations.
        - **Each field should be only 2-3 words long, exactly matching the UI.**
        
        **RESPONSE FORMAT (Strict Length Rules):**
        {
            "marketPosition": "Emerging Leader / Market Leader / Challenger / Niche Player",
            "strategyAnalysis": {
                "pricing": "Premium pricing / Budget-friendly / Value-based",
                "marketing": "Digital-first / Influencer-driven / Traditional media",
                "product": "Rapid innovation / High customization / Mass production"
            },
            "strengths": [
                "Strong brand identity",
                "Innovative technology",
                "Customer service"
            ],
            "weaknesses": [
                "Limited market reach",
                "High operational costs",
                "Product gaps"
            ]
        }`;

        const requestBody = {
            model: "llama-3.3-70b-versatile",
            messages: [
                { role: "system", content: "You are an AI competitor analysis expert. Provide only **latest, fact-based, and structured** insights in JSON format." },
                { role: "user", content: prompt }
            ],
            temperature: 0.3,
            max_tokens: 600
        };

        // 🔹 Fetch AI Response
        const response = await groqAPI.post("/chat/completions", requestBody);
        console.log("✅ AI Competitor Analysis Response:", response.data);

        // 🔹 Extract and Parse JSON Response
        const responseText = response.data.choices[0]?.message?.content || "";
        const jsonMatch = responseText.match(/\{[\s\S]*\}/);

        if (!jsonMatch) {
            console.error("❌ No valid JSON found in response:", responseText);
            return res.status(500).json({ error: "Failed to extract competitor analysis data." });
        }

        let competitorAnalysis = JSON.parse(jsonMatch[0]);

        // 🔹 Validate Length of Each Field
        const truncateToMaxLength = (text, maxLength) => text.split(" ").slice(0, maxLength).join(" ");

        competitorAnalysis.marketPosition = truncateToMaxLength(competitorAnalysis.marketPosition, 3);
        competitorAnalysis.strategyAnalysis.pricing = truncateToMaxLength(competitorAnalysis.strategyAnalysis.pricing, 3);
        competitorAnalysis.strategyAnalysis.marketing = truncateToMaxLength(competitorAnalysis.strategyAnalysis.marketing, 3);
        competitorAnalysis.strategyAnalysis.product = truncateToMaxLength(competitorAnalysis.strategyAnalysis.product, 3);
        
        competitorAnalysis.strengths = competitorAnalysis.strengths.map(strength => truncateToMaxLength(strength, 3));
        competitorAnalysis.weaknesses = competitorAnalysis.weaknesses.map(weakness => truncateToMaxLength(weakness, 3));

        // 🔹 Send JSON Response
        return res.json(competitorAnalysis);

    } catch (error) {
        console.error("❌ Competitor analysis error:", error.response?.data || error.message);
        return res.status(500).json({ error: "Failed to analyze competitor." });
    }
});


/* 🔹 3️⃣ Market Predictions */
app.post("/predict-market", async (req, res) => {
    try {
        const { industry } = req.body;

        if (!industry) {
            return res.status(400).json({ error: "❌ Missing required parameter: industry." });
        }

        const prompt = `Provide a **highly accurate, data-backed 5-year market prediction** for the ${industry} industry.
        
        🔹 **STRICT REQUIREMENTS:**
        - **All insights must be the latest and 100% authentic.** No outdated, vague, or fabricated data.
        - **Use real, fact-based industry trends.** No assumptions or estimates.
        - **Ensure the response strictly follows the JSON format below.** No extra text or explanations.

        **RESPONSE FORMAT:**
        {
            "growthScore": {
                "score": 85,
                "trend": "Rising / Stable / Declining"
            },
            "futureBusinessModels": [
                "Short phrase 1",
                "Short phrase 2",
                "Short phrase 3"
            ],
            "growthProjection": {
                "years": ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"],
                "values": [X1, X2, X3, X4, X5]
            }
        }`;

        const requestBody = {
            model: "llama-3.3-70b-versatile",
            messages: [
                { role: "system", content: "You are an AI market prediction expert. Provide only **latest, fact-based, and structured** insights in JSON format." },
                { role: "user", content: prompt }
            ],
            temperature: 0.3,
            max_tokens: 600
        };

        // 🔹 Fetch AI Response
        const response = await groqAPI.post("/chat/completions", requestBody);
        console.log("✅ Market Prediction Response:", response.data);

        // 🔹 Extract and Parse JSON Response
        const responseText = response.data.choices[0]?.message?.content || "";
        const jsonMatch = responseText.match(/\{[\s\S]*\}/);

        if (!jsonMatch) {
            console.error("❌ No valid JSON found in response:", responseText);
            return res.status(500).json({ error: "Failed to extract market prediction data." });
        }

        const marketPrediction = JSON.parse(jsonMatch[0]);

        // 🔹 Send JSON Response
        return res.json(marketPrediction);

    } catch (error) {
        console.error("❌ Market prediction error:", error.response?.data || error.message);
        return res.status(500).json({ error: "Failed to generate market predictions." });
    }
});


/* 🔹 4️⃣ AI Business Strategy Generator */
app.post("/generate-business-strategy", async (req, res) => {
    try {
        const { businessIdea } = req.body;

        if (!businessIdea) {
            return res.status(400).json({ error: "❌ Missing required parameter: businessIdea." });
        }

        const prompt = `Generate a **highly structured** business strategy for "${businessIdea}" following these **STRICT GUIDELINES**:
        
        - **All insights must be the latest and data-backed.** No vague or fabricated data.
        - **Use short, precise, and actionable phrases** as shown in the format below.
        - **Do NOT add unnecessary explanations or intro text.**
        - **Ensure formatting is strictly followed.**

        **RESPONSE FORMAT (Strict JSON, no extra text):**
        {
            "targetAudience": [
                "Segment 1 (e.g., Tech-savvy professionals 25-45)",
                "Segment 2 (e.g., Small business owners)",
                "Segment 3 (e.g., Remote workers)"
            ],
            "productStrategy": [
                "MVP launch within 3 months",
                "Iterative development based on user feedback",
                "Premium features for enterprise clients"
            ],
            "financialProjections": {
                "estimatedROI": "XXX% in first year",
                "breakEvenPoint": "X months",
                "initialInvestment": "$XX,XXX - $XX,XXX"
            },
            "marketingPlan": [
                "Content marketing focus on LinkedIn and Medium",
                "Early adopter program with X% discount",
                "Partnership with industry influencers"
            ]
        }`;

        const requestBody = {
            model: "llama-3.3-70b-versatile",
            messages: [
                { role: "system", content: "You are an AI business strategy expert. Provide only structured, latest, and concise responses in JSON format." },
                { role: "user", content: prompt }
            ],
            temperature: 0.3,
            max_tokens: 700
        };

        // ✅ Fetch AI Response
        const response = await groqAPI.post("/chat/completions", requestBody);
        console.log("✅ AI Business Strategy Response:", response.data);

        // ✅ Extract and Parse JSON Response
        const responseText = response.data.choices[0]?.message?.content || "";
        const jsonMatch = responseText.match(/\{[\s\S]*\}/);

        if (!jsonMatch) {
            console.error("❌ No valid JSON found in response:", responseText);
            return res.status(500).json({ error: "Failed to extract business strategy data." });
        }

        const businessStrategy = JSON.parse(jsonMatch[0]);

        // ✅ Send JSON Response
        return res.json(businessStrategy);

    } catch (error) {
        console.error("❌ Business strategy error:", error.response?.data || error.message);
        return res.status(500).json({ error: "Failed to generate business strategy." });
    }
});


/* 🔹 5️⃣ AI Content Strategy Generator */
app.post("/generate-content-strategy", async (req, res) => {
    try {
        const { businessIdea } = req.body;

        if (!businessIdea) {
            return res.status(400).json({ error: "❌ Missing required parameter: businessIdea." });
        }

        const prompt = `Generate a **100% latest, fact-based AI content strategy** for the business niche: "${businessIdea}". 
        
        🔹 **STRICT FORMAT REQUIREMENTS**:
        - **All insights MUST be the latest and authentic.** No outdated, vague, or fabricated data.
        - **Strictly follow the structure below. Do NOT add any intros or explanations.**
        - **Ensure descriptions are short and to the point.**
        
        **RESPONSE FORMAT (STRICTLY FOLLOW THIS STRUCTURE)**:
        {
            "targetKeywords": [
                "Keyword 1",
                "Keyword 2",
                "Keyword 3",
                "Keyword 4"
            ],
            "blogTopics": [
                "Blog Title 1",
                "Blog Title 2",
                "Blog Title 3",
                "Blog Title 4"
            ],
            "socialMediaStrategy": {
                "LinkedIn": {
                    "description": "Short description",
                    "frequency": "X posts/week"
                },
                "Twitter": {
                    "description": "Short description",
                    "frequency": "Daily"
                },
                "Medium": {
                    "description": "Short description",
                    "frequency": "X posts/week"
                }
            },
            "viralMarketingAngles": [
                "Angle 1",
                "Angle 2",
                "Angle 3",
                "Angle 4"
            ],
            "contentPlatforms": {
                "Company Blog": {
                    "traffic": "Traffic Level",
                    "conversion": "X% conversion"
                },
                "Medium Publication": {
                    "traffic": "Traffic Level",
                    "conversion": "X% conversion"
                },
                "LinkedIn Articles": {
                    "traffic": "Traffic Level",
                    "conversion": "X% conversion"
                }
            }
        }`;

        const requestBody = {
            model: "llama-3.3-70b-versatile",
            messages: [
                { role: "system", content: "You are a professional AI content strategist. Provide only **latest, concise, fact-based insights** in JSON format." },
                { role: "user", content: prompt }
            ],
            temperature: 0.3,
            max_tokens: 800
        };

        // 🔹 Fetch AI Response
        const response = await groqAPI.post("/chat/completions", requestBody);
        console.log("✅ AI Content Strategy Response:", response.data);

        // 🔹 Extract and Parse JSON Response
        const responseText = response.data.choices[0]?.message?.content || "";
        const jsonMatch = responseText.match(/\{[\s\S]*\}/);

        if (!jsonMatch) {
            console.error("❌ No valid JSON found in response:", responseText);
            return res.status(500).json({ error: "Failed to extract content strategy data." });
        }

        const contentStrategy = JSON.parse(jsonMatch[0]);

        // 🔹 Send JSON Response
        return res.json(contentStrategy);

    } catch (error) {
        console.error("❌ Content Strategy error:", error.response?.data || error.message);
        return res.status(500).json({ error: "Failed to generate content strategy." });
    }
});


/* 🔹 6️⃣ Start the Server */
app.listen(PORT, () => {
    console.log(`✅ Server running on http://localhost:${PORT}`);
});
