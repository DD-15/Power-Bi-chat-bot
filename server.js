import express from "express";
import dotenv from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { RetrievalQAChain } from "langchain/chains";

dotenv.config();
const app = express();
app.use(express.json());

// Initialize Gemini model
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-1.5-flash",
  temperature: 0.3,
});

// Load Chroma vector store
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
});

const vectorstore = await Chroma.fromExistingCollection(
  embeddings,
  {
    collectionName: "powerbi",
    url: "http://localhost:8000",
    embeddingFunction: embeddings, // ✅ This is what fixes the crash
  }
);

// Setup RetrievalQAChain
const chain = RetrievalQAChain.fromLLM(model, vectorstore.asRetriever());

app.post("/ask", async (req, res) => {
  const { question } = req.body;

  if (!question) {
    return res.status(400).json({ error: "Missing question" });
  }

  try {
    const result = await chain.call({ query: question });
    res.json({ answer: result.text });
  } catch (error) {
    console.error("❌ Error:", error);
    res.status(500).json({ error: "Failed to generate answer" });
  }
});

app.listen(3000, () => {
  console.log("✅ LangChain Gemini API running at http://localhost:3000");
});
