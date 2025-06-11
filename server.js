import express from "express";
import dotenv from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { RunnableSequence } from "@langchain/core/runnables";

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

const vectorstore = await Chroma.fromExistingCollection(embeddings, {
  collectionName: "powerbi",
  url: "http://localhost:8000",
  embeddingFunction: embeddings,
});

// Setup retrieval + response chain using RunnableSequence
const retriever = vectorstore.asRetriever();

const chain = RunnableSequence.from([
  async (question) => {
    const docs = await retriever.getRelevantDocuments(question);
    return {
      input: question,
      context: docs.map(doc => doc.pageContent).join("\n"),
    };
  },
  async ({ input, context }) => {
    const response = await model.invoke([
      {
        role: "user",
        content: `Context:\n${context}\n\nQuestion: ${input}`,
      },
    ]);
    return { text: response.content };
  },
]);

// Health check route
app.get("/", (req, res) => {
  res.send("✅ LangChain Gemini API is running");
});

// POST /ask endpoint
app.post("/ask", async (req, res) => {
  const { question } = req.body;

  if (!question) {
    return res.status(400).json({ error: "Missing question" });
  }

  try {
    const result = await chain.invoke(question);
    res.json({ answer: result.text });
  } catch (error) {
    console.error("❌ Error:", error);
    res.status(500).json({ error: "Failed to generate answer" });
  }
});

app.listen(3000, () => {
  console.log("✅ LangChain Gemini API running at http://localhost:3000");
});
