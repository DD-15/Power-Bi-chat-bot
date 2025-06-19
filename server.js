import express from "express";
import dotenv from "dotenv";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { RunnableSequence } from "@langchain/core/runnables";
import { Document } from "@langchain/core/documents";

dotenv.config();
const app = express();
app.use(express.json());

// Initialize Gemini model
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-2.5-flash",
  temperature: 0.3,
});

// Use Gemini's embedding model
const rawEmbedder = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
});

const embeddings = {
  embedQuery: async (text) => {
    const result = await rawEmbedder.embedQuery(text);
    if (result && Array.isArray(result.values)) return [result.values];
    if (Array.isArray(result)) return [result];
    throw new Error("❌ embedQuery returned unexpected format");
  },
  embedDocuments: async (docs) => {
    const results = await rawEmbedder.embedDocuments(docs);
    return results.map((r, i) => {
      if (r && Array.isArray(r.values)) return r.values;
      if (Array.isArray(r)) return r;
      throw new Error(`❌ embedDocuments doc ${i} invalid format`);
    });
  },
};

// Ingest block (run only once)
const ingestDocuments = async () => {
  const docs = [
    new Document({
      pageContent: "Reference ID: r4J9H6M\nDatetime: 2025-06-04",
      metadata: { "Reference ID": "r4J9H6M", Datetime: "2025-06-04" },
    }),
    new Document({
      pageContent: "Reference ID: stgVtdq\nDatetime: 2025-05-31",
      metadata: { "Reference ID": "stgVtdq", Datetime: "2025-05-31" },
    }),
    new Document({
      pageContent: "Reference ID: zxYp31k\nDatetime: 2025-06-01",
      metadata: { "Reference ID": "zxYp31k", Datetime: "2025-06-01" },
    }),
  ];

  const vectorstore = await Chroma.fromDocuments(docs, embeddings, {
    collectionName: "powerbi", // ✅ Matches the collection used in loadExistingVectorstore
    url: "http://localhost:8000",
  });

  console.log("✅ Vectorstore initialized with fresh documents");
  return vectorstore;
};


// Load from existing Chroma collection
const loadExistingVectorstore = async () => {
  const vectorstore = await Chroma.fromExistingCollection(embeddings, {
    collectionName: "powerbi",
    url: "http://localhost:8000",
    embeddingFunction: embeddings,
  });

  console.log("✅ Vectorstore loaded from existing collection");
  return vectorstore;
};

const main = async () => {
  // ⚠️ Use only one: comment/uncomment accordingly
  // const vectorstore = await ingestDocuments();
  const vectorstore = await loadExistingVectorstore();

  // ✅ Debug log (after vectorstore is ready)
  const results = await vectorstore.similaritySearch("reference id", 10);
  console.log("🔍 Retrieved documents:", results.map(doc => doc.pageContent));

  const retriever = vectorstore.asRetriever();

  // Chain: question → retrieve → format → Gemini
  const chain = RunnableSequence.from([
    async ({ input }) => {
      const docs = await retriever.invoke(input);

      const metadataSummary = docs.map((doc, i) => {
        const ref = doc.metadata?.["Reference ID"] || `Doc${i + 1}`;
        const dt = doc.metadata?.Datetime || "N/A";
        return `- Reference ID: ${ref}, Date: ${dt}`;
      }).join("\n");

      const context = docs.map((doc, i) => {
        const ref = doc.metadata?.["Reference ID"] || `Doc${i + 1}`;
        const dt = doc.metadata?.Datetime || "N/A";
        return `---\nReference ID: ${ref}\nDate: ${dt}\n---`;
      }).join("\n");

      return { input, context, metadataSummary };
    },
    async ({ input, context, metadataSummary }) => {
      const response = await model.invoke([
        {
          role: "user",
          content: `
You are a data analyst assistant working with Power BI usage logs. Each record includes only a Reference ID and a Datetime.

Based on the records retrieved below, help answer the user's analytics question.

Metadata Summary:
${metadataSummary}

Context:
${context}

Question:
${input}
          `.trim(),
        },
      ]);
      return { text: response.content };
    },
  ]);

  // Health check
  app.get("/", (req, res) => {
    res.send("✅ LangChain Gemini API is running");
  });

  // Ask endpoint
  app.post("/ask", async (req, res) => {
    try {
      const { question } = req.body;
      if (!question) {
        return res.status(400).json({ error: "Missing question" });
      }

      const result = await chain.invoke({ input: question });
      return res.json({ answer: result.text });
    } catch (error) {
      console.error("❌ /ask error:", error);
      return res.status(500).json({ error: error.message || "Internal error" });
    }
  });

  app.listen(3000, () => {
    console.log("✅ LangChain Gemini API running at http://localhost:3000");
  });
};

main().catch(console.error);
