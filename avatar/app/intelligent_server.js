import express from "express";
import { WebSocketServer } from "ws";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json());
app.use(express.static(__dirname)); // sert index.html, main.js, .glb, etc.

app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  if (req.method === "OPTIONS") return res.sendStatus(200);
  next();
});

// Charger les intents
const intentsPath = path.join(__dirname, "intents_lsf.json");
let INTENTS = [];
try {
  INTENTS = JSON.parse(fs.readFileSync(intentsPath, "utf-8"));
  console.log("Intents chargés:", INTENTS.length);
} catch (err) {
  console.warn("intents_lsf.json introuvable ou invalide", err);
}

function matchIntent(text) {
  const t = (text || "").toLowerCase();
  for (const it of INTENTS) {
    for (const ex of it.examples || []) {
      if (t.includes(ex.toLowerCase())) return it;
    }
  }
  return (
    INTENTS.find((x) => x.intent === "fallback") || {
      reply: "Un instant s’il vous plaît.",
      clip: "WAIT_PLEASE_LSF",
    }
  );
}

app.post("/chat", (req, res) => {
  const message = req.body?.message || "";
  const chosen = matchIntent(message);

  const replyObj = {
    reply: chosen.reply || "Un instant s’il vous plaît.",
    clip: chosen.clip || "WAIT_PLEASE_LSF",
    caption: chosen.reply || "",
  };

  // Broadcast via WebSocket à tous les clients connectés
  try {
    if (wsServer && wsServer.clients) {
      const payload = JSON.stringify({
        type: "play",
        clip: replyObj.clip,
        speed: 1.0,
        loop: false,
      });
      const captionMsg = JSON.stringify({
        type: "caption",
        text: replyObj.caption,
      });

      wsServer.clients.forEach((client) => {
        if (client.readyState === 1) {
          client.send(payload);
          client.send(captionMsg);
        }
      });
    }
  } catch (err) {
    console.error("Erreur broadcast WS:", err);
  }

  res.json(replyObj);
});

// Optionnel : route racine -> index.html
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

const server = app.listen(3000, () =>
  console.log("✅ Serveur sur http://localhost:3000")
);
const wsServer = new WebSocketServer({ server });

wsServer.on("connection", (socket, req) => {
  console.log("Nouvelle connexion WS");
  socket.send(
    JSON.stringify({ type: "connected", message: "Bienvenue (WS connecté)" })
  );
  socket.on("message", (m) => {
    console.log("Message reçu WS:", m.toString());
  });
  socket.on("close", () => console.log("WS fermé"));
});
app.use(express.static(__dirname));

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

