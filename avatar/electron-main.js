import { app, BrowserWindow } from "electron";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import http from "http";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let mainWindow;
let serverProcess;

function waitForServer(url, callback) {
  const retry = () => {
    http.get(url, () => callback()).on("error", () => {
      setTimeout(retry, 500);
    });
  };
  retry();
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    backgroundColor: "#0b0c0f",
    webPreferences: {
      contextIsolation: true
    }
  });

  mainWindow.loadURL("http://localhost:3000");

  // Ouvre la console automatiquement (dev)
  mainWindow.webContents.openDevTools();
}

app.whenReady().then(() => {
  console.log("🚀 Démarrage serveur Express...");

  serverProcess = spawn("node", ["intelligent_server.js"], {
    cwd: __dirname,
    stdio: "inherit"
  });

  // ⏳ Attendre que le serveur soit vraiment prêt
  waitForServer("http://localhost:3000", createWindow);
});

app.on("window-all-closed", () => {
  if (serverProcess) serverProcess.kill();
  app.quit();
});
