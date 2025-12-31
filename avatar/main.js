import * as THREE from "https://unpkg.com/three@0.164.0/build/three.module.js";
import { GLTFLoader } from "https://unpkg.com/three@0.164.0/examples/jsm/loaders/GLTFLoader.js";

// === THREE.JS / AVATAR ===

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  0.1,
  100
);
camera.position.set(0, 1.6, 3);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(2, 4, 3);
scene.add(dirLight);
scene.add(new THREE.AmbientLight(0xffffff, 0.4));

const loader = new GLTFLoader();
const clock = new THREE.Clock();

let avatarGltf = null;
let helloGltf = null;
let mixer = null;
let currentGltf = null;

function showModel(gltf) {
  if (currentGltf && currentGltf.scene.parent === scene) {
    scene.remove(currentGltf.scene);
  }
  currentGltf = gltf;
  scene.add(currentGltf.scene);

  mixer = new THREE.AnimationMixer(currentGltf.scene);
}

function playHelloAnim() {
  if (!helloGltf || !avatarGltf) {
    console.warn("hello.glb ou avatar.glb pas encore chargé");
    return;
  }

  showModel(helloGltf);

  const clips = helloGltf.animations;
  if (!clips || clips.length === 0) {
    console.warn("Pas d'animation dans hello.glb");
    return;
  }

  const clip = clips[0];
  const action = mixer.clipAction(clip);
  action.reset();
  action.setLoop(THREE.LoopOnce);
  action.clampWhenFinished = true;
  action.play();

  // Retour à l'avatar neutre après la durée du clip
  setTimeout(() => {
    showModel(avatarGltf);
  }, clip.duration * 1000);
}

// Charger l'avatar de base
loader.load(
  "avatar.glb",
  (gltf) => {
    avatarGltf = gltf;
    showModel(avatarGltf);
    console.log("avatar.glb chargé");
  },
  undefined,
  (err) => console.error("Erreur chargement avatar.glb", err)
);

// Charger l'animation hello
loader.load(
  "hello.glb",
  (gltf) => {
    helloGltf = gltf;
    console.log("hello.glb chargé");
  },
  undefined,
  (err) => console.error("Erreur chargement hello.glb", err)
);

// Boucle d'animation
function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();
  if (mixer) mixer.update(delta);
  renderer.render(scene, camera);
}
animate();

// Resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// === CAPTION & WEBSOCKET ===

const captionEl = document.getElementById("caption");
function setCaption(text) {
  captionEl.textContent = text || "";
}

const ws = new WebSocket(`ws://${location.host}`);

ws.addEventListener("open", () => {
  console.log("WS ouvert");
});

ws.addEventListener("message", (event) => {
  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch {
    return;
  }

  if (msg.type === "connected") {
    console.log("WS:", msg.message);
  }

  if (msg.type === "caption") {
    setCaption(msg.text);
  }

  if (msg.type === "play") {
    console.log("Play demandé:", msg.clip);

    // Quand le serveur envoie HELLO_LSF → lancer l'animation hello.glb
    if (msg.clip === "HELLO_LSF") {
      playHelloAnim();
    }
    // Tu pourras ajouter d'autres clips ici si besoin
  }
});

// === FORMULAIRE DE CHAT (ENVOI /chat) ===

const input = document.getElementById("chat-input");
const btn = document.getElementById("chat-send");

async function sendMessage() {
  const text = (input.value || "").trim();
  if (!text) return;

  // Appel à l'API /chat de ton serveur
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
    });

    const data = await res.json();
    console.log("Réponse /chat:", data);
    // La vraie animation sera déclenchée par le WebSocket (msg.type === "play")
    // mais on peut afficher la légende tout de suite si on veut :
    setCaption(data.caption || data.reply || "");
  } catch (err) {
    console.error("Erreur envoi /chat:", err);
  }

  input.value = "";
}

btn.addEventListener("click", sendMessage);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    sendMessage();
  }
});
