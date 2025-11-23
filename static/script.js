document.getElementById("detectBtn").addEventListener("click", async function () {
  const msg = document.getElementById("messageInput").value.trim();
  if (msg === "") return alert("Message cannot be empty");

  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: msg })
  });

  const data = await response.json();

  const resultBox = document.getElementById("resultBox");
  const resultText = document.getElementById("resultText");

  resultBox.classList.remove("hidden");
  resultBox.className = data.prediction === "SPAM" ? "spam" : "ham";
  resultText.innerText =
    data.prediction === "SPAM" ? "🚨 SPAM MESSAGE DETECTED!" : "✔ SAFE MESSAGE";

  addToHistory(msg, data.prediction);
});

function addToHistory(msg, prediction) {
  const list = document.getElementById("historyList");
  const li = document.createElement("li");
  li.innerHTML = `<strong>${prediction}:</strong> ${msg}`;
  li.style.borderLeft = `4px solid ${prediction === "SPAM" ? "#ff4d4d" : "#4caf50"}`;
  list.prepend(li);
}
console.log("JS FILE LOADED");
