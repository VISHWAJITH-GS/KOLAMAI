function handleFileUpload(event) {
  const file = event.target.files[0];
  if (file) {
    showResults(`Uploaded file: ${file.name}`);
  }
}

function handleDragOver(event) {
  event.preventDefault();
  event.currentTarget.classList.add("dragover");
}

function handleFileDrop(event) {
  event.preventDefault();
  event.currentTarget.classList.remove("dragover");

  const file = event.dataTransfer.files[0];
  if (file) {
    showResults(`Dropped file: ${file.name}`);
  }
}

function showResults(message) {
  document.getElementById("resultsContainer").innerHTML = `
    <div class="alert alert-info">${message}</div>
  `;
}

function analyzeSample() {
  showResults("Analyzing sample Kolam...<br><strong>Result:</strong> Symmetric Kolam (Chikku Style)");
}
