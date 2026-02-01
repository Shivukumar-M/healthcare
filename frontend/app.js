async function analyzeImage() {
  const input = document.getElementById("imageInput");
  const loading = document.getElementById("loading");
  const result = document.getElementById("result");

  if (!input.files.length) {
    alert("Please upload a medical image.");
    return;
  }

  const formData = new FormData();
  formData.append("file", input.files[0]);

  loading.classList.remove("hidden");
  result.classList.add("hidden");

  try {
    const response = await fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (data.status !== "success") {
      alert("Unable to analyze image.");
      loading.classList.add("hidden");
      return;
    }

    document.getElementById("imageType").innerText =
      data.detected_image_type.replace("_", " ");

    document.getElementById("disease").innerText =
      "Detected Condition: " + data.detected_disease;

    document.getElementById("confidence").innerText =
      "Confidence: " + data.confidence + "%";

    // confidence bar
    document.getElementById("confidenceBar").style.width =
      Math.min(data.confidence, 100) + "%";

    // probabilities
    document.getElementById("allProbs").innerText =
      JSON.stringify(data.all_probabilities, null, 2);

    // ✅ AI EXPLANATION (THIS WAS MISSING / BROKEN)
    document.getElementById("aiExplanation").innerText =
      data.ai_explanation || "No explanation available.";

    loading.classList.add("hidden");
    result.classList.remove("hidden");

  } catch (error) {
    loading.classList.add("hidden");
    alert("Error connecting to backend.");
    console.error(error);
  }
}
