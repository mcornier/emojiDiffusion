// Imports from main.js:
// DOM Elements: trainingStatusEl, trainingModeSelect, epochsInput, learningRateInput, batchSizeInput, startTrainingBtn, lossChartCanvas, lossChartCtx
// Global state/data: diffusionModel, dataset, EMOJI_TO_INDEX
// Constants: TIMESTEPS
// Functions: forwardNoise (which itself needs alphasCumprod from main.js)

// Module-local state
let trainingHistory = [];

// --- Training Functions ---

export function drawLossChart(currentTrainingHistory, lossChartCanvas, lossChartCtx) {
    const margin = {top: 20, right: 20, bottom: 40, left: 40};
    const width = lossChartCanvas.width - margin.left - margin.right;
    const height = lossChartCanvas.height - margin.top - margin.bottom;
    lossChartCtx.clearRect(0, 0, lossChartCanvas.width, lossChartCanvas.height);
    lossChartCtx.fillStyle = "#f9f9f9"; // Background color for the chart
    lossChartCtx.fillRect(0,0,lossChartCanvas.width, lossChartCanvas.height);

    if (currentTrainingHistory.length === 0) {
        lossChartCtx.fillStyle = 'black';
        lossChartCtx.textAlign = 'center';
        lossChartCtx.fillText("Aucune donnée d'entraînement.", lossChartCanvas.width / 2, lossChartCanvas.height / 2);
        return;
    }

    const maxLoss = Math.max(...currentTrainingHistory, 0.1); // Ensure a minimum y-axis scale

    // Draw axes
    lossChartCtx.strokeStyle = '#ccc';
    lossChartCtx.lineWidth = 1;
    lossChartCtx.beginPath();
    lossChartCtx.moveTo(margin.left, margin.top);
    lossChartCtx.lineTo(margin.left, height + margin.top);
    lossChartCtx.lineTo(width + margin.left, height + margin.top);
    lossChartCtx.stroke();

    // Draw Y-axis labels
    lossChartCtx.fillStyle = 'grey';
    lossChartCtx.textAlign = 'right';
    for(let i = 0; i <= 5; i++) { // 5 labels on y-axis
        const y = margin.top + (height - (i/5 * height));
        lossChartCtx.fillText((maxLoss * i/5).toFixed(2), margin.left - 5, y + 3);
    }

    // Draw X-axis labels (epochs)
    lossChartCtx.textAlign = 'center';
    const numEpochLabels = Math.min(currentTrainingHistory.length, 10); // Max 10 labels for epochs
    for(let i = 0; i < numEpochLabels; i++) {
        const epochIdx = Math.floor(i * (currentTrainingHistory.length / numEpochLabels));
        const x = margin.left + ((epochIdx / (currentTrainingHistory.length -1 || 1)) * width);
        lossChartCtx.fillText(epochIdx + 1, x, height + margin.top + 15);
    }
    lossChartCtx.fillText("Époque", width / 2 + margin.left, height + margin.top + 30);


    // Draw loss line
    lossChartCtx.beginPath();
    lossChartCtx.strokeStyle = 'red';
    lossChartCtx.lineWidth = 2;
    currentTrainingHistory.forEach((loss, index) => {
        const x = margin.left + (index / (currentTrainingHistory.length - 1 || 1)) * width;
        const y = margin.top + height - (loss / maxLoss) * height;
        if (index === 0) lossChartCtx.moveTo(x, y);
        else lossChartCtx.lineTo(x, y);
    });
    lossChartCtx.stroke();
}


export function setupTraining(
    // DOM Elements
    startTrainingBtn,
    trainingStatusEl,
    trainingModeSelect,
    epochsInput,
    learningRateInput,
    batchSizeInput,
    lossChartCanvas,
    lossChartCtx,
    // Callbacks to access shared state and functions from main.js
    getContext
) {
    startTrainingBtn.addEventListener('click', async () => {
        const {
            diffusionModel,
            dataset,
            EMOJI_TO_INDEX,
            TIMESTEPS,
            forwardNoiseFunc // Renamed to avoid conflict
        } = getContext();

        if (!diffusionModel) {
            trainingStatusEl.textContent = "Erreur: Modèle non défini. Générez un dataset d'abord.";
            return;
        }
        if (!dataset || dataset.tensors.length === 0) {
            trainingStatusEl.textContent = "Erreur: Le dataset est vide.";
            return;
        }

        startTrainingBtn.disabled = true;
        trainingStatusEl.textContent = "Préparation de l'entraînement...";
        trainingHistory = []; // Reset history for this training session
        drawLossChart(trainingHistory, lossChartCanvas, lossChartCtx);

        const trainingMode = trainingModeSelect.value;
        const epochs = parseInt(epochsInput.value);
        const learningRate = parseFloat(learningRateInput.value);
        const batchSize = parseInt(batchSizeInput.value);
        const optimizer = tf.train.adam(learningRate);

        const allIndices = Array.from({length: dataset.tensors.length}, (_, i) => i);
        const numBatches = Math.ceil(allIndices.length / batchSize);

        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochLoss = 0;
            tf.util.shuffle(allIndices);

            for (let i = 0; i < numBatches; i++) {
                const batchIndices = allIndices.slice(i * batchSize, (i + 1) * batchSize);
                if (batchIndices.length === 0) continue;

                const calculateLoss = () => { // Renamed from f to be more descriptive
                    return tf.tidy(() => {
                        const batchImagesArray = batchIndices.map(idx => dataset.tensors[idx]);
                        if (batchImagesArray.some(t => !t || t.isDisposed)) {
                            console.error("One or more tensors in batchImagesArray are disposed or undefined.");
                            // Handle this error, perhaps by skipping the batch or stopping training
                            return tf.scalar(0); // Or some other way to signal error
                        }
                        const batchImages = tf.stack(batchImagesArray);

                        const batchEmojis = batchIndices.map(idx => dataset.emojis[idx]);
                        const batchContextsArray = batchEmojis.map(emoji => [EMOJI_TO_INDEX[emoji] || 0]); // Default to MASK if emoji not found
                        const batchContexts = tf.tensor2d(batchContextsArray, [batchIndices.length, 1], 'int32');

                        const t = tf.randomUniform([batchIndices.length, 1], 0, TIMESTEPS, 'int32');
                        let loss;

                        if (trainingMode === 'diffusion') {
                            const { noisedImage, noise } = forwardNoiseFunc(batchImages, t); // Use passed forwardNoise
                            const predictedNoise = diffusionModel.apply([noisedImage, t, batchContexts]);
                            loss = tf.losses.meanSquaredError(noise, predictedNoise).mean();
                        } else { // autoencoder mode
                             const t_zeros = tf.zerosLike(t);
                             const context_zeros = tf.zerosLike(batchContexts); // Use MASK context for autoencoder
                             const predictedImages = diffusionModel.apply([batchImages, t_zeros, context_zeros]);
                             loss = tf.losses.meanSquaredError(batchImages, predictedImages).mean();
                        }
                        return loss;
                    });
                };

                // Log the structure of trainableWeights for diagnostics
                console.log('Trainable weights for the model:', diffusionModel.trainableWeights);
                if (diffusionModel.trainableWeights && Array.isArray(diffusionModel.trainableWeights)) {
                    diffusionModel.trainableWeights.forEach((w, idx) => {
                        console.log(`Weight ${idx}: Name: ${w.name}, Shape: ${w.shape}, Is tf.Variable: ${w instanceof tf.Variable}`);
                    });
                }

                let varList = [];
                if (diffusionModel && Array.isArray(diffusionModel.trainableWeights)) {
                    // Filter to ensure all elements are actual tf.Variable instances
                    varList = diffusionModel.trainableWeights.filter(w => w instanceof tf.Variable);

                    if (varList.length !== diffusionModel.trainableWeights.length) {
                        console.warn("Some trainable weights were not tf.Variable instances and were filtered out. Original count:", diffusionModel.trainableWeights.length, "Filtered count:", varList.length);
                        // This might indicate an issue with how weights are registered or collected if potentially trainable weights are excluded.
                    }
                } else {
                    console.error("diffusionModel.trainableWeights is not an array or diffusionModel is undefined.");
                    trainingStatusEl.textContent = "Erreur: Liste de poids entraînables invalide.";
                    startTrainingBtn.disabled = false;
                    return; // Exit if varList cannot be formed
                }

                if (varList.length === 0 && diffusionModel.trainableWeights && diffusionModel.trainableWeights.length > 0) {
                    console.error("No tf.Variable instances found in trainableWeights after filtering. This likely means the structure of trainableWeights is not a flat list of tf.Variable.");
                    // At this point, one might need to inspect the logged structure and implement a more specific mapping.
                    // For example, if weights are wrapped like {name: 'weight_name', variable: tf.Variable_instance},
                    // the mapping would be: diffusionModel.trainableWeights.map(w_obj => w_obj.variable).filter(v => v instanceof tf.Variable);
                    // However, without logs, this is speculative. The current filter is the most direct approach based on the error.
                    trainingStatusEl.textContent = "Erreur: Aucun poids tf.Variable valide trouvé.";
                    startTrainingBtn.disabled = false;
                    return;
                }

                const {grads, value} = optimizer.computeGradients(calculateLoss, varList);

                if (value && !value.isDisposed && grads && Object.keys(grads).length > 0) { // Check if value and grads are valid and grads is not empty
                    optimizer.applyGradients(grads);
                    tf.dispose(grads);

                    const lossValue = await value.data();
                    value.dispose();
                    epochLoss += lossValue[0];
                    trainingStatusEl.textContent = `Époque ${epoch + 1}/${epochs}, Lot ${i+1}/${numBatches}, Loss: ${lossValue[0].toFixed(4)}`;
                } else {
                    console.warn("Skipping batch due to disposed tensor or missing gradients.");
                    if(value && !value.isDisposed) value.dispose(); // Clean up if value exists
                }

                await tf.nextFrame(); // Allow UI updates
            }

            epochLoss /= numBatches;
            trainingHistory.push(epochLoss);
            drawLossChart(trainingHistory, lossChartCanvas, lossChartCtx);
            console.log(`Epoch ${epoch + 1} Loss: ${epochLoss}`);
        }

        trainingStatusEl.textContent = "Entraînement terminé !";
        startTrainingBtn.disabled = false;
    });
}

// This function is called from main.js when the training tab is selected
export function refreshLossChart(lossChartCanvas, lossChartCtx) {
    drawLossChart(trainingHistory, lossChartCanvas, lossChartCtx);
}
