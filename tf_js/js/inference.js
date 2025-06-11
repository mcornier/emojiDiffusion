// Imports from main.js:
// DOM Elements: mainCanvas, noiseCanvas, emojiContextInput, startInferenceBtn, denoiseStepBtn, inferenceStatusEl, inferenceTimestepEl
// Global state/data: diffusionModel, EMOJI_TO_INDEX
// Constants: TIMESTEPS, IMG_SIZE, IMG_CHANNELS, alphas, alphasCumprod, betas (diffusion schedule parameters)

// Module-local state
let inferenceState = { x: null, t: 0, contextIdx: 0 }; // x: current noisy tensor, t: current timestep

// --- Inference Functions ---

export async function drawTensorToCanvas(tensor, canvas) {
    if (!tensor || tensor.isDisposed) {
        console.warn("drawTensorToCanvas: Tensor is disposed or null.");
        // Optionally clear the canvas or draw a placeholder
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#d1d5db'; // A placeholder color
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'black';
        ctx.textAlign = 'center';
        ctx.fillText("Tensor not available", canvas.width / 2, canvas.height / 2);
        return;
    }
    const tensorToDraw = tf.tidy(() => {
        // Ensure tensor is 3D (HWC) or 4D (BHWC, take first B)
        const squeezed = tensor.shape.length > 3 ? tensor.slice([0,0,0,0], [1, ...tensor.shape.slice(1,4)]).squeeze([0]) : tensor.squeeze();
        return squeezed.add(1).mul(127.5).clipByValue(0, 255).asType('int32');
    });
    await tf.browser.toPixels(tensorToDraw, canvas);
    tensorToDraw.dispose();
}

export function setupInference(
    // DOM Elements
    mainCanvas,
    noiseCanvas,
    emojiContextInput,
    startInferenceBtn,
    denoiseStepBtn,
    inferenceStatusEl,
    inferenceTimestepEl,
    // Callbacks to access shared state and functions from main.js
    getContext
) {
    const noiseCtx = noiseCanvas.getContext('2d'); // Get context for clearing

    startInferenceBtn.addEventListener('click', async () => {
        const {
            diffusionModel,
            EMOJI_TO_INDEX,
            TIMESTEPS,
            IMG_SIZE,  // Assuming these are uppercase constants from getContext().constants
            IMG_CHANNELS
        } = getContext();

        if (!diffusionModel) {
            alert("Le modèle n'est pas prêt. Veuillez générer un dataset et l'entraîner, ou charger des poids.");
            return;
        }
        const contextEmoji = emojiContextInput.value;
        let contextIdx = 0; // Default to [MASK]
        if(contextEmoji && EMOJI_TO_INDEX.hasOwnProperty(contextEmoji)) {
            contextIdx = EMOJI_TO_INDEX[contextEmoji];
        } else if (contextEmoji) {
            alert("Emoji non trouvé dans le vocabulaire. Utilisez un emoji du dataset ou laissez vide.");
            return;
        }

        inferenceState.t = TIMESTEPS - 1;
        inferenceState.contextIdx = contextIdx;

        if (inferenceState.x) inferenceState.x.dispose();
        // Ensure IMG_SIZE and IMG_CHANNELS are numbers
        const currentImgSize = Number(IMG_SIZE);
        const currentImgChannels = Number(IMG_CHANNELS);
        inferenceState.x = tf.randomNormal([1, currentImgSize, currentImgSize, currentImgChannels]);

        await drawTensorToCanvas(inferenceState.x, mainCanvas);
        noiseCtx.clearRect(0, 0, noiseCanvas.width, noiseCanvas.height); // Clear previous noise

        inferenceStatusEl.textContent = "Prêt à débruiter";
        inferenceTimestepEl.textContent = inferenceState.t.toString();
        denoiseStepBtn.disabled = false;
        startInferenceBtn.textContent = "Réinitialiser l'Inférence"; // Change button text
    });

    denoiseStepBtn.addEventListener('click', async () => {
        const {
            diffusionModel,
            alphas,         // Diffusion schedule parameter
            alphasCumprod,  // Diffusion schedule parameter
            betas           // Diffusion schedule parameter
        } = getContext();

        if (!diffusionModel || !inferenceState.x || inferenceState.x.isDisposed || inferenceState.t < 0) {
            inferenceStatusEl.textContent = "État invalide pour le débruitage.";
            denoiseStepBtn.disabled = true;
            return;
        }

        denoiseStepBtn.disabled = true;
        inferenceStatusEl.textContent = `Débruitage de l'étape ${inferenceState.t}...`;

        const predictedNoise = tf.tidy(() => {
            const tTensor = tf.tensor2d([[inferenceState.t]], [1, 1], 'int32');
            const contextTensor = tf.tensor2d([[inferenceState.contextIdx]], [1, 1], 'int32');
            // Ensure inferenceState.x is 4D [1, H, W, C] for the model
            const xForModel = inferenceState.x.shape.length === 3 ? inferenceState.x.expandDims(0) : inferenceState.x;
            return diffusionModel.apply([xForModel, tTensor, contextTensor]);
        });

        await drawTensorToCanvas(predictedNoise, noiseCanvas);

        const newX = tf.tidy(() => {
            const xCurrent = inferenceState.x.shape.length === 3 ? inferenceState.x.expandDims(0) : inferenceState.x;

            const alphaT = alphas.gather(inferenceState.t);
            const alphaBarT = alphasCumprod.gather(inferenceState.t);
            const oneOverSqrtAlphaT = tf.div(1.0, tf.sqrt(alphaT));
            const betaT = betas.gather(inferenceState.t);

            const term1Numerator = betaT;
            const term1Denominator = tf.sqrt(tf.sub(1.0, alphaBarT));
            const term1 = tf.mul(predictedNoise, tf.div(term1Numerator, term1Denominator));

            const term2 = tf.mul(oneOverSqrtAlphaT, tf.sub(xCurrent, term1));

            let finalX;
            if (inferenceState.t > 0) {
                const z = tf.randomNormal(xCurrent.shape);
                const alphaBarTPrev = alphasCumprod.gather(inferenceState.t - 1);
                const posteriorVariance = tf.mul(
                    tf.div(tf.sub(1.0, alphaBarTPrev), tf.sub(1.0, alphaBarT)),
                    betaT
                );
                // Clamp posteriorVariance to avoid NaNs from sqrt of negative very small numbers
                const clampedVariance = tf.maximum(posteriorVariance, 1e-6);
                finalX = tf.add(term2, tf.mul(tf.sqrt(clampedVariance), z));
            } else {
                finalX = term2; // This is x_0 at t=0
            }
            return finalX.squeeze([0]); // Squeeze back to 3D if it was expanded
        });

        predictedNoise.dispose();
        if (inferenceState.x) inferenceState.x.dispose();
        inferenceState.x = tf.keep(newX); // Keep the new tensor

        await drawTensorToCanvas(inferenceState.x, mainCanvas);
        inferenceState.t--;

        if (inferenceState.t < 0) {
            inferenceStatusEl.textContent = "Terminé !";
            inferenceTimestepEl.textContent = "0";
            denoiseStepBtn.disabled = true;
            startInferenceBtn.textContent = "Démarrer / Réinitialiser l'Inférence";
        } else {
            inferenceStatusEl.textContent = "Prêt pour la prochaine étape";
            inferenceTimestepEl.textContent = inferenceState.t.toString();
            denoiseStepBtn.disabled = false;
        }
    });
}
