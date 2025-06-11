import { buildDiffusionModel } from './model.js';
import { setupDatasetGeneration, generateEmojiImage, updateDatasetPreview } from './dataset.js';
import { setupTraining, refreshLossChart, drawLossChart } from './training.js';
import { setupInference, drawTensorToCanvas } from './inference.js';
import { runHeadlessTest } from './headless.js';

// --- Configuration & Constants ---
export const IMG_SIZE = 16;
export const IMG_CHANNELS = 3; // RGB
export const LATENT_DIM = 256; // Dimension pour l'image encodée
export const CONTEXT_DIM = 64; // Dimension pour l'embedding de l'emoji contexte
export const TIME_DIM = 64;    // Dimension pour l'embedding du pas de temps
export const TIMESTEPS = 25;   // Nombre d'étapes de diffusion

// --- Global State ---
export let diffusionModel = null;
export let dataset = { tensors: [], base64: [], emojis: [] }; // {tensors: [tf.Tensor], base64: [string], emojis: [char]}
export let EMOJI_VOCAB = ['[MASK]']; // Vocabulaire des emojis, [MASK] est à l'index 0
export let EMOJI_TO_INDEX = {'[MASK]': 0};

// --- DOM Elements ---
const tabs = {
    inference: document.getElementById('tab-inference'),
    training: document.getElementById('tab-training'),
    dataset: document.getElementById('tab-dataset'),
    weights: document.getElementById('tab-weights'),
};
const tabButtons = document.querySelectorAll('.tab-button');
const mainCanvas = document.getElementById('main-canvas');
const mainCtx = mainCanvas.getContext('2d');
const noiseCanvas = document.getElementById('noise-canvas');
// noiseCtx is used locally in inference.js, so not needed globally here

// Inference DOM Elements
const emojiContextInput = document.getElementById('emoji-context-1');
const startInferenceBtn = document.getElementById('start-inference-btn');
const denoiseStepBtn = document.getElementById('denoise-step-btn');
const inferenceStatusEl = document.getElementById('inference-status');
const inferenceTimestepEl = document.getElementById('inference-timestep');

// Training DOM Elements
const trainingModeSelect = document.getElementById('training-mode');
const epochsInput = document.getElementById('epochs');
const learningRateInput = document.getElementById('learning-rate');
const batchSizeInput = document.getElementById('batch-size');
const startTrainingBtn = document.getElementById('start-training-button');
const trainingStatusEl = document.getElementById('training-status');
const lossChartCanvas = document.getElementById('loss-chart');
const lossChartCtx = lossChartCanvas.getContext('2d');

// Dataset DOM Elements
const emojiListInput = document.getElementById('emoji-list');
const generateDatasetBtn = document.getElementById('generate-dataset-button');
const datasetGenStatusEl = document.getElementById('dataset-generation-status');
const datasetPreviewContainer = document.getElementById('dataset-preview-container');
const totalDatasetImagesEl = document.getElementById('total-dataset-images');

// Weights DOM Elements
const exportWeightsBtn = document.getElementById('export-weights-button');
const importWeightsFile = document.getElementById('import-weights-file');
const weightsStatusEl = document.getElementById('weights-status');

// --- Diffusion Schedule ---
export function linearBetaSchedule(timesteps) {
    const betaStart = 0.0001;
    const betaEnd = 0.02;
    return tf.linspace(betaStart, betaEnd, timesteps);
}

export const betas = linearBetaSchedule(TIMESTEPS);
export const alphas = tf.sub(1.0, betas);
export const alphasCumprod = tf.cumprod(alphas);

export function forwardNoise(x0, t) { // x0 is a batch of images
    return tf.tidy(() => {
        const noise = tf.randomNormal(x0.shape);
        // Ensure t is correctly shaped for gather, might need reshape if t is scalar for multiple images in batch
        const tCorrected = t.rank === 1 ? t.reshape([-1]) : t; // Ensure t is 1D for gather
        const alphaBarT = alphasCumprod.gather(tCorrected).reshape([-1, 1, 1, 1]); // Reshape for broadcasting

        const sqrtAlphaBarT = tf.sqrt(alphaBarT);
        const sqrtOneMinusAlphaBarT = tf.sqrt(tf.sub(1.0, alphaBarT));

        const noisedImage = tf.add(
            tf.mul(x0, sqrtAlphaBarT),
            tf.mul(noise, sqrtOneMinusAlphaBarT)
        );
        return { noisedImage, noise };
    });
}

// --- Context for Modules ---
// Provides access to shared state and functions, helps avoid circular dependencies or overly complex prop drilling.
function getModuleContext() {
    return {
        // State
        diffusionModel,
        dataset,
        EMOJI_VOCAB,
        EMOJI_TO_INDEX,
        // Constants (already exported, but can be passed for explicitness)
        IMG_SIZE, IMG_CHANNELS, LATENT_DIM, CONTEXT_DIM, TIME_DIM, TIMESTEPS,
        alphas, alphasCumprod, betas,
        // Functions
        forwardNoiseFunc: forwardNoise, // Pass the actual forwardNoise function
        buildDiffusionModelFromOwnModule: buildDiffusionModelWrapper, // Wrapper to pass constants correctly
        // DOM Elements (if a module absolutely needs one not passed directly)
        // Setter functions for state managed in main.js
        setDiffusionModel: (model) => { diffusionModel = model; },
        setDataset: (newDataset) => { dataset = newDataset; },
        setEmojiVocab: (vocab) => { EMOJI_VOCAB = vocab; },
        setEmojiToIndex: (map) => { EMOJI_TO_INDEX = map; },
        // For dataset.js to access constants needed by buildDiffusionModel
        constants: { IMG_CHANNELS, LATENT_DIM, TIMESTEPS, TIME_DIM, CONTEXT_DIM }
    };
}

// Wrapper for buildDiffusionModel to ensure it gets all necessary constants from main.js scope
function buildDiffusionModelWrapper(currentImgSize, currentImgChannels, currentLatentDim, currentTimesteps, currentTimeDim, currentEmojiVocabLength, currentContextDim) {
    return buildDiffusionModel(currentImgSize, currentImgChannels, currentLatentDim, currentTimesteps, currentTimeDim, currentEmojiVocabLength, currentContextDim);
}


// --- Weight Management ---
exportWeightsBtn.addEventListener('click', async () => {
    if (!diffusionModel) {
        weightsStatusEl.textContent = "Aucun modèle à exporter.";
        return;
    }
    try {
        const modelArtifacts = await diffusionModel.save(tf.io.withSaveHandler(async (modelArtifacts) => {
            return modelArtifacts;
        }));
        const modelJson = {
            modelTopology: modelArtifacts.modelTopology,
            weightSpecs: modelArtifacts.weightSpecs,
            weightData: btoa(String.fromCharCode.apply(null, new Uint8Array(modelArtifacts.weightData))), // binary to base64
            EMOJI_VOCAB: EMOJI_VOCAB,
            EMOJI_TO_INDEX: EMOJI_TO_INDEX,
            IMG_SIZE: IMG_SIZE, // Save relevant constants
            IMG_CHANNELS: IMG_CHANNELS,
            LATENT_DIM: LATENT_DIM,
            CONTEXT_DIM: CONTEXT_DIM,
            TIME_DIM: TIME_DIM,
            TIMESTEPS: TIMESTEPS
        };

        const jsonStr = JSON.stringify(modelJson);
        const blob = new Blob([jsonStr], {type: "application/json"});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = "diffusion_model_weights.json";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        weightsStatusEl.textContent = "Poids exportés avec succès !";
    } catch (error) {
        console.error("Erreur d'exportation des poids:", error);
        weightsStatusEl.textContent = "Erreur d'exportation: " + error.message;
    }
});

importWeightsFile.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) {
        weightsStatusEl.textContent = "Aucun fichier sélectionné.";
        return;
    }
    weightsStatusEl.textContent = "Importation des poids...";
    try {
        const fileContent = await file.text();
        const modelJson = JSON.parse(fileContent);

        // Check for constants mismatch here if needed (e.g., modelJson.IMG_SIZE !== IMG_SIZE)
        // For now, we assume imported model matches current config or we update config based on it.
        // Update EMOJI_VOCAB and EMOJI_TO_INDEX from the loaded file
        EMOJI_VOCAB = modelJson.EMOJI_VOCAB || ['[MASK]'];
        EMOJI_TO_INDEX = modelJson.EMOJI_TO_INDEX || {'[MASK]': 0};
        // Potentially update other constants like IMG_SIZE, LATENT_DIM etc. if they were saved
        // This part needs careful consideration: do loaded weights dictate config, or does current config try to load weights?
        // For now, let's assume the model architecture defined by current constants is compatible.

        const weightData = new Uint8Array(atob(modelJson.weightData).split("").map(char => char.charCodeAt(0))).buffer;

        if (diffusionModel) diffusionModel.dispose(); // Dispose existing model

        // Rebuild model with potentially new vocab size from imported file
        diffusionModel = buildDiffusionModel(
            modelJson.IMG_SIZE || IMG_SIZE,
            modelJson.IMG_CHANNELS || IMG_CHANNELS,
            modelJson.LATENT_DIM || LATENT_DIM,
            modelJson.TIMESTEPS || TIMESTEPS,
            modelJson.TIME_DIM || TIME_DIM,
            EMOJI_VOCAB.length, // Use the length of the new vocab
            modelJson.CONTEXT_DIM || CONTEXT_DIM
        );

        await diffusionModel.loadWeights({
            modelTopology: modelJson.modelTopology,
            weightSpecs: modelJson.weightSpecs,
            weightData: weightData,
        });

        weightsStatusEl.textContent = "Poids importés et chargés avec succès !";
        console.log("Modèle chargé à partir du JSON.");
        diffusionModel.summary();

        // Potentially update dataset preview if emojis changed, or prompt user.
        // For now, just log. If a dataset exists, it might be based on old emojis.
        if (dataset.tensors.length > 0) {
            console.warn("Les poids ont été importés. Le dataset existant pourrait ne pas correspondre au nouveau vocabulaire d'emojis.");
            // Consider clearing dataset or re-generating if EMOJI_VOCAB has significantly changed
            // updateDatasetPreview(dataset, datasetPreviewContainer, totalDatasetImagesEl); // Refresh with current dataset
        }

    } catch (error) {
        console.error("Erreur d'importation des poids:", error);
        weightsStatusEl.textContent = "Erreur d'importation: " + error.message;
        if (diffusionModel) diffusionModel.dispose();
        diffusionModel = null; // Ensure model is null on error
    }
});


// --- App Initialization ---
function initializeApp() {
    // Setup Tabs
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.dataset.tab;
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            Object.values(tabs).forEach(content => content.classList.remove('active'));
            tabs[targetTab].classList.add('active');
            if (targetTab === 'training') {
                // trainingHistory is local to training.js, so we call a refresh function
                refreshLossChart(lossChartCanvas, lossChartCtx);
            }
        });
    });

    // Initialize main canvas
    mainCanvas.width = 256; mainCanvas.height = 256; // Explicitly set canvas size
    mainCtx.fillStyle = '#d1d5db';
    mainCtx.fillRect(0, 0, mainCanvas.width, mainCanvas.height);
    mainCtx.fillStyle = 'black';
    mainCtx.textAlign = 'center';
    mainCtx.font = '16px sans-serif';
    mainCtx.fillText("En attente de génération...", mainCanvas.width / 2, mainCanvas.height / 2);

    // Initialize noise canvas
    noiseCanvas.width = 256; noiseCanvas.height = 256;
    const noiseCtxInstance = noiseCanvas.getContext('2d');
    noiseCtxInstance.fillStyle = '#e5e7eb'; // A light gray, similar to original
    noiseCtxInstance.fillRect(0, 0, noiseCanvas.width, noiseCanvas.height);


    // Initialize modules by passing DOM elements and context
    setupDatasetGeneration(
        generateDatasetBtn,
        emojiListInput,
        datasetGenStatusEl,
        datasetPreviewContainer,
        totalDatasetImagesEl,
        IMG_SIZE, // Pass the constant
        getModuleContext // Pass the context getter
    );

    setupTraining(
        startTrainingBtn,
        trainingStatusEl,
        trainingModeSelect,
        epochsInput,
        learningRateInput,
        batchSizeInput,
        lossChartCanvas,
        lossChartCtx,
        getModuleContext
    );

    setupInference(
        mainCanvas,
        noiseCanvas,
        emojiContextInput,
        startInferenceBtn,
        denoiseStepBtn,
        inferenceStatusEl,
        inferenceTimestepEl,
        getModuleContext
    );

    // Initial drawing of the loss chart (empty)
    drawLossChart([], lossChartCanvas, lossChartCtx);


    console.log("Application initialisée. Veuillez générer un dataset ou importer des poids.");
    weightsStatusEl.textContent = "Prêt. Importez des poids ou générez un dataset.";
}

// Run initialization when the DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initializeApp();
        // Check for headless mode after app initialization (or instead of part of it)
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.get('headless') === 'true') {
            console.log("Headless mode requested via URL parameter.");
            // Optionally hide UI elements if test is exclusive
            // document.body.style.display = 'none'; // Example: hide everything
            runHeadlessTest().then(() => {
                console.log("Headless test sequence finished.");
            }).catch(e => {
                console.error("Error during headless test execution:", e);
            });
        }
    });
} else {
    initializeApp();
    // Also check if running immediately (DOM already loaded)
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('headless') === 'true') {
        console.log("Headless mode requested via URL parameter (DOM already loaded).");
        runHeadlessTest().then(() => {
            console.log("Headless test sequence finished.");
        }).catch(e => {
            console.error("Error during headless test execution (DOM already loaded):", e);
        });
    }
}
