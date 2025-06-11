// Imports from main.js:
// IMG_SIZE, datasetPreviewContainer, totalDatasetImagesEl, emojiListInput, datasetGenStatusEl
// Access to global state from main.js:
// dataset, EMOJI_VOCAB, EMOJI_TO_INDEX, diffusionModel (getter/setter or direct access)

// Imports from model.js:
// buildDiffusionModel

// --- Dataset Functions ---

// These are module-local as they are implementation details for generateEmojiImage
const datasetOffscreenCanvas = document.createElement('canvas');
// IMG_SIZE will be imported
// datasetOffscreenCanvas.width = IMG_SIZE;
// datasetOffscreenCanvas.height = IMG_SIZE;
const datasetCtx = datasetOffscreenCanvas.getContext('2d', { willReadFrequently: true });

export async function generateEmojiImage(emoji, currentImgSize) {
    // Ensure canvas is sized correctly (in case IMG_SIZE is not yet available at module load)
    if (datasetOffscreenCanvas.width !== currentImgSize) {
        datasetOffscreenCanvas.width = currentImgSize;
        datasetOffscreenCanvas.height = currentImgSize;
    }

    datasetCtx.fillStyle = 'white';
    datasetCtx.fillRect(0, 0, currentImgSize, currentImgSize);
    datasetCtx.textAlign = 'center';
    datasetCtx.textBaseline = 'middle';
    datasetCtx.font = `${currentImgSize-2}px sans-serif`; // Ensure font size scales with image size
    datasetCtx.fillText(emoji, currentImgSize / 2, currentImgSize / 2 + 1); // Small adjustment for vertical centering

    const imageData = datasetCtx.getImageData(0, 0, currentImgSize, currentImgSize);
    const tensor = await tf.browser.fromPixels(imageData);
    // Normalize to [-1, 1]
    const normalizedTensor = tensor.toFloat().div(127.5).sub(1);
    const base64 = datasetOffscreenCanvas.toDataURL();
    tensor.dispose(); // Dispose the intermediate tensor
    return { tensor: normalizedTensor, base64 };
}

export function updateDatasetPreview(currentDataset, previewContainer, totalImagesEl) {
    previewContainer.innerHTML = ''; // Clear previous previews
    if (currentDataset.base64.length === 0) {
        previewContainer.innerHTML = '<p class="text-xs text-gray-500 italic col-span-full">Aucun dataset chargé ou généré.</p>';
    } else {
        currentDataset.base64.forEach(b64 => {
            const img = document.createElement('img');
            img.src = b64;
            img.className = 'dataset-image-preview'; // Ensure this class is defined in your CSS
            previewContainer.appendChild(img);
        });
    }
    totalImagesEl.textContent = currentDataset.tensors.length;
}

// This function will be called from main.js to attach the event listener
export function setupDatasetGeneration(
    generateDatasetBtn,
    emojiListInput,
    datasetGenStatusEl,
    datasetPreviewContainer,
    totalDatasetImagesEl,
    IMG_SIZE_CONST, // Constant
    getContext // Function to get current context { dataset, EMOJI_VOCAB, EMOJI_TO_INDEX, diffusionModel }
) {
    const { buildDiffusionModelFromOwnModule } = getContext(); // Function to access buildDiffusionModel from model.js via main.js

    generateDatasetBtn.addEventListener('click', async () => {
        let { dataset, EMOJI_VOCAB, EMOJI_TO_INDEX, diffusionModel, setDiffusionModel, setDataset, setEmojiVocab, setEmojiToIndex } = getContext();

        const emojiString = emojiListInput.value;
        // Filter out empty strings and duplicates more robustly
        const uniqueEmojis = [...new Set(Array.from(emojiString))].filter(e => e.trim() !== '' && /\S/.test(e));

        if (uniqueEmojis.length === 0) {
            datasetGenStatusEl.textContent = "Aucun emoji valide à traiter.";
            return;
        }
        datasetGenStatusEl.textContent = `Génération de ${uniqueEmojis.length} images...`;

        // Dispose old tensors
        if (dataset.tensors) {
            dataset.tensors.forEach(t => t.dispose());
        }
        let newDataset = { tensors: [], base64: [], emojis: [] };
        let newEmojiVocab = ['[MASK]'];
        let newEmojiToIndex = {'[MASK]': 0};

        for (const emoji of uniqueEmojis) {
            try {
                const { tensor, base64 } = await generateEmojiImage(emoji, IMG_SIZE_CONST);
                newDataset.tensors.push(tensor);
                newDataset.base64.push(base64);
                newDataset.emojis.push(emoji);

                if (!newEmojiToIndex.hasOwnProperty(emoji)) {
                    newEmojiToIndex[emoji] = newEmojiVocab.length;
                    newEmojiVocab.push(emoji);
                }
            } catch (error) {
                console.error(`Failed to generate image for emoji: ${emoji}`, error);
                datasetGenStatusEl.textContent = `Erreur lors de la génération de l'emoji: ${emoji}.`;
                // Optionally, decide if you want to stop or continue
            }
        }

        setDataset(newDataset);
        setEmojiVocab(newEmojiVocab);
        setEmojiToIndex(newEmojiToIndex);

        updateDatasetPreview(newDataset, datasetPreviewContainer, totalDatasetImagesEl);
        datasetGenStatusEl.textContent = `Dataset généré : ${newDataset.tensors.length} images.`;

        // Re-build the model because EMOJI_VOCAB.length might have changed
        if (diffusionModel) {
            diffusionModel.dispose();
        }
        // Call buildDiffusionModel (which should be imported into main.js and passed here)
        // These constants will be passed from main.js
        const { IMG_CHANNELS, LATENT_DIM, TIMESTEPS, TIME_DIM, CONTEXT_DIM } = getContext().constants;
        const newModel = buildDiffusionModelFromOwnModule(IMG_SIZE_CONST, IMG_CHANNELS, LATENT_DIM, TIMESTEPS, TIME_DIM, newEmojiVocab.length, CONTEXT_DIM);
        setDiffusionModel(newModel);
    });
}
