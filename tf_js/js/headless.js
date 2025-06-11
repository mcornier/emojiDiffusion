// Assuming tf is globally available e.g. via window.tf or direct script include before modules.
// If not, tf might need to be imported if main.js exports it or from a tfjs module.

import { generateEmojiImage } from './dataset.js';
import { buildDiffusionModel } from './model.js';
// It's assumed these constants are exported from main.js
// If not, they might need to be defined locally or imported from a dedicated config file.
import { IMG_SIZE, IMG_CHANNELS, LATENT_DIM, CONTEXT_DIM, TIME_DIM, TIMESTEPS } from './main.js';

export async function runHeadlessTest() {
    console.log("Starting headless test...");
    try {
        // --- Local State for Headless Test ---
        let localEmojiVocab = ['[MASK]'];
        let localEmojiToIndex = {'[MASK]': 0};
        let headlessDataset = { tensors: [], base64: [], emojis: [] }; // base64 not strictly needed for headless
        let headlessModel = null;

        // --- 1. Dataset Generation ---
        console.log("Headless: Generating dataset...");
        const testEmojis = ['😀', '😎', '👍', '🎉', '🚀']; // A small set of emojis

        // Dispose previous tensors if any (though this is a fresh run)
        headlessDataset.tensors.forEach(t => t.dispose());
        headlessDataset = { tensors: [], base64: [], emojis: [] };
        localEmojiVocab = ['[MASK]'];
        localEmojiToIndex = {'[MASK]': 0};

        for (const emoji of testEmojis) {
            // generateEmojiImage relies on a canvas defined in dataset.js and IMG_SIZE from main.js
            const { tensor, base64 } = await generateEmojiImage(emoji, IMG_SIZE);
            headlessDataset.tensors.push(tensor);
            // headlessDataset.base64.push(base64); // Not needed for headless logic
            headlessDataset.emojis.push(emoji);

            if (!localEmojiToIndex.hasOwnProperty(emoji)) {
                localEmojiToIndex[emoji] = localEmojiVocab.length;
                localEmojiVocab.push(emoji);
            }
        }

        if (headlessDataset.tensors.length === 0) {
            console.error("Headless: Dataset generation failed, no images produced.");
            return;
        }
        console.log(`Headless: Dataset generated with ${headlessDataset.tensors.length} images.`);

        // --- 2. Model Building ---
        console.log("Headless: Building model...");
        // buildDiffusionModel needs: IMG_SIZE, IMG_CHANNELS, LATENT_DIM, TIMESTEPS, TIME_DIM, EMOJI_VOCAB.length, CONTEXT_DIM
        headlessModel = buildDiffusionModel(
            IMG_SIZE,
            IMG_CHANNELS,
            LATENT_DIM,
            TIMESTEPS, // TIMESTEPS for time embedding layer dim
            TIME_DIM,
            localEmojiVocab.length,
            CONTEXT_DIM
        );

        if (!headlessModel) {
            console.error("Headless: Model building failed.");
            return;
        }
        console.log("Headless: Model built successfully.");
        // headlessModel.summary(); // Optional: log model summary

        // --- 3. Training ---
        console.log("Headless: Starting training...");
        const epochs = 2; // Minimal epochs for a smoke test
        const learningRate = 0.0002;
        const batchSize = Math.min(4, headlessDataset.tensors.length); // Small batch size, ensure it's not larger than dataset
        const optimizer = tf.train.adam(learningRate);

        const allIndices = Array.from({length: headlessDataset.tensors.length}, (_, i) => i);

        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochLoss = 0;
            let batchesProcessed = 0;
            tf.util.shuffle(allIndices); // Shuffle data for each epoch

            for (let i = 0; i < headlessDataset.tensors.length; i += batchSize) {
                const batchIndices = allIndices.slice(i, Math.min(i + batchSize, headlessDataset.tensors.length));
                if (batchIndices.length === 0) continue;

                const currentBatchSize = batchIndices.length;

                const lossFunction = () => { // Renamed from 'f'
                    return tf.tidy(() => {
                        const batchImagesArray = batchIndices.map(idx => headlessDataset.tensors[idx]);
                        if (batchImagesArray.some(t => !t || t.isDisposed)) {
                             console.error("Headless: Disposed tensor detected in batch for training.");
                             return tf.scalar(0); // Avoid error propagation
                        }
                        const batchImages = tf.stack(batchImagesArray);

                        // Using autoencoder mode for simplicity in headless test
                        const t_zeros = tf.zeros([currentBatchSize, 1], 'int32');

                        // For autoencoder, context can be zero (MASK) or actual. Using MASK.
                        const context_zeros = tf.zeros([currentBatchSize, 1], 'int32');

                        const predictedImages = headlessModel.apply([batchImages, t_zeros, context_zeros]);
                        const loss = tf.losses.meanSquaredError(batchImages, predictedImages).mean();
                        return loss;
                    });
                };

                // Filter trainableWeights (as done in UI training)
                let varList = [];
                if (headlessModel && Array.isArray(headlessModel.trainableWeights)) {
                    varList = headlessModel.trainableWeights.filter(w => w instanceof tf.Variable);
                    if (varList.length !== headlessModel.trainableWeights.length) {
                        console.warn("Headless: Some trainable weights were filtered out.");
                    }
                } else {
                     console.error("Headless: trainableWeights is not an array or model undefined.");
                     throw new Error("Headless: Invalid trainable weights list.");
                }
                if (varList.length === 0 && headlessModel.trainableWeights.length > 0) {
                    console.error("Headless: No tf.Variable instances found in trainableWeights after filtering.");
                    throw new Error("Headless: No valid tf.Variables for training.");
                }


                const {value, grads} = optimizer.computeGradients(lossFunction, varList);

                if (value && value.isDisposed === false && grads && Object.keys(grads).length > 0) {
                    optimizer.applyGradients(grads);
                    epochLoss += await value.dataSync()[0]; // Use dataSync for headless simplicity if async not strictly needed here
                    batchesProcessed++;
                } else {
                    console.warn("Headless: Skipping batch due to missing gradients or disposed loss tensor.");
                }

                tf.dispose(grads);
                tf.dispose(value);
                await tf.nextFrame(); // Yield to prevent blocking, even in headless
            }
            if (batchesProcessed > 0) {
                console.log(`Headless: Epoch ${epoch + 1}/${epochs}, Avg Loss: ${(epochLoss / batchesProcessed).toFixed(5)}`);
            } else {
                console.log(`Headless: Epoch ${epoch + 1}/${epochs}, No batches processed.`);
            }
        }
        console.log("Headless: Training finished.");

        // --- 4. Cleanup ---
        console.log("Headless: Cleaning up resources...");
        if (headlessModel) {
            headlessModel.dispose();
            console.log("Headless: Model disposed.");
        }
        headlessDataset.tensors.forEach(t => t.dispose());
        console.log("Headless: Dataset tensors disposed.");

        console.log("------------------------------------");
        console.log("Headless test completed successfully.");
        console.log("------------------------------------");

    } catch (error) {
        console.error("------------------------------------");
        console.error("Headless test failed:", error);
        console.error("------------------------------------");
    }
}
