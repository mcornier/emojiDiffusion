// Imports will be added here for constants like IMG_SIZE, LATENT_DIM, etc.
// For now, assume they are available globally or will be passed as arguments.
// We'll need to import:
// IMG_SIZE, IMG_CHANNELS, LATENT_DIM, CONTEXT_DIM, TIME_DIM, TIMESTEPS, EMOJI_VOCAB (for EMOJI_VOCAB.length)

// --- Model Definition ---

export class MultiHeadAttention extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.latentDim = config.latentDim;
        this.numHeads = config.numHeads;
        if (this.latentDim % this.numHeads !== 0) {
            throw new Error("latentDim must be divisible by numHeads");
        }
        this.headDim = this.latentDim / this.numHeads;
    }

    build(inputShape) {
        this.toQ = this.addWeight('q_kernel', [inputShape[inputShape.length - 1], this.latentDim], 'float32', tf.initializers.glorotUniform());
        this.toK = this.addWeight('k_kernel', [inputShape[inputShape.length - 1], this.latentDim], 'float32', tf.initializers.glorotUniform());
        this.toV = this.addWeight('v_kernel', [inputShape[inputShape.length - 1], this.latentDim], 'float32', tf.initializers.glorotUniform());
        this.outputDense = this.addWeight('o_kernel', [this.latentDim, this.latentDim], 'float32', tf.initializers.glorotUniform());
        this.outputBias = this.addWeight('o_bias', [this.latentDim], 'float32', tf.initializers.zeros());
    }

    call(inputs, kwargs) {
        // IMPORTANT: NO tf.tidy() here. It prevents gradient flow.
        const [query] = Array.isArray(inputs) ? inputs : [inputs];

        let q = tf.matMul(query, this.toQ.val); // Use .val to get the tensor
        let k = tf.matMul(query, this.toK.val);
        let v = tf.matMul(query, this.toV.val);

        q = q.expandDims(1);
        k = k.expandDims(1);
        v = v.expandDims(1);

        const reshapeForMultihead = (tensor) => {
            const [batchSize, seqLen, _] = tensor.shape;
            return tensor.reshape([batchSize, seqLen, this.numHeads, this.headDim]).transpose([0, 2, 1, 3]);
        };
        q = reshapeForMultihead(q);
        k = reshapeForMultihead(k);
        v = reshapeForMultihead(v);

        const scale = tf.scalar(Math.sqrt(this.headDim));
        let scores = tf.matMul(q, k, false, true).div(scale);
        const attentionWeights = tf.softmax(scores, -1);

        let attentionOutput = tf.matMul(attentionWeights, v);

        attentionOutput = attentionOutput.transpose([0, 2, 1, 3]);
        attentionOutput = attentionOutput.reshape([-1, 1, this.latentDim]).squeeze([1]);

        return tf.add(tf.matMul(attentionOutput, this.outputDense.val), this.outputBias.val);
    }

    computeOutputShape(inputShape) {
         return [inputShape[0], this.latentDim];
    }

    static get className() {
        return 'MultiHeadAttention';
    }
}
tf.serialization.registerClass(MultiHeadAttention);


export function applyTransformerBlock(x, latentDim, numHeads, blockName) {
    const norm1 = tf.layers.layerNormalization({name: `${blockName}_norm1`}).apply(x);
    const attention = new MultiHeadAttention({latentDim, numHeads, name: `${blockName}_mha`}).apply(norm1);
    const add1 = tf.layers.add({name: `${blockName}_add1`}).apply([x, attention]);

    const norm2 = tf.layers.layerNormalization({name: `${blockName}_norm2`}).apply(add1);
    const dense1 = tf.layers.dense({ units: latentDim * 4, activation: 'gelu', name: `${blockName}_dense1` }).apply(norm2);
    const dense2 = tf.layers.dense({ units: latentDim, name: `${blockName}_dense2` }).apply(dense1);
    const add2 = tf.layers.add({name: `${blockName}_add2`}).apply([add1, dense2]);
    return add2;
}

export function buildDiffusionModel(IMG_SIZE, IMG_CHANNELS, LATENT_DIM, TIMESTEPS, TIME_DIM, EMOJI_VOCAB_LENGTH, CONTEXT_DIM) {
    try {
        const imageInput = tf.input({shape: [IMG_SIZE, IMG_SIZE, IMG_CHANNELS], name: "image_input"});
        const timeInput = tf.input({shape: [1], name: "time_input"});
        const contextInput = tf.input({shape: [1], name: "context_input"});

        let x = tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(imageInput);
        x = tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu', strides: 2 }).apply(x);
        x = tf.layers.conv2d({ filters: 128, kernelSize: 3, padding: 'same', activation: 'relu', strides: 2 }).apply(x);
        x = tf.layers.flatten().apply(x);
        const imageEmbedding = tf.layers.dense({ units: LATENT_DIM, name: 'image_embedding' }).apply(x);

        const timeEmbeddingLayer = tf.layers.embedding({inputDim: TIMESTEPS + 1, outputDim: TIME_DIM});
        const timeEmbedding = timeEmbeddingLayer.apply(timeInput);
        const timeEmbeddingFlat = tf.layers.flatten().apply(timeEmbedding);

        const contextEmbeddingLayer = tf.layers.embedding({inputDim: EMOJI_VOCAB_LENGTH, outputDim: CONTEXT_DIM});
        const contextEmbedding = contextEmbeddingLayer.apply(contextInput);
        const contextEmbeddingFlat = tf.layers.flatten().apply(contextEmbedding);

        let fused = tf.layers.concatenate().apply([imageEmbedding, timeEmbeddingFlat, contextEmbeddingFlat]);
        const transformerInput = tf.layers.dense({ units: LATENT_DIM }).apply(fused);

        let transformerOutput = transformerInput;
        transformerOutput = applyTransformerBlock(transformerOutput, LATENT_DIM, 4, 'block1'); // Assuming 4 heads, make this configurable if needed
        transformerOutput = applyTransformerBlock(transformerOutput, LATENT_DIM, 4, 'block2');
        transformerOutput = applyTransformerBlock(transformerOutput, LATENT_DIM, 4, 'block3');

        let d = tf.layers.dense({ units: 4 * 4 * 128 }).apply(transformerOutput); // Consider if 4*4*128 needs to be dynamic based on LATENT_DIM or IMG_SIZE
        d = tf.layers.reshape({ targetShape: [4, 4, 128] }).apply(d);
        d = tf.layers.conv2dTranspose({ filters: 64, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' }).apply(d);
        d = tf.layers.conv2dTranspose({ filters: 32, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' }).apply(d);
        const predictedOutput = tf.layers.conv2d({ filters: IMG_CHANNELS, kernelSize: 1, padding: 'same', name: 'predicted_output' }).apply(d);

        const model = tf.model({
            inputs: [imageInput, timeInput, contextInput],
            outputs: predictedOutput
        });

        console.log("Modèle de diffusion construit.");
        model.summary();
        return model;
    } catch(e) {
        console.error("Erreur lors de la construction du modèle:", e);
        alert("Erreur lors de la construction du modèle: " + e.message);
        return null;
    }
}
