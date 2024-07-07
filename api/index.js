// api/index.js
const express = require('express');
const axios = require('axios');
const tf = require('@tensorflow/tfjs');
const app = express();
app.use(express.json());

// Fetch forex data
async function getForexData(symbol) {
    const response = await axios.get('https://api.twelvedata.com/time_series', {
        params: {
            symbol: symbol,
            interval: '1day',
            apikey: '262b3ecc49434af287fd0c93647418cb',
            outputsize: 500
        }
    });
    if (response.data['code'] || !response.data['values']) throw new Error('Invalid API response');
    return response.data['values'].map(entry => ({
        date: entry.datetime,
        close: parseFloat(entry.close)
    })).reverse();
}

// Normalize data
function normalizeData(data) {
    const values = data.map(d => d.close);
    const min = Math.min(...values);
    const max = Math.max(...values);
    return values.map(v => (v - min) / (max - min));
}

// Create sequences
function createSequences(data, seqLength) {
    const sequences = [];
    for (let i = 0; i < data.length - seqLength; i++) {
        sequences.push(data.slice(i, i + seqLength + 1));
    }
    return sequences;
}

// Build and train the LSTM model
async function buildAndTrainModel(data) {
    const seqLength = 20; // Increased length for more context
    const normalizedData = normalizeData(data);

    const sequences = createSequences(normalizedData, seqLength);

    const xs = tf.tensor2d(sequences.map(seq => seq.slice(0, seqLength)), [sequences.length, seqLength, 1]);
    const ys = tf.tensor2d(sequences.map(seq => [seq[seqLength]]), [sequences.length, 1]);

    // Build model
    const model = tf.sequential();
    model.add(tf.layers.lstm({ units: 100, returnSequences: true, inputShape: [seqLength, 1] }));
    model.add(tf.layers.lstm({ units: 100 }));
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });

    // Train model
    await model.fit(xs, ys, { epochs: 20, batchSize: 64 });
    return model;
}

// Endpoint to get prediction
app.post('/predict', async (req, res) => {
    try {
        const { symbol } = req.body;
        const data = await getForexData(symbol);

        // Build and train model
        const model = await buildAndTrainModel(data);

        // Predict next value
        const input = tf.tensor2d([normalizeData(data.slice(-20))], [1, 20, 1]);
        const prediction = model.predict(input).dataSync();
        res.json({ prediction: prediction[0] });
    } catch (error) {
        console.error('Error:', error);
        res.status(500).send('Error processing request');
    }
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
