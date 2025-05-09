// src/web_deployment/app.js

// DOM Elements
const imageUpload = document.getElementById('imageUpload');
const outputCanvas = document.getElementById('outputCanvas');
const canvasCtx = outputCanvas.getContext('2d');
const statusDiv = document.getElementById('status');
const benchmarkInfoDiv = document.getElementById('benchmarkInfo');
const uploadedImageElement = document.getElementById('uploadedImage'); // Hidden img element
const ortVersionSpan = document.getElementById('ortVersion');
const executionProviderSpan = document.getElementById('executionProvider');
const loaderDiv = document.getElementById('loader');
const providerSelectionDiv = document.getElementById('providerSelection');
const loadModelButton = document.getElementById('loadModelButton');

// Configuration
const modelPath = 'model.onnx'; // Ensure this model is in the same directory or provide correct path
const modelInputShape = [1, 3, 640, 640]; // NCHW format, [batch_size, channels, height, width]

// >>> IMPORTANT: Replace this with the actual class names from your data.yaml <<<
const classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

let inferenceSession;
let currentExecutionProvider = null; // Will be set by user

// --- Utility Functions ---
function updateStatus(message, type = 'loading') {
    statusDiv.textContent = message;
    statusDiv.className = 'status-message'; // Reset classes
    if (type === 'loading') {
        statusDiv.classList.add('status-loading');
        loaderDiv.classList.remove('hidden');
    } else if (type === 'ready') {
        statusDiv.classList.add('status-ready');
        loaderDiv.classList.add('hidden');
    } else if (type === 'error') {
        statusDiv.classList.add('status-error');
        loaderDiv.classList.add('hidden');
    } else { // 'info' or other
        statusDiv.classList.remove('status-loading', 'status-ready', 'status-error');
        loaderDiv.classList.add('hidden');
    }
    console.log(`Status: ${message} (Type: ${type})`);
}

function updateBenchmarkInfo(text) {
    benchmarkInfoDiv.innerHTML = text;
    console.log(`Benchmark Info: ${text.replace(/<br\s*\/?>/gi, '\n')}`);
}

// --- Core ONNX Runtime Functions ---
async function initializeORT(selectedProvider) {
    // Disable button and selection while loading
    loadModelButton.disabled = true;
    providerSelectionDiv.querySelectorAll('input[type="radio"]').forEach(radio => radio.disabled = true);
    updateStatus(`Attempting to load model with ${selectedProvider.toUpperCase()}...`, 'loading');
    console.info(`Initializing ONNX Runtime with selected provider: ${selectedProvider.toUpperCase()}`);

    try {
        if (!ort) {
             throw new Error("ONNX Runtime (ort) library not found. Ensure ort.min.js is loaded.");
        }

        // Display ORT version as soon as 'ort' is confirmed available
        ortVersionSpan.textContent = ort.env.versions.ortRelease || 'Unknown';

        inferenceSession = await ort.InferenceSession.create(modelPath, {
            executionProviders: [selectedProvider],
            graphOptimizationLevel: 'all'
            // You might need to configure wasm paths if not using CDN defaults:
            // wasmPaths: 'path/to/ort-wasm-files/'
        });

        currentExecutionProvider = selectedProvider; // Store the successfully loaded provider
        executionProviderSpan.textContent = selectedProvider.toUpperCase();
        updateStatus(`Model loaded successfully with ${selectedProvider.toUpperCase()}! Ready for inference.`, 'ready');
        console.log(`ONNX Runtime session created successfully with ${selectedProvider.toUpperCase()}.`);
        console.log("Model Input Names:", inferenceSession.inputNames);
        console.log("Model Output Names:", inferenceSession.outputNames);

        // Enable image upload now that the model is loaded
        imageUpload.disabled = false;

    } catch (e) {
        updateStatus(`Error loading model with ${selectedProvider.toUpperCase()}: ${e.message}`, 'error');
        console.error(`Error creating ONNX inference session with ${selectedProvider.toUpperCase()}:`, e);
        console.error("Ensure 'model.onnx' is accessible and valid. Also check browser console for specific WebGL/WebGPU errors if applicable.");
        // Re-enable UI elements if loading failed
        loadModelButton.disabled = false;
        providerSelectionDiv.querySelectorAll('input[type="radio"]').forEach(radio => radio.disabled = false);
        executionProviderSpan.textContent = 'N/A'; // Reset provider display
        inferenceSession = null; // Ensure session is null if failed
    }
}

async function preprocessImageForYOLO(imageElement) {
    updateStatus('Preprocessing image...', 'loading');
    console.log("Starting image preprocessing...");

    const targetHeight = modelInputShape[2];
    const targetWidth = modelInputShape[3];
    console.log(`Target model input dimensions: ${targetWidth}x${targetHeight}`);

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = targetWidth;
    tempCanvas.height = targetHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(imageElement, 0, 0, targetWidth, targetHeight);
    console.log("Image drawn to temporary canvas for resizing.");

    const imageData = tempCtx.getImageData(0, 0, targetWidth, targetHeight);
    const { data } = imageData;
    console.log(`Image data length from temp canvas: ${data.length} (Expected: ${targetWidth * targetHeight * 4})`);

    const float32Data = new Float32Array(modelInputShape[1] * modelInputShape[2] * modelInputShape[3]);
    const stride = targetHeight * targetWidth;

    for (let h = 0; h < targetHeight; h++) {
        for (let w = 0; w < targetWidth; w++) {
            const R_SRC_INDEX = (h * targetWidth + w) * 4;
            const G_SRC_INDEX = (h * targetWidth + w) * 4 + 1;
            const B_SRC_INDEX = (h * targetWidth + w) * 4 + 2;

            float32Data[h * targetWidth + w] = data[R_SRC_INDEX] / 255.0; // R
            float32Data[stride + h * targetWidth + w] = data[G_SRC_INDEX] / 255.0; // G
            float32Data[2 * stride + h * targetWidth + w] = data[B_SRC_INDEX] / 255.0; // B
        }
    }

    const inputTensor = new ort.Tensor('float32', float32Data, modelInputShape);
    console.log("Preprocessing complete. Input tensor created with shape:", inputTensor.dims);
    updateStatus('Image preprocessed.', 'info');
    return inputTensor;
}

function calculateIoU(box1, box2) {
    // box format: [x1, y1, x2, y2]
    const x1_1 = box1[0], y1_1 = box1[1], x2_1 = box1[2], y2_1 = box1[3];
    const x1_2 = box2[0], y1_2 = box2[1], x2_2 = box2[2], y2_2 = box2[3];

    const inter_x1 = Math.max(x1_1, x1_2);
    const inter_y1 = Math.max(y1_1, y1_2);
    const inter_x2 = Math.min(x2_1, x2_2);
    const inter_y2 = Math.min(y2_1, y2_2);

    const inter_width = Math.max(0, inter_x2 - inter_x1);
    const inter_height = Math.max(0, inter_y2 - inter_y1);
    const inter_area = inter_width * inter_height;

    if (inter_area === 0) return 0;

    const area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
    const area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
    const union_area = area1 + area2 - inter_area;

    return union_area > 0 ? inter_area / union_area : 0;
}

function nonMaxSuppressionForClass(proposals, iouThreshold) {
    if (!proposals || proposals.length === 0) return [];
    proposals.sort((a, b) => b.score - a.score);
    const keptProposals = [];
    const removedIndices = new Set();
    for (let i = 0; i < proposals.length; i++) {
        if (removedIndices.has(i)) continue;
        keptProposals.push(proposals[i]);
        removedIndices.add(i);
        for (let j = i + 1; j < proposals.length; j++) {
            if (removedIndices.has(j)) continue;
            const iou = calculateIoU(proposals[i].box, proposals[j].box);
            if (iou > iouThreshold) removedIndices.add(j);
        }
    }
    return keptProposals;
}

function classSpecificNMS(proposalsByClass, iouThreshold) {
    console.log(`Starting Class-Specific NMS. IoU Threshold: ${iouThreshold}`);
    const finalDetections = [];
    let totalKept = 0;
    for (const classId in proposalsByClass) {
        const classProposals = proposalsByClass[classId];
        if (classProposals && classProposals.length > 0) {
            const keptForClass = nonMaxSuppressionForClass(classProposals, iouThreshold);
            finalDetections.push(...keptForClass);
            totalKept += keptForClass.length;
        }
    }
    console.log(`Class-Specific NMS complete. Total detections kept: ${totalKept}`);
    return finalDetections;
}

function postprocessYOLOOutput(outputMap, originalImageWidth, originalImageHeight) {
    updateStatus('Postprocessing detections...', 'loading');
    console.groupCollapsed("Postprocessing Details (JS - Mirroring Python Logic)");
    console.log("Raw model output map:", outputMap);

    const outputTensorName = inferenceSession.outputNames[0];
    const outputTensor = outputMap[outputTensorName];
    console.log(`Output tensor '${outputTensorName}' received. Dimensions:`, outputTensor.dims, `Type: ${outputTensor.type}`);

    if (!outputTensor || !outputTensor.data || outputTensor.dims.length < 3) {
        console.error("Output tensor is invalid or has unexpected dimensions.");
        updateStatus('Error: Invalid model output structure.', 'error');
        console.groupEnd();
        return [];
    }

    let predictionsRawData = outputTensor.data; // Assume Float32Array
    let numPredictions, numElementsPerPrediction;
    const batchSize = outputTensor.dims[0];
    const dim1 = outputTensor.dims[1];
    const dim2 = outputTensor.dims[2];
    const numClassesAvailable = classNames.length;
    const expectedFeatures = 4 + numClassesAvailable;

    // Handle potential transpose
    if (dim1 === expectedFeatures && dim2 > expectedFeatures) {
        console.info(`Output shape ${outputTensor.dims} suggests (batch, features, proposals). Transposing in JS...`);
        numPredictions = dim2;
        numElementsPerPrediction = dim1;
        const transposedData = new Float32Array(batchSize * numPredictions * numElementsPerPrediction);
        for (let b = 0; b < batchSize; ++b) {
            for (let p = 0; p < numPredictions; ++p) {
                for (let f = 0; f < numElementsPerPrediction; ++f) {
                    transposedData[b * numPredictions * numElementsPerPrediction + p * numElementsPerPrediction + f] =
                        predictionsRawData[b * dim1 * dim2 + f * dim2 + p];
                }
            }
        }
        predictionsRawData = transposedData;
        console.log(`Shape after JS transpose: [${batchSize}, ${numPredictions}, ${numElementsPerPrediction}]`);
    } else if (dim2 === expectedFeatures) {
        console.info(`Output shape ${outputTensor.dims} suggests (batch, proposals, features). Using as is.`);
        numPredictions = dim1;
        numElementsPerPrediction = dim2;
    } else {
        console.error(`Unexpected output tensor dimensions: ${outputTensor.dims}. Cannot determine structure. Expected features: ${expectedFeatures}`);
        updateStatus('Error: Unexpected model output shape.', 'error');
        console.groupEnd();
        return [];
    }

    const numModelClasses = numElementsPerPrediction - 4;
    console.log(`Processing ${numPredictions} proposals.`);
    console.log(`Elements per proposal: ${numElementsPerPrediction} (implies ${numModelClasses} classes in model output).`);
    if (numClassesAvailable !== numModelClasses) {
         console.warn(`Potential Mismatch: Model output suggests ${numModelClasses} classes, but ${numClassesAvailable} classNames provided.`);
    }

    const proposalsByClass = {};
    const modelInputH = modelInputShape[2];
    const modelInputW = modelInputShape[3];
    const originalH = originalImageHeight;
    const originalW = originalImageWidth;
    const scaleX = originalW / modelInputW;
    const scaleY = originalH / modelInputH;
    console.log(`Scaling: ScaleX=${scaleX.toFixed(3)}, ScaleY=${scaleY.toFixed(3)}`);

    const confidenceThreshold = 0.25;
    const iouThresholdNMS = 0.45;
    let detectionsLoggedCount = 0;

    // Process predictions assuming batch size is 1
    const predictions = predictionsRawData.slice(0, numPredictions * numElementsPerPrediction); // Extract first batch if needed

    for (let i = 0; i < numPredictions; i++) {
        const offset = i * numElementsPerPrediction;

        const classConfidences = predictions.slice(offset + 4, offset + numElementsPerPrediction);

        let classId = -1;
        let maxScore = 0.0;
        for (let j = 0; j < numModelClasses; ++j) {
            if (classConfidences[j] > maxScore) {
                maxScore = classConfidences[j];
                classId = j;
            }
        }

        if (maxScore < confidenceThreshold) continue;

        const cx_model = predictions[offset + 0];
        const cy_model = predictions[offset + 1];
        const w_model = predictions[offset + 2];
        const h_model = predictions[offset + 3];

        const x1_model = cx_model - w_model / 2;
        const y1_model = cy_model - h_model / 2;
        const x2_model = cx_model + w_model / 2;
        const y2_model = cy_model + h_model / 2;

        const x1_orig = x1_model * scaleX;
        const y1_orig = y1_model * scaleY;
        const x2_orig = x2_model * scaleX;
        const y2_orig = y2_model * scaleY;

        let className = `ID ${classId}`;
        if (classId >= 0 && classId < classNames.length) {
            className = classNames[classId];
        } else {
            console.warn(`Detected classId ${classId} is out of bounds for provided classNames (length ${classNames.length}).`);
        }

        const proposalData = {
            box: [x1_orig, y1_orig, x2_orig, y2_orig],
            score: maxScore,
            classId: classId,
            className: className
        };

        if (detectionsLoggedCount < 5) {
             console.debug(`  Pre-NMS Proposal ${i}:`);
             console.debug(`    Raw Box (cx,cy,w,h): (${cx_model.toFixed(1)}, ${cy_model.toFixed(1)}, ${w_model.toFixed(1)}, ${h_model.toFixed(1)})`);
             console.debug(`    Scores: ClassID=${classId}, MaxScore=${maxScore.toFixed(3)}`);
             console.debug(`    Scaled Box (x1,y1,x2,y2): (${x1_orig.toFixed(1)}, ${y1_orig.toFixed(1)}, ${x2_orig.toFixed(1)}, ${y2_orig.toFixed(1)}) -> ${className}`);
             detectionsLoggedCount++;
        }

        if (!proposalsByClass[classId]) proposalsByClass[classId] = [];
        proposalsByClass[classId].push(proposalData);
    }

    const numProposalsBeforeNMS = Object.values(proposalsByClass).reduce((sum, arr) => sum + arr.length, 0);
    console.log(`Collected ${numProposalsBeforeNMS} proposals (score > ${confidenceThreshold}) for NMS.`);

    const finalDetections = classSpecificNMS(proposalsByClass, iouThresholdNMS);

    const processedDetections = finalDetections.map(det => {
        const [x1, y1, x2, y2] = det.box;
        const final_x1 = Math.max(0, x1);
        const final_y1 = Math.max(0, y1);
        const final_x2 = Math.min(originalW, x2);
        const final_y2 = Math.min(originalH, y2);

        if (final_x2 > final_x1 && final_y2 > final_y1) {
            return { ...det, box: [final_x1, final_y1, final_x2, final_y2] };
        }
        return null;
    }).filter(det => det !== null);

    console.log("Postprocessing complete. Final detections after NMS & clipping:", processedDetections);
    console.groupEnd();
    updateStatus('Detections postprocessed.', 'info');
    return processedDetections;
}


function drawBoundingBoxesOnCanvas(detectedObjects) {
    canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    canvasCtx.drawImage(uploadedImageElement, 0, 0, outputCanvas.width, outputCanvas.height);
    console.log(`Drawing ${detectedObjects.length} bounding boxes on canvas.`);

    detectedObjects.forEach(obj => {
        const [x1, y1, x2, y2] = obj.box;
        const score = obj.score;
        const className = obj.className;

        const boxWidth = x2 - x1;
        const boxHeight = y2 - y1;

        if (boxWidth <= 0 || boxHeight <= 0) {
            console.debug(`Skipping zero-area box for ${className}`);
            return;
        }

        canvasCtx.strokeStyle = 'red';
        canvasCtx.lineWidth = Math.max(1.5, Math.min(outputCanvas.width, outputCanvas.height) / 300);
        canvasCtx.strokeRect(x1, y1, boxWidth, boxHeight);

        canvasCtx.fillStyle = 'red';
        const fontSize = Math.max(10, Math.min(outputCanvas.width, outputCanvas.height) / 50);
        canvasCtx.font = `bold ${fontSize}px Arial`;
        const label = `${className}: ${score.toFixed(2)}`;
        const textMetrics = canvasCtx.measureText(label);
        const textWidth = textMetrics.width;
        const textHeight = fontSize;

        let rectY = y1 - textHeight - (canvasCtx.lineWidth * 1.5);
        let textY = y1 - (canvasCtx.lineWidth * 0.5) - (textHeight * 0.1);

        if (rectY < 0) {
            rectY = y1 + (canvasCtx.lineWidth * 0.5);
            textY = y1 + textHeight + (canvasCtx.lineWidth * 0.5);
        }
        const rectX = Math.max(0, x1);
        const rectWidth = Math.min(textWidth + 4, outputCanvas.width - rectX);

        canvasCtx.fillRect(rectX, rectY, rectWidth, textHeight + 4);
        canvasCtx.fillStyle = 'white';
        canvasCtx.fillText(label, rectX + 2, textY);

    });
    console.log("Bounding boxes drawing complete.");
}


// --- Event Listeners ---

// Load Model Button Listener
loadModelButton.onclick = async () => {
    const selectedProviderInput = providerSelectionDiv.querySelector('input[name="provider"]:checked');
    if (selectedProviderInput) {
        await initializeORT(selectedProviderInput.value);
    } else {
        updateStatus("Please select a backend provider first.", "error");
    }
};

// Image Upload Listener
imageUpload.onchange = async (event) => {
    const file = event.target.files[0];
    if (file && inferenceSession) {
        const reader = new FileReader();
        reader.onload = async (e) => {
            uploadedImageElement.onload = async () => {
                console.log("--- New Image Uploaded ---");
                console.log("Image loaded into hidden img tag. Dimensions:", uploadedImageElement.width, "x", uploadedImageElement.height);
                outputCanvas.width = uploadedImageElement.width;
                outputCanvas.height = uploadedImageElement.height;
                canvasCtx.drawImage(uploadedImageElement, 0, 0, uploadedImageElement.width, uploadedImageElement.height);
                updateStatus('Image loaded. Processing...', 'loading');
                benchmarkInfoDiv.innerHTML = ""; // Clear previous benchmark info

                try {
                    const inputTensor = await preprocessImageForYOLO(uploadedImageElement);
                    const modelInputName = inferenceSession.inputNames[0];
                    if (!modelInputName) throw new Error("Could not get input name from model.");

                    const feeds = { [modelInputName]: inputTensor };
                    console.log(`Feed created for input name: '${modelInputName}'`);

                    updateStatus('Running inference...', 'loading');
                    console.time("InferenceTime");
                    const startTime = performance.now();
                    const outputMap = await inferenceSession.run(feeds);
                    const endTime = performance.now();
                    console.timeEnd("InferenceTime");
                    const latency = endTime - startTime;
                    updateBenchmarkInfo(`Inference Latency: <b>${latency.toFixed(2)} ms</b> (Provider: ${currentExecutionProvider.toUpperCase()})`);

                    const detectedObjects = postprocessYOLOOutput(outputMap, uploadedImageElement.width, uploadedImageElement.height);
                    drawBoundingBoxesOnCanvas(detectedObjects);
                    updateStatus(`Detection complete! Found ${detectedObjects.length} objects.`, 'ready');

                } catch (e) {
                    updateStatus(`Error during inference pipeline: ${e.message}`, 'error');
                    console.error("Error during inference pipeline:", e);
                    if (e.stack) console.error(e.stack);
                }
            };
            uploadedImageElement.onerror = () => {
                updateStatus('Error: Could not load the selected image file.', 'error');
                console.error('Error loading image file into hidden img tag.');
            };
            uploadedImageElement.src = e.target.result;
        };
        reader.onerror = () => {
            updateStatus('Error: Could not read the selected file.', 'error');
            console.error('Error reading file with FileReader.');
        };
        reader.readAsDataURL(file);
    } else if (!inferenceSession) {
        updateStatus("Model not loaded yet. Please load the model first.", 'error');
        console.warn("Image uploaded, but inference session is not ready.");
    }
};

// --- Initial Setup ---
// Display ORT version if 'ort' is already loaded (e.g., from CDN)
if (typeof ort !== 'undefined' && ort.env && ort.env.versions) {
     ortVersionSpan.textContent = ort.env.versions.ortRelease || 'Unknown';
} else {
     console.warn("ONNX Runtime (ort) not immediately available on script load. Version will be updated after loading.");
}
// Don't initialize automatically, wait for button click
// (async () => {
//     console.log("Initializing ONNX Runtime and loading model...");
//     await initializeORT();
// })();
