// Combined activity logger, classifier, and DSP using TensorFlow on Bangle.js 2
// Created on 2024-10-27 14:00 by Thomas Vikström

var activity = "";  // Variable to hold the current activity label
var recording = false;
var bufferSize = 128; // Buffer size for spectral analysis (matching FFT length)
var accelBuffer = [];  // Initialize an empty buffer for raw accelerometer data
var storageFile = null;  // File reference for storage
var inferenceInterval = null;  // Interval reference for inference timing
var currentActivityStart = null; // Start time for an activity
var activityTracking = {}; // Object to track activity repetition and timing
var model = null; // TensorFlow model reference

function spectralAnalysis(accelData, samplingFreq) {
  const bufferSize = accelData.length;
  const fftLength = 128; // As per original Python script
  const spectralPeaksCount = 3;
  const spectralPeaksThreshold = 0.1;
  const filterCutoff = 3;
  const filterOrder = 6;
  
  // 1. Filter the accelerometer data
  let filteredData = lowPassFilter(accelData, filterCutoff, samplingFreq, filterOrder);

  // 2. Offset data by mean to remove DC component
  let mean = filteredData.reduce((acc, val) => acc + val, 0) / bufferSize;
  filteredData = filteredData.map(x => x - mean);

  // 3. Calculate Root Mean Square (RMS)
  let rms = Math.sqrt(filteredData.reduce((acc, val) => acc + (val * val), 0) / bufferSize);

  // 4. Calculate skewness and kurtosis
  let skewness = calculateSkewness(filteredData, mean);
  let kurtosis = calculateKurtosis(filteredData, mean);

  // 5. Perform FFT and calculate spectral peaks
  let fftResult = performFFT(filteredData, fftLength);
  let peaks = findPeaksInFFT(fftResult, spectralPeaksThreshold, spectralPeaksCount);

  // 6. Aggregate features into an array
  let features = [rms, skewness, kurtosis, ...peaks.flat()];

  console.log("Extracted features: ", features);
  return features;
}

function lowPassFilter(data, cutoff, samplingFreq, order) {
  // Simple example of a low pass filter using a moving average
  // Replace with more sophisticated filter design if needed
  const alpha = cutoff / (samplingFreq / 2); // Normalized cutoff frequency
  let filtered = new Array(data.length).fill(0);
  filtered[0] = data[0];
  for (let i = 1; i < data.length; i++) {
    filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i - 1];
  }
  return filtered;
}

function calculateSkewness(data, mean) {
  const n = data.length;
  const s3 = data.reduce((acc, val) => acc + Math.pow(val - mean, 3), 0) / n;
  const s2 = Math.sqrt(data.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / n);
  return s3 / Math.pow(s2, 3);
}

function calculateKurtosis(data, mean) {
  const n = data.length;
  const s4 = data.reduce((acc, val) => acc + Math.pow(val - mean, 4), 0) / n;
  const s2 = Math.sqrt(data.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / n);
  return s4 / Math.pow(s2, 4);
}

function performFFT(data, fftLength) {
  let real = data.slice(0, fftLength);
  let imag = new Array(fftLength).fill(0);
  let fft = E.FFT(real, imag);
  return fft;
}

function findPeaksInFFT(fftResult, threshold, count) {
  const mag = fftResult.map(c => Math.sqrt(c.real * c.real + c.imag * c.imag));
  let peaks = [];
  for (let i = 1; i < mag.length - 1; i++) {
    if (mag[i] > threshold && mag[i] > mag[i - 1] && mag[i] > mag[i + 1]) {
      peaks.push([i, mag[i]]);
    }
  }
  peaks.sort((a, b) => b[1] - a[1]);
  return peaks.slice(0, count);
}

function loadModel() {
  try {
    let modelData = require("Storage").read("impulse4.b64");
    if (modelData) {
      modelData = atob(modelData);
      model = require("tensorflow").create(modelData.length, modelData);
      console.log("Model loaded successfully");
    } else {
      console.log("Model data not found");
    }
  } catch (e) {
    console.log("Error loading model:", e);
  }
}

function onAccel(a) {
  if (recording) {
    // Collect data for spectral analysis
    if (accelBuffer.length < bufferSize) {
      accelBuffer.push(a.x, a.y, a.z); // Add x, y, z values to the buffer
    }

    // Once buffer is full, perform spectral analysis and reset buffer
    if (accelBuffer.length >= bufferSize * 3) { // 3 values (x, y, z) per sample
      let samplingFreq = 12.5; // Sampling frequency for Bangle.js 2
      let features = spectralAnalysis(accelBuffer, samplingFreq);

      // Reset the buffer for the next cycle
      accelBuffer = [];

      // Use the extracted features for inference
      if (model) {
        try {
          model.getInput().set(features);
          model.invoke();
          let output = model.getOutput();
          let detectedActivity = getActivityLabel(output);
          
          // Log detected activity and update the status
          trackActivity(detectedActivity);
          showStatus("Detected: " + detectedActivity);
        } catch (e) {
          console.log("Error during inference with spectral features.", e);
        }
      }
    }
  }

  // Store data to file if recording
  if (storageFile) {
    var dataLine = [Date.now(), a.x, a.y, a.z, activity].join(",") + "\n";
    try {
      storageFile.write(dataLine);
    } catch (e) {
      console.log("Error writing to file:", e);
    }
  }
}

function toggleRecording() {
  if (activity === "" && !recording) {
    console.log("Select activity");
    return;
  }
  recording = !recording;
  if (recording) {
    console.log("Started recording activity:", activity);
    accelBuffer = []; // Reset buffer for new recording session
    let now = new Date();
    let filename = "acti_" +
      now.getFullYear() + "_" +
      (now.getMonth() + 1).toString().padStart(2, '0') + "_" +
      now.getDate().toString().padStart(2, '0') + "_" +
      now.getHours().toString().padStart(2, '0') + "_" +
      now.getMinutes().toString().padStart(2, '0') + ".csv";
    storageFile = require("Storage").open(filename, "a");
    if (storageFile.getLength() === 0) {
      storageFile.write("timestamp,x,y,z,activity\n");
    }
    currentActivityStart = now;
    Bangle.on('accel', onAccel);
    showStatus("Recording: " + activity);
  } else {
    console.log("Stopped recording.");
    Bangle.removeListener('accel', onAccel);
    if (storageFile) {
      let now = new Date();
      storageFile.write(currentActivityStart.toISOString() + "," + now.toISOString() + "," + activity + "\n");
      storageFile = null;
    }
    currentActivityStart = null;
    showStatus("Stopped recording");
    setTimeout(() => {
      E.showMenu(mainMenu);
    }, 1000);
  }
}

function toggleInference() {
  if (inferenceInterval) {
    console.log("Stopped inference.");
    clearInterval(inferenceInterval);
    inferenceInterval = null;
    Bangle.removeListener('accel', onAccel);
    if (storageFile) {
      let now = new Date();
      storageFile.write(currentActivityStart.toISOString() + "," + now.toISOString() + "," + activity + "\n");
      storageFile = null;
    }
    showStatus("Stopped inferring");
    setTimeout(() => {
      E.showMenu(mainMenu);
    }, 1000);
  } else {
    console.log("Started inference.");
    accelBuffer.fill(0);
    let now = new Date();
    let filename = "exercise_" +
      now.getFullYear() + "_" +
      (now.getMonth() + 1).toString().padStart(2, '0') + "_" +
      now.getDate().toString().padStart(2, '0') + "_" +
      now.getHours().toString().padStart(2, '0') + "_" +
      now.getMinutes().toString().padStart(2, '0') + ".csv";
    storageFile = require("Storage").open(filename, "a");
    if (storageFile.getLength() === 0) {
      storageFile.write("start_timestamp,end_timestamp,activity\n");
    }
    currentActivityStart = now;
    Bangle.on('accel', onAccel);
    inferenceInterval = setInterval(() => {
      performInference();
    }, 2000);
    showStatus("Inferring...");
  }
}

function performInference() {
  if (!model) {
    console.log("Model not loaded");
    return;
  }
  try {
    let features = spectralAnalysis(accelBuffer, 12.5);
    model.getInput().set(features);
    model.invoke();
    let output = model.getOutput();
    let detectedActivity = getActivityLabel(output);
    trackActivity(detectedActivity);
    showStatus("Detected: " + detectedActivity);
    console.log("Detected Activity:", detectedActivity);
  } catch (e) {
    console.log("Error during inference.", e);
  }
}

function getActivityLabel(output) {
  const labels = ["running", "sitting", "walking", "jumping", "cycling"];
  return labels[output.indexOf(Math.max(...output))] || "unknown";
}

function showStatus(message) {
  g.clear();
  g.setFont("6x8", 2);
  g.setFontAlign(0, 0);
  g.drawString(message, g.getWidth() / 2, g.getHeight() / 2);
  g.flip();
}

function selectActivity() {
  const activities = ["walking", "running", "sitting", "jumping", "cycling"];
  var menu = {
    "Walking": () => { setActivity("walking"); },
    "Running": () => { setActivity("running"); },
    "Sitting": () => { setActivity("sitting"); },
    "Jumping": () => { setActivity("jumping"); },
    "Cycling": () => { setActivity("cycling"); },
    "< Back": () => { E.showMenu(mainMenu); }
  };
  E.showMenu(menu);
}

function setActivity(newActivity) {
  activity = newActivity;
  console.log("Activity set to: " + activity);
  var menu = {
    "" : { "title" : "Activity: " + activity },
    "Start Recording": () => {
      toggleRecording();
    },
    "< Back": () => { E.showMenu(mainMenu); }
  };
  E.showMenu(menu);
}

var mainMenu = {
  "" : { "title" : "Exerciser",
       "fontHeight": 8,},
  "Collect Data": () => { selectActivity(); },
  "Inference": () => {
    toggleInference();
  },
  "Exercise": () => {
    toggleInference();
  },
  "< Back": () => { load(); }
};

E.showMenu(mainMenu);

loadModel();

setWatch(() => {
  if (recording) {
    toggleRecording();
  } else {
    toggleInference();
  }
}, BTN, {repeat: true, edge: "falling", debounce: 50});
