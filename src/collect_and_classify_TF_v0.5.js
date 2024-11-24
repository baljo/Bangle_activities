// Activity classification using TensorFlow on Bangle.js 2
// Created on 2024-10-17 12:30:00 by Thomas Vikström

var m = require("Storage").read("base64.txt");
var model = atob(m);
var tf = require("tensorflow").create(9999, E.toString(model)); 


var activity = "";  // Variable to hold the current activity label
var recording = false;
var bufferSize = 75;  // Adjustable buffer size (x, y, z for 25 samples)
var accelBuffer = new Float32Array(bufferSize).fill(0);  // Initialize fixed-size buffer with zeros
var model;  // Placeholder for the TensorFlow Lite model

// Load the TensorFlow Lite model (model data must be provided by the user)
function loadModel() {
  try {
    model = require("tensorflow").create(model.length, model);
    console.log("Model loaded successfully");
  } catch (e) {
    console.log("Error loading model:", e);
  }
}

// Function to start/stop recording using the physical button (short press)
function toggleRecording() {
  recording = !recording;
  if (recording) {
    console.log("Started recording accelerometer data for classification.");
    accelBuffer.fill(0);  // Reset the buffer
    Bangle.on('accel', onAccel);
    showStatus("Recording...");
  } else {
    console.log("Stopped recording.");
    Bangle.removeListener('accel', onAccel);
    showStatus("Stopped recording");
  }
}

// Function to handle accelerometer data and perform continuous classification
function onAccel(a) {
  // Manually shift data in the buffer to make room for new values (remove oldest values)
  for (let i = 0; i < bufferSize - 3; i++) {
    accelBuffer[i] = accelBuffer[i + 3];
  }

  // Add new accelerometer values to the end of the buffer
  accelBuffer[bufferSize - 3] = a.x;
  accelBuffer[bufferSize - 2] = a.y;
  accelBuffer[bufferSize - 1] = a.z;

  // Perform inference on updated buffer
  performInference();
}

// Perform real-time activity classification based on the current buffer
function performInference() {
  if (!model) {
    return;
  }

  try {
    // Set the input data into the model
    model.getInput().set(accelBuffer);
    
    // Invoke the model to perform inference
    model.invoke();
    
    // Get the output from the model
    let output = model.getOutput();

    //console.log(output);
    // Find the activity index manually to avoid using Math.max(...)
    let maxIndex = 0;
    for (let i = 1; i < output.length; i++) {
      if (output[i] > output[maxIndex]) {
        maxIndex = i;
      }
    }
    
    let detectedActivity = getActivityLabel(maxIndex);

    // Show the detected activity on the watch screen
    showStatus("Detected: " + detectedActivity);
    console.log("Detected Activity:", detectedActivity);
    
  } catch (e) {
    console.log("Error during inference.");
  }
}

// Utility to map activity index to label (depends on the trained model)
function getActivityLabel(index) {
  const labels = ["idling", "running", "sitting", "walking"];
  return labels[index] || "unknown";
}

// Function to show status on the watch screen
function showStatus(message) {
  g.clear();
  g.setFont("6x8", 2);
  g.setFontAlign(0, 0); // Center the text
  g.drawString(message, g.getWidth() / 2, g.getHeight() / 2);
  g.flip();
}

// Main menu for the app
var mainMenu = {
  "" : { "title" : "Activity Classifier" },
  "Start/Stop Recording": toggleRecording,
  "Exit": () => { load(); }
};

// Set up the app to show the main menu
E.showMenu(mainMenu);

// Load the TensorFlow Lite model on startup
loadModel();

// Use the single button to start or stop recording (short press only)
setWatch(toggleRecording, BTN, {repeat: true, edge: "falling", debounce: 50});

// Display initial status message
showStatus("Ready to Detect Activity");
