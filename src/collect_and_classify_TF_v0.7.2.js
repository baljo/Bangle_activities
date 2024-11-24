// Combined activity logger and classifier using TensorFlow on Bangle.js 2
// Created on 2024-10-25 20:32 by Thomas Vikström

var activity = "";  // Variable to hold the current activity label
var recording = false;
var bufferSize = 75;  // Adjustable buffer size (x, y, z for 25 samples)
var accelBuffer = new Float32Array(bufferSize).fill(0);  // Initialize fixed-size buffer with zeros
var storageFile = null;  // File reference for storage
var inferenceInterval = null;  // Interval reference for inference timing


function convertTFLiteToBase64(filename) {
  try {
    // Read the binary data from the file stored in flash
    let file = require("Storage").read(filename);
    if (!file) {
      console.log("Error: File not found");
      return;
    }

    // Convert the binary data to base64
    let base64Data = btoa(file);

    // Log or store the base64 data as needed
    console.log("Base64 Conversion Successful:");
    // console.log(base64Data);
    // Optionally, write it to a file in Storage
    require("Storage").write(filename + ".b64", base64Data);

  } catch (e) {
    console.log("Error during conversion:", e);
  }
}

// Example usage:
convertTFLiteToBase64("impulse4");

var m = require("Storage").read("impulse4.b64");
var model = atob(m);
var tf = require("tensorflow").create(15000, E.toString(model));


// Load the TensorFlow Lite model (assumed to be provided by user)
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
  if (activity === "" && !recording) {
    console.log("Select activity");
    return;
  }

  recording = !recording;
  if (recording) {
    console.log("Started recording activity:", activity);
    accelBuffer.fill(0);  // Reset the buffer

    // Create a filename based on the current date and time
    let now = new Date();
    let filename = "acti_" +
      now.getFullYear() + "_" +
      (now.getMonth() + 1).toString().padStart(2, '0') + "_" +
      now.getDate().toString().padStart(2, '0') + "_" +
      now.getHours().toString().padStart(2, '0') + "_" +
      now.getMinutes().toString().padStart(2, '0') + ".csv";

    storageFile = require("Storage").open(filename, "a");

    // Write header if the file is new (file size == 0)
    if (storageFile.getLength() === 0) {
      storageFile.write("timestamp,x,y,z,activity\n");
    }
    Bangle.on('accel', onAccel);
    showStatus("Recording: " + activity);
  } else {
    console.log("Stopped recording.");
    Bangle.removeListener('accel', onAccel);
    if (storageFile) {
      storageFile = null;  // Close the file reference
    }
    showStatus("Stopped recording");
    setTimeout(() => {
      E.showMenu(mainMenu);  // Show main menu after stopping recording
    }, 1000); // Delay for 1 second to show stopped recording message
  }
}


// Function to start/stop inference using the physical button (short press)
function toggleInference() {
  if (inferenceInterval) {
    console.log("Stopped inference.");
    clearInterval(inferenceInterval);
    inferenceInterval = null;
    Bangle.removeListener('accel', onAccel);
    showStatus("Stopped inferring");
    setTimeout(() => {
      E.showMenu(mainMenu);  // Show main menu after stopping inference
    }, 1000); // Delay for 1 second to show stopped inference message
  } else {
    console.log("Started inference.");
    accelBuffer.fill(0);  // Reset the buffer
    Bangle.on('accel', onAccel);
    inferenceInterval = setInterval(performInference, 2000); // Run inference every 2 seconds
    showStatus("Inferring...");
  }
}

// Function to handle accelerometer data and perform classification/recording
function onAccel(a) {
  // Manually shift data in the buffer to make room for new values (remove oldest values)
  for (let i = 0; i < bufferSize - 3; i++) {
    accelBuffer[i] = accelBuffer[i + 3];
  }

  // Add new accelerometer values to the end of the buffer
  accelBuffer[bufferSize - 3] = a.x;
  accelBuffer[bufferSize - 2] = a.y;
  accelBuffer[bufferSize - 1] = a.z;

  // Save accelerometer data if recording
  if (recording && storageFile) {
    var dataLine = [Date.now(), a.x, a.y, a.z, activity].join(",") + "\n";
    try {
      storageFile.write(dataLine);
      console.log("Data written: ", dataLine);
    } catch (e) {
      console.log("Error writing to file:", e);
    }
  }
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
    console.log(output);
    
    // Find the activity index manually to avoid using Math.max(...)
    let maxIndex = 0;
    for (let i = 1; i < output.length; i++) {
      if (output[i] > output[maxIndex]) {
        maxIndex = i;
      }
    }
    
    let detectedActivity = getActivityLabel(maxIndex);

    // Show the detected activity on the watch screen
    showStatus("Act: " + detectedActivity);
    console.log("Detected Activity:", detectedActivity);
    
  } catch (e) {
    console.log("Error during inference.");
  }
}

// Utility to map activity index to label (depends on the trained model)
function getActivityLabel(index) {
  const labels = ["running", "sitting", "walking"];
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

// Function to change activity label using touchscreen
function selectActivity() {
  const activities = ["walking", "running", "sitting", "jumping", "cycling"];
  
  // Show activity options on screen for selection
  var menu = {
    "Select Activity": () => {},
    "Walking": () => { setActivity("walking"); },
    "Running": () => { setActivity("running"); },
    "Sitting": () => { setActivity("sitting"); },
    "Jumping": () => { setActivity("jumping"); },
    "Cycling": () => { setActivity("cycling"); },
    "< Back": () => { E.showMenu(mainMenu); }
  };
  E.showMenu(menu);
}

// Function to set activity and update the display, including back option
function setActivity(newActivity) {
  activity = newActivity;
  console.log("Activity set to: " + activity);
  
  // Show status with selected activity and an option to go back
  var menu = {
    "" : { "title" : "Activity: " + activity },
    "Start Recording": () => {
      toggleRecording();
    },
    "< Back": () => { selectActivity(); }
  };
  E.showMenu(menu);
}

// Main menu for the app
var mainMenu = {
  "" : { "title" : "Activity Logger & Classifier" },
  "Collect Data": () => { selectActivity(); },
  "Start Inference": () => {
    toggleInference();
  },
  "Exit": () => { load(); }
};

// Set up the app to show the main menu
E.showMenu(mainMenu);

// Load the TensorFlow Lite model on startup
loadModel();

// Use the single button to start or stop recording/inference (short press only)
setWatch(() => {
  if (recording) {
    toggleRecording();
  } else {
    toggleInference();
  }
}, BTN, {repeat: true, edge: "falling", debounce: 50});
