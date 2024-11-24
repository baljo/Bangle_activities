// Combined activity logger and classifier using TensorFlow on Bangle.js 2, now with exercise tracking
// Created on 2024-10-26 19:49 by Thomas Vikström

var activity = "";  // Variable to hold the current activity label
var recording = false;
var bufferSize = 75;  // Adjustable buffer size (x, y, z for 25 samples)
var accelBuffer = new Float32Array(bufferSize).fill(0);  // Initialize fixed-size buffer with zeros
var storageFile = null;  // File reference for storage
var inferenceInterval = null;  // Interval reference for inference timing
var activityTracking = {}; // Object to track activity repetition and timing
var currentActivityStart = null; // Start time for an activity

function convertTFLiteToBase64(filename) {
  try {
    let file = require("Storage").read(filename);
    if (!file) {
      console.log("Error: File not found");
      return;
    }
    let base64Data = btoa(file);
    console.log("Base64 Conversion Successful:");
    require("Storage").write(filename + ".b64", base64Data);
  } catch (e) {
    console.log("Error during conversion:", e);
  }
}

convertTFLiteToBase64("impulse4");

var m = require("Storage").read("impulse4.b64");
var model = atob(m);
var tf = require("tensorflow").create(15000, E.toString(model));

function loadModel() {
  try {
    model = require("tensorflow").create(model.length, model);
    console.log("Model loaded successfully");
  } catch (e) {
    console.log("Error loading model:", e);
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
    accelBuffer.fill(0);
    let now = new Date();
    let filename = "acti_" +
      now.getFullYear() + "_" +
      (now.getMonth() + 1).toString().padStart(2, '0') + "_" +
      now.getDate().toString().padStart(2, '0') + "_" +
      now.getHours().toString().padStart(2, '0') + "_" +
      now.getMinutes().toString().padStart(2, '0') + ".csv";
    storageFile = require("Storage").open(filename, "a");
    if (storageFile.getLength() === 0) {
      storageFile.write("timestamp,x,y,z,activity");
    }
    currentActivityStart = now;
    Bangle.on('accel', onAccel);
    showStatus("Recording: " + activity);
  } else {
    console.log("Stopped recording.");
    Bangle.removeListener('accel', onAccel);
    if (storageFile) {
      let now = new Date();
      storageFile.write(currentActivityStart.toISOString() + "," + now.toISOString() + "," + activity + "");
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
      storageFile.write(currentActivityStart.toISOString() + "," + now.toISOString() + "," + activity + "");
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
      storageFile.write("start_timestamp,end_timestamp,activity");
    }
    currentActivityStart = now;
    Bangle.on('accel', onAccel);
    inferenceInterval = setInterval(performInference, 2000);
    showStatus("Inferring...");
  }

}

function onAccel(a) {
  for (let i = 0; i < bufferSize - 3; i++) {
    accelBuffer[i] = accelBuffer[i + 3];
  }
  accelBuffer[bufferSize - 3] = a.x;
  accelBuffer[bufferSize - 2] = a.y;
  accelBuffer[bufferSize - 1] = a.z;

  if (recording && storageFile) {
    var dataLine = [Date.now(), a.x, a.y, a.z, activity].join(",") + "";
    try {
      storageFile.write(dataLine);
      console.log("Data written: ", dataLine);
    } catch (e) {
      console.log("Error writing to file:", e);
    }
    try {
      storageFile.write(dataLine);
      console.log("Data written: ", dataLine);
    } catch (e) {
      console.log("Error writing to file:", e);
    }
  }
}

function performInference() {
  if (!model) {
    return;
  }
  try {
    model.getInput().set(accelBuffer);
    model.invoke();
    let output = model.getOutput();
    console.log(output);
    let maxIndex = 0;
    for (let i = 1; i < output.length; i++) {
      if (output[i] > output[maxIndex]) {
        maxIndex = i;
      }
    }
    let detectedActivity = getActivityLabel(maxIndex);
    trackActivity(detectedActivity);
  if (activity !== detectedActivity && activityTracking[detectedActivity].count >= 10) {
    if (currentActivityStart && storageFile) {
      let now = new Date();
      storageFile.write(currentActivityStart.toISOString() + "," + now.toISOString() + "," + activity + "");
    }
    activity = detectedActivity;
    currentActivityStart = new Date();
  }
    showStatus("Act: " + detectedActivity);
    console.log("Detected Activity:", detectedActivity);
  } catch (e) {
    console.log("Error during inference.");
  }
}

function trackActivity(detectedActivity) {
  if (!activityTracking[detectedActivity]) {
    activityTracking[detectedActivity] = { count: 0, startTime: null };
  }
  if (!activityTracking[detectedActivity].startTime) {
    activityTracking[detectedActivity].startTime = new Date();
  }
  if (activityTracking[detectedActivity].startTime === null) {
    activityTracking[detectedActivity].startTime = new Date();
  }
  if (activityTracking[detectedActivity].startTime && (new Date() - activityTracking[detectedActivity].startTime) >= 20000) {
    if (detectedActivity !== activity) {
      if (currentActivityStart) {
        let now = new Date();
        storageFile.write(currentActivityStart.toISOString() + "," + now.toISOString() + "," + activity + "\n");
      }
      activity = detectedActivity;
      currentActivityStart = new Date();
    }
  }
}

function getActivityLabel(index) {
  const labels = ["running", "sitting", "walking", "jumping", "cycling"];
  return labels[index] || "unknown";
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

function setActivity(newActivity) {
  activity = newActivity;
  console.log("Activity set to: " + activity);
  var menu = {
    "" : { "title" : "Activity: " + activity },
    "Start Recording": () => {
      activity = 'exercise'; toggleRecording();
    },
    "< Back": () => { selectActivity(); }
  };
  E.showMenu(menu);
}

var mainMenu = {
  "" : { "title" : "Activity Logger & Classifier",
       "fontHeight": 8,},
  "Collect Data": () => { selectActivity(); },
  "Start Inference": () => {
    toggleInference();
  },
  "Exercise": () => {
    toggleInference();
  },
  "Exit": () => { load(); }
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
