// App to collect accelerometer data and classify activities on Bangle.js 2
// Created on 2024-10-16 21:00:00 by Thomas Vikström

var activity = "";  // Variable to hold the current activity label
var recording = false;
var storageFile = null;  // File reference for storage

Bangle.setOptions({
  gestureEndThresh: Math.pow(500, 2),  // Lower the threshold for ending a gesture
  gestureStartThresh: Math.pow(400, 2),  // Lower the threshold for starting a gesture
  gestureInactiveCount: 3,  // Increase frames required for inactivity
  gestureMinLength: 10      // Keep the minimum length for sustained movement
});


// Load the TensorFlow Lite model on startup (assumed to be already deployed via Edge Impulse)
function loadModel() {
  console.log("Model is assumed to be loaded via Edge Impulse firmware.");
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
    storageFile = require("Storage").open("activity_data.csv", "a");

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

// Function to handle accelerometer data (recording only)
function onAccel(a) {
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
  "" : { "title" : "Activity Logger" },
  "Select Activity": selectActivity,
  "Exit": () => { load(); }
};

// Set up the app to show the main menu
E.showMenu(mainMenu);

// Use the single button to start or stop recording (short press only)
setWatch(toggleRecording, BTN, {repeat: true, edge: "falling", debounce: 50});

// Load the TensorFlow Lite model on startup
loadModel();

// Listen for AI gesture classification results
Bangle.on('aiGesture', (gesture, raw) => {
  console.log("Detected Activity:", gesture, "Raw Output:", raw);
  showStatus("-> " + gesture);
  
  // Clear the message after 1 second
  setTimeout(() => g.clear(), 2000);
});

// Display initial status message
showStatus("Ready to Detect Activity");
