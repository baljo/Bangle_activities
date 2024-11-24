// App to collect accelerometer data for different activities on Bangle.js 2
// Created on 2024-10-16 19:15:00 by Thomas Vikström

var activity = "";  // Variable to hold the current activity label
var recording = false;
var storageFile = null;  // File reference for storage

// Function to start/stop recording using the physical button (short press)
function toggleRecording() {
  if (activity === "") {
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

// Function to handle accelerometer data
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
    "Idling" : () => { setActivity("idling"); },
    "Jumping": () => { setActivity("jumping"); },
    "Cycling": () => { setActivity("cycling"); },
    "< Back" : () => { E.showMenu(mainMenu); }
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
