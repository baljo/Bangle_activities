// Set the accelerometer polling interval to 10 ms (equivalent to 100 Hz)
Bangle.setPollInterval(10);

// Now you can listen for accelerometer events at this frequency
Bangle.on('accel', (accelData) => {
  console.log("Accel Data:", accelData);
});
