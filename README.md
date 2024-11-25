# Classifying Exercise Activities on a Smartwatch with Edge Impulse

## Intro

This tutorial shows how you can utilize Edge Impulse to classify different exercises or daily activities you are doing. The hardware used in this project is the Bangle.js 2 programmable smartwatch, but any other programmable smartwatch with an accelerometer can be used as long as they support Tensorflow Lite.

![](/images/Bangle-01.jpg)

## Use-case explanation

Overwhelming evidence exists that lifelong exercise is associated with a longer health span, delaying the onset of [40](https://perspectivesinmedicine.cshlp.org/content/8/7/a029694.short) chronic conditions/diseases. The health benefits of doing regular exercise have been shown in many studies. Just to pick [one of them](https://www.kheljournal.com/archives/2016/vol3issue5/PartA/3-4-55-201.pdf) which mentions that physical activity and exercise can reduce stress and anxiety, boost happy chemicals, improve self-confidence, increase the brain power, sharpen the memory and increase our muscles and bones strength. It also helps in preventing and reducing heart disease, obesity, blood sugar fluctuations, cardiovascular diseases and cancer.

Some of us happily exercise without any technology whatsoever and just listen to our body. Others, like me, are motivated (sometimes perhaps obsessed) by following statistics and trends collected by smartwatches or smartrings. Any decent smartwatch today uses GPS, accelerometer, gyrometer and other sensors to collect different types of data and consolidates this data into activity summaries. They perform exceptionally well with correctly classifying long repetetive exercises like running and walking outdoors or on a treadmill, skiing, cycling, rowing, etc. Where many of them still have room for improvement though, is correctly classifying the many different activities you might be performing in a gym. I've been a Garmin fan for over 10 years, and is is currently on my fourth Garmin sport watch, not anyone of them can consistently correctly classify gym activities, even if all of the watches are considered premium sport watches.

Since I last winter started working out with my three decades old Kettler Sport Variant home gym, it bothered me that I needed to constantly switch activities on my Garmin watch when I changed from one activity to another, especially  as I like to avoid longer monotonous repetitions. This made me wonder if I with machine learning can do better than the big boys. Since I from before have experience with using Tensorflow Lite on the affordable smartwatch Bangle.js, I thought I could at least try.

The result is an app where you first collect exercise training data for export to Edge Impulse, and after uploading a trained ML-model just click on Exercise to let the watch classify all different exercises you're performing and the length of them. Afterwards you can upload the collected data to e.g. Excel for further analysis or storage.

![](/images/Kettler%20Sport%20Variant%2003.jpg)


## Components and Hardware/Software Configuration

### Components Needed

- A programmable smartwatch supporting Tensorflow Lite, in this case I used [Bangle.js 2](https://www.espruino.com/Bangle.js2) which is slightly under €100 (with taxes).
- A computer supporting Bluetooth Low Energy (BLE). More or less any computer manufactured the last decade is equipped with BLE, but there are also BLE-adapters with USB-connector for older computers.
- Depending on the activities you plan to do, you might need shoes for walking/running outdoors, gym equipment, kettle bells etc. If you go to a gym, they probably have all the exercise equipment you need.


### Hardware and Software Configuration

#### Hardware Configuration
- There's practically nothing to configure hardware wise! While Bangle is not in same premium class as e.g. Apple or Samsung watches, the initial experience is quite similar, most often everything just works out of the box and it's easy to get started. Do read the [Getting Started guide](https://banglejs.com/start2).

#### Software Configuration

- By following the [getting started steps](https://www.espruino.com/Bangle.js+Development) you'll learn how to develop apps on the smartwatch.
- For this tutorial, it's enough to open the [Espruino IDE](https://www.espruino.com/ide/), connect to your Bangle according to the instructions found in above guide, and paste [this program](/src/collect_and_classify_TF_v0.8.js) to the **right** side in the IDE.

![](/images/Espr_IDE-10.jpg)

- Click on the `RAM`-button to upload the program to the watch. Bangle has both volatile RAM-memory as well as flash-memory for long-term storage. RAM content disappears after power-down, while content in flash remains after power-down. When testing and developing, it is safer to just upload to RAM as possible serious program crashes won't mess up the watch that much as if you save to flash. That said, it is close to impossible to completely brick the watch with a buggy program, a factory reset should help in almost all cases.
- You'll be presented with a simple menu with three options:
    - `Collect Data`    - collect data for different activities
    - `Inference`       - run inference to test the current ML-model without storing any further data
    - `Exercise`        - run inference and also collect what activities were performed and the length of them into a CSV-file


![](/images/Bangle-Exerc-01.jpg)

## Data Collection Process

- Strap the watch to your non-dominant hand.
- Select `Collect Data` on the watch, and select one of the predefined activities, scroll down to see more.
- When you're ready to do the activity, select `Start Recording`.
- Start performing the activity, e.g. walking.
    - Don't change from one activity to another while you are collecting data.
    - There's however no need to try to perform the activity as a robot, just do it naturally. E.g., when walking, just walk as you normally do, take a few turns every now and then, and vary the speed a bit.
    - For the first time, collect a minute or so of data for each activity.
    - Stop recording by quickly pressing the physical button. This will take you back to the main menu.
- Rinse and repeat above for each activity.
- Also collect data for a "non-activity", like sitting.

![](/images/Bangle-Exerc-02.jpg)

![](/images/Bangle-Exerc-03.jpg)

## Building and Training the Model

Building and training a ML-model in this project consist of following major steps:
1. Downloading the recorded activity files to your computer
2. Creating an Edge Impulse project
3. Uploading the activity files to Edge Impulse
4. Creating and training the model

### 1. Download the Activity Files

- In the Espruino IDE, click on the `Storage` icon

![](/images/Espr_IDE-15.jpg)

- Your activity files start with ´acti_...` followed by a timestamp when the file was created.
- For one activity file at a time, click on the `Save` icon.  

![](/images/Espr_IDE-20.jpg)

- Files will be stored in your Downloads-folder on your computer
- Files will be in CSV-format, but have an ` (StorageFile)`-appendix that you'll need to remove. So, rename `acti_2024_10_29_20_03.csv (StorageFile)` to `acti_2024_10_29_20_03.csv`.

### 2. Create an Edge Impulse Project

- Head over to Edge Impulse, and create a object classification project. If you are completely new to the platform, check out their [getting started tutorial](https://docs.edgeimpulse.com/docs/readme/for-beginners#getting-started-in-a-few-steps).

### 3. Upload the Activity Files

This consists of two steps:
- 3.1 Configure the CSV-Wizard
- 3.2 Uploading the activity files themselves

#### 3.1 Configure the CSV-Wizard

Here you'll use one of your activity files as an example model to let the wizard know how the file is structured. This configuration only needs to be done once. Documentation for the CSV-Wizard is found [here](https://docs.edgeimpulse.com/docs/edge-impulse-studio/data-acquisition/csv-wizard).

For this project I've used the following steps:
- Click on `Data acquisition`
- Click on `CSV-Wizard`
- Click on `Choose File`, select any of your activity files, and click `Upload file`
- Check that you have the following columns: 
    - Row #, timestamp, x, y, z, activity
    - Click `Looks good, next`
- Fill in the screen like this:
    - **Is this time-series data?:** `Yes, this is time-series data...`
    - **How is your time-series data formatted?:** `Each row contains a reading, and sensor values are columns.`
    - **Do you have a timestamp or time elapsed column?:**  `Yes, it's <timestamp>`
    - **What type of data is in your timestamp column?:** `Time elapsed in milliseconds`
    - **Override timestamp difference?:** `80 ms` (the default accelerometer is 12.5 Hz which means one sample is 80 ms in length)
- Click `Great, let's look at your values`
- Now you are in Step 4, fill in it like this:
    - **Do you have a column that contains the label (the value you want to predict)?** `Yes, it's <activity>`
    - **Which columns contain your values?** `<x, y, z>`
- Click `Next, split up into samples`
- In this Step 5, use these settings:
    - **How long do you want your samples to be?** `Limit to <3040> ms`
    - **How should we deal with multiple labels in a sample?** `Use the last value of "activity" as the label for each sample...`
- Click `Finish wizard`

#### 3.2 Upload Activity Files Using the CSV-Wizard

Now you can upload all activity files, including the one you used to configure the CSV-Wizard, to Edge Impulse. See the [documentation](https://docs.edgeimpulse.com/docs/edge-impulse-studio/data-acquisition/uploader) for detailed steps.

### 4. Creating, Training, and Testing the Model

This consists of a few steps:
- 4.1 Create an impulse
- 4.2 Generate features
- 4.3 Train the model
- 4.4 Test the model

#### 4.1 Create an Impulse

![](/images/EI-010.jpg)

## Model Deployment

Now it's time to test the model in real life!

- Head over to the Deployment tab, and search for 'OpenMV'

![](/Images/Deployment_compressed.png)

- When just testing, and with smaller models like mine, it's ok to use the library option, but for real production usage it's better to build a firmware version.
- After the build process is completed, instructions are shown for how to deploy the model to the OpenMV camera. With the library option, you just extract the files from the ZIP-file to the camera's memory, while you with the firmware option need to flash the compiled firmware to the camera with help of the OpenMV IDE.
- When the camera is powered, it automatically runs ```main.py``` from its memory. Ensure this progam has the proper image conversions you used in the capturing phase! 
- Run the [Python program](/nuts_conveyor/Dobot%20conveyor%20-%20object%20counting.py) or your own version to receive inferencing data from the OpenMV camera. 
    - Remember that if you want a live video feed, you need to connect a separate camera to your computer



## Results

The results from this project met the objectives, to be able to count objects with the OpenMV camera, using FOMO. The whole solution is not perfect as the accuracy could be improved by adding more data. The current version is counting all the nuts it identifies, but adding a running total would obviously be beneficial in a production scenario. This needs partially another approach on the Python-side as the conveyor belt would need to be paused, inference run on the camera, before resuming. I tried to implement this, but as the conveyor belt is running completely asynchronously, it is challenging to stop at a given time. As the ML-model itself is technically working perfectly, I decided to leave this improvement for later.

![](/Videos/Counting_nuts_with_conveyor_belt.gif)

## Conclusion

The goal of this tutorial was to show how to count objects, using FOMO and the OpenMV Cam RT-1062. As mentioned, the goal was achieved, and while a few technical issues occurred on the conveyor belt side, the overall process was quite straightforward. 

All the code and files used in this write-up are found from [Github](https://github.com/baljo/count_nuts), the public Edge Impulse project is [here](https://studio.edgeimpulse.com/studio/527570). Feel free to clone the project for your own use case.
