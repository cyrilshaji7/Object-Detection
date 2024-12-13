

---

# Object Detection Application Documentation

## 1. Introduction

This document provides detailed instructions for using an application built with Flask that uses the YOLOv8 model to perform image recognition tasks. The application allows users to detect objects in images or video streams and includes functionality for labeling, saving data, and retraining models.

## 2. System Requirements

Before using the application, ensure that your system meets the following requirements:

- **Operating System:** Windows, macOS, or Linux
- **Python:** Version 3.7 or above
- **Docker:** Installed and running (Optional but recommended)
- **Git:** Installed for cloning the repository (Optional)

If you do not have Python, Docker, or Git installed, follow the steps below.

## 3. Installing Prerequisites

### 3.1. Installing Python

1. **Download Python:**
   - Visit the official Python website: [python.org](https://www.python.org/downloads/)
   - Download the latest version for your operating system.

2. **Install Python:**
   - Run the downloaded installer.
   - Make sure to check the option to "Add Python to PATH" before clicking Install.

3. **Verify the Installation:**
   - Open a terminal or command prompt.
   - Type the following command to verify the installation:
     ```sh
     python --version
     ```
   - You should see the Python version you installed.

### 3.2. Installing Git

1. **Download Git:**
   - Visit the official Git website: [git-scm.com](https://git-scm.com/)
   - Download the installer for your operating system.

2. **Install Git:**
   - Run the downloaded installer.
   - Follow the default installation options unless you have specific preferences.

3. **Verify the Installation:**
   - Open a terminal or command prompt.
   - Type the following command to verify the installation:
     ```sh
     git --version
     ```
   - You should see the Git version you installed.

### 3.3. Installing Docker (Optional)

Docker is recommended but not required. If you choose to install Docker, follow these steps:

1. **Download Docker Desktop:**
   - Visit the official Docker website: [docker.com](https://www.docker.com/products/docker-desktop)
   - Download Docker Desktop for your operating system.

2. **Install Docker Desktop:**
   - Run the downloaded installer.
   - Follow the installation instructions.

3. **Verify the Installation:**
   - Open a terminal or command prompt.
   - Type the following command to verify the installation:
     ```sh
     docker --version
     ```
   - You should see the Docker version you installed.

## 4. Project Structure

Below is the structure of the project:

```
IMG-REC/
│
├── flask/
│   ├── .venv/              # Python virtual environment
│   ├── runs/               # YOLO model results storage
│   ├── templates/          # HTML templates for the Flask app
│   ├── train/              # Training data and configuration files
│   ├── app.py              # Main Flask application file
│   ├── data.yaml           # YOLO model data configuration file
│   ├── Dockerfile          # Dockerfile for the Flask app
│   ├── label_counter.txt   # File for tracking the number of labels
│   ├── requirements.txt    # Python dependencies for the Flask app
│   ├── yolov8n.pt          # YOLOv8 model file
│
├── label-studio/
│   ├── Dockerfile          # Dockerfile for Label Studio integration
└── docker-compose.yml  # Docker Compose configuration
```

## 5. Installation

### 5.1. Clone the Repository

First, clone the repository from GitHub to your local machine.

#### 5.1.1. If Git is Installed

1. Open a terminal or command prompt.
2. Clone the repository:
   ```sh
   git clone https://github.com/cyrilshaji7/Object-Detection
   cd IMG-REC
   ```

#### 5.1.2. If Git is Not Installed

1. Download the repository as a ZIP file:
   - Go to the GitHub repository page.
   - Click on the "Code" button.
   - Select "Download ZIP".

2. Extract the ZIP file:
   - Right-click on the downloaded ZIP file and select "Extract All" or use your preferred extraction tool.
   - Navigate to the extracted folder.

### 5.2. Python Environment Setup

If Python is installed, you can set up the environment.

#### Option 1: Virtual Environment (Recommended)

1. Create a virtual environment:
   ```sh
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - **Windows:**
     ```sh
     .venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```sh
     source .venv/bin/activate
     ```

3. Install the dependencies:
   ```sh
   pip install -r flask/requirements.txt
   ```

#### Option 2: Without Virtual Environment

If you prefer to install the dependencies globally:
```sh
pip install -r flask/requirements.txt
```

### 5.3. Running the Application

1. Navigate to the Flask directory:
   ```sh
   cd flask
   ```

2. Run the Flask application:
   ```sh
   python app.py
   ```

3. **Access the Application:** Open your web browser and go to [http://localhost:5000](http://localhost:5000) to use the application.

## 6. Docker Setup (Optional)

If you installed Docker, you can use it to run the application in a containerized environment.

### 6.1. Docker for Flask Application

1. Build the Docker image:
   ```sh
   docker-compose build
   ```

2. Run the Docker container:
   ```sh
   docker-compose up
   ```

3. **Access the Application:** Open your web browser and go to [http://localhost:5000](http://localhost:5000).

4. **Access Label Studio:** Open your web browser and go to [http://localhost:8080](http://localhost:8080) to use Label Studio.

## 7. Usage

![image](https://github.com/user-attachments/assets/cbce7f3a-0455-483b-977c-be963460d3f1)


### 7.1. Object Detection

1. Navigate to the main page of the application.
2. Click on "Start Detection" to run the YOLOv8 model and see the detection logs in real time.
3. You can click on “Download Csv” to download the logs with a timestamp.

### 7.2. Labeling and Saving Data

1. After detection, you can label the detected objects by going to Label Studio.
2. Select your project and go to settings, then click on "Cloud Storage" and select "Local Storage."
3. Select the absolute path of `train/images` from the root directory and paste it there.
4. Select and label the images you want.
5. The images and labels are saved in the `train/images` and `train/labels` directories automatically after you click submit.

### 7.3. Retraining the Model

1. To retrain the model, click on “Retraining” to start training on the available images and labels.

---
