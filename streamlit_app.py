{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "!pip install face_recognition\n",
        "!pip install opencv-python\n",
        "!pip install pandas"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dgS8LKfx8FNo",
        "outputId": "82ba06ac-c972-4fec-8166-914aebf9e986"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting face_recognition\n",
            "  Downloading face_recognition-1.3.0-py2.py3-none-any.whl.metadata (21 kB)\n",
            "Collecting face-recognition-models>=0.3.0 (from face_recognition)\n",
            "  Downloading face_recognition_models-0.3.0.tar.gz (100.1 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m100.1/100.1 MB\u001b[0m \u001b[31m8.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "Requirement already satisfied: Click>=6.0 in /usr/local/lib/python3.10/dist-packages (from face_recognition) (8.1.7)\n",
            "Requirement already satisfied: dlib>=19.7 in /usr/local/lib/python3.10/dist-packages (from face_recognition) (19.24.2)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from face_recognition) (1.26.4)\n",
            "Requirement already satisfied: Pillow in /usr/local/lib/python3.10/dist-packages (from face_recognition) (10.4.0)\n",
            "Downloading face_recognition-1.3.0-py2.py3-none-any.whl (15 kB)\n",
            "Building wheels for collected packages: face-recognition-models\n",
            "  Building wheel for face-recognition-models (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for face-recognition-models: filename=face_recognition_models-0.3.0-py2.py3-none-any.whl size=100566162 sha256=a0f58e2a17261ba724f33b2e7b7684039708c7449933b11113e4a3cfb21dc67e\n",
            "  Stored in directory: /root/.cache/pip/wheels/7a/eb/cf/e9eced74122b679557f597bb7c8e4c739cfcac526db1fd523d\n",
            "Successfully built face-recognition-models\n",
            "Installing collected packages: face-recognition-models, face_recognition\n",
            "Successfully installed face-recognition-models-0.3.0 face_recognition-1.3.0\n",
            "Requirement already satisfied: opencv-python in /usr/local/lib/python3.10/dist-packages (4.10.0.84)\n",
            "Requirement already satisfied: numpy>=1.21.2 in /usr/local/lib/python3.10/dist-packages (from opencv-python) (1.26.4)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (2.2.2)\n",
            "Requirement already satisfied: numpy>=1.22.4 in /usr/local/lib/python3.10/dist-packages (from pandas) (1.26.4)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas) (2024.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import face_recognition\n",
        "import pickle\n",
        "import cv2\n",
        "\n",
        "# Function to train the model and save the face encodings\n",
        "def train_and_save_model(student_images_folder, model_file='trained_model.pickle'):\n",
        "    student_encodings = []\n",
        "    student_names = []\n",
        "\n",
        "    # Loop through student images and encode their faces\n",
        "    for filename in os.listdir(student_images_folder):\n",
        "        if filename.endswith(('.jpg', '.png')):\n",
        "            img_path = os.path.join(student_images_folder, filename)\n",
        "            image = face_recognition.load_image_file(img_path)\n",
        "\n",
        "            # Resize the image for faster processing\n",
        "            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)\n",
        "\n",
        "            # Encode the face\n",
        "            encoding = face_recognition.face_encodings(image)[0]\n",
        "            student_encodings.append(encoding)\n",
        "            student_names.append(os.path.splitext(filename)[0])  # Use the filename as the student name\n",
        "\n",
        "    # Save the encodings and names to a file\n",
        "    with open(model_file, 'wb') as f:\n",
        "        pickle.dump((student_encodings, student_names), f)\n",
        "\n",
        "    print(f'Model trained and saved to {model_file}')\n",
        "\n",
        "# Call the function to train and save the model\n",
        "student_images_folder = '/content/drive/MyDrive/Smart Attandance System/Images'  # Folder containing student images\n",
        "train_and_save_model(student_images_folder)\n"
      ],
      "metadata": {
        "id": "H5KImEv950x-",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d65c40de-3581-4851-8bc3-ade217f20bde"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/PIL/Image.py:3368: DecompressionBombWarning: Image size (108000000 pixels) exceeds limit of 89478485 pixels, could be decompression bomb DOS attack.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model trained and saved to trained_model.pickle\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import cv2\n",
        "import face_recognition\n",
        "import pandas as pd\n",
        "import pickle\n",
        "from datetime import datetime\n",
        "\n",
        "# Load the pretrained model (face encodings and student names)\n",
        "def load_trained_model(model_file='trained_model.pickle'):\n",
        "    with open(model_file, 'rb') as f:\n",
        "        student_encodings, student_names = pickle.load(f)\n",
        "    return student_encodings, student_names\n",
        "\n",
        "# Mark attendance based on recognized faces\n",
        "def mark_attendance(student_names, recognized_names):\n",
        "    now = datetime.now()\n",
        "    date = now.strftime('%d-%m-%Y')\n",
        "    time = now.strftime('%H:%M')\n",
        "\n",
        "    # Create attendance DataFrame\n",
        "    attendance = pd.DataFrame(student_names, columns=['Student Name'])\n",
        "    attendance['Date'] = date\n",
        "    attendance['Time'] = time\n",
        "    attendance['Status'] = ['Present' if name in recognized_names else 'Absent' for name in student_names]\n",
        "\n",
        "    return attendance\n",
        "\n",
        "# Recognize faces in a group photo\n",
        "def recognize_faces_in_group_photo(group_photo_path, student_encodings, student_names):\n",
        "    group_photo = face_recognition.load_image_file(group_photo_path)\n",
        "\n",
        "    # Resize group photo for faster face detection\n",
        "    small_frame = cv2.resize(group_photo, (0, 0), fx=0.5, fy=0.5)\n",
        "\n",
        "    # Detect faces in the group photo (scaled down)\n",
        "    face_locations = face_recognition.face_locations(small_frame)\n",
        "\n",
        "    # Scale back up face locations\n",
        "    face_locations = [(top*2, right*2, bottom*2, left*2) for top, right, bottom, left in face_locations]\n",
        "\n",
        "    # Extract face encodings from the detected faces\n",
        "    face_encodings = face_recognition.face_encodings(group_photo, face_locations)\n",
        "\n",
        "    recognized_names = []\n",
        "\n",
        "    for encoding in face_encodings:\n",
        "        # Compare the group photo encodings with the student encodings\n",
        "        matches = face_recognition.compare_faces(student_encodings, encoding)\n",
        "        face_distances = face_recognition.face_distance(student_encodings, encoding)\n",
        "        best_match_index = face_distances.argmin()\n",
        "\n",
        "        if matches[best_match_index]:\n",
        "            recognized_names.append(student_names[best_match_index])\n",
        "\n",
        "    return recognized_names\n",
        "\n",
        "# Main function to load the model and mark attendance\n",
        "def main(group_photo_path, model_file='trained_model.pickle'):\n",
        "    # Step 1: Load the pretrained model\n",
        "    student_encodings, student_names = load_trained_model(model_file)\n",
        "\n",
        "    # Step 2: Recognize faces in the group photo\n",
        "    recognized_names = recognize_faces_in_group_photo(group_photo_path, student_encodings, student_names)\n",
        "\n",
        "    # Step 3: Mark attendance\n",
        "    attendance = mark_attendance(student_names, recognized_names)\n",
        "\n",
        "    # Step 4: Save or display attendance\n",
        "    print(attendance)\n",
        "    attendance.to_csv('attendance.csv', index=False)\n",
        "    print(\"Attendance saved to 'attendance.csv'.\")\n",
        "\n",
        "# Example usage\n",
        "group_photo_path = '/content/drive/MyDrive/Smart Attandance System/IMG20241023152326.jpg'  # Path to the group photo\n",
        "main(group_photo_path)\n"
      ],
      "metadata": {
        "id": "yaSTGonV51j5",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6124aae4-77ab-4f78-d8d8-4460ddeccdf6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/PIL/Image.py:3368: DecompressionBombWarning: Image size (108000000 pixels) exceeds limit of 89478485 pixels, could be decompression bomb DOS attack.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "        Student Name        Date   Time   Status\n",
            "0       Shivam_Kumar  11-11-2024  06:58  Present\n",
            "1           Hans_Raj  11-11-2024  06:58  Present\n",
            "2  Amritangshu_Singh  11-11-2024  06:58  Present\n",
            "3         Khyati_Raj  11-11-2024  06:58  Present\n",
            "4        Nawneet_Raj  11-11-2024  06:58  Present\n",
            "Attendance saved to 'attendance.csv'.\n"
          ]
        }
      ]
    }
  ]
}
