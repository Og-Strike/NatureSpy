# Nature Spy - Animals and Plants Detection System System 

The goal of this project is to create a **robust system for identifying and classifying animals and plants**. It uses a **hierarchical two-layer classification structure**, combining **ResNet18** for general object recognition and **ResNet152** for fine-grained categorization. Additionally, **Faster R-CNN** is used for object detection. The entire system is integrated into a **Kivy app** with an intuitive interface and **chatbot functionality** to enhance user engagement.

---

## 🚀 Approach!

### 🔹 First Layer: ResNet18 (General Classification)
- **Model**: Fine-tuned ResNet18
- **Objective**: Distinguish between animals, plants, and irrelevant objects
- **Preprocessing**:
  - Resized images to **224x224** for consistency and performance
  - Applied **data augmentation** techniques (e.g., rotation) to improve robustness
- **Outcome**: Efficient filtering of general categories in real-world conditions

---

### 🔹 Second Layer: ResNet152 (Detailed Classification)
- **Model**: Fine-tuned ResNet152
- **Objective**: Provide detailed classification
  - Animals (e.g., dog, cat)
  - Plants (e.g., rose, sunflower)
- **Preprocessing**:
  - Consistent resizing and augmentation as in the first layer
- **Outcome**: Enhanced classification precision and generalization

---

### 🔹 Object Detection: Faster R-CNN
- **Technique**: Faster R-CNN with default weights
- **Objective**: Detect and localize relevant objects in the image
- **Method**:
  - Used **bounding boxes** to filter for animals and plants
- **Outcome**: Accurate and fast object identification

---

## 🖼️ User-Friendly Interface (Kivy)
- **Framework**: Kivy
- **Objective**: Create a seamless and interactive user experience
- **Features**:
  - Options to choose between **ROI-based** and **object detection-based** analysis
  - Resizing and augmentation integrated directly into the app
- **Outcome**: Simplified usage for end-users without technical background

---

## 🤖 ChatBot Interaction (Botpress + GPT)
- **Tool**: Botpress with NLP and GPT-trained models
- **Objective**: Enhance user interaction via natural language conversations
- **Features**:
  - Ask detailed questions about the identified object
  - Get contextual and characteristic insights
- **Technologies Used**:
  - Natural Language Understanding (NLU)
  - GPT-based responses
- **Outcome**: Interactive, informative, and engaging user experience

---

## 🌟 Summary
This project effectively integrates **deep learning models (ResNet18, ResNet152, Faster R-CNN)** with a **Kivy-based UI** and **chatbot interface** to deliver a powerful, accurate, and user-friendly system for identifying and understanding animals and plants from images.
