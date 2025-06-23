import streamlit as st
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

# Define the same CNN model architecture from the notebook
class Cifar10CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def forward(self, xb):
        return self.network(xb)

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Image preprocessing for CIFAR-10
def preprocess_image(image):
    """Preprocess uploaded image for model prediction"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transformations
    image_tensor = transform(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def predict_image(model, image_tensor):
    """Make prediction on preprocessed image"""
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), confidence.item(), probabilities[0]

def create_trained_model():
    """Load and return a model with trained weights"""
    model = Cifar10CnnModel()
    model_path = "cifar10-cnn.pth"

    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            return model
        except Exception as e:
            st.error("❌ Failed to load the trained model weights.")
            st.exception(e)
            return model
    else:
        st.warning("⚠️ Trained model weights not found. Using random weights.")
        return model


def load_sample_images():
    """Load sample CIFAR-10 images for demonstration"""
    try:
        # Download CIFAR-10 dataset for samples
        dataset = CIFAR10(root='./data', train=False, download=True, 
                         transform=transforms.ToTensor())
        
        # Get a few sample images
        samples = []
        for i in range(min(10, len(dataset))):
            image, label = dataset[i]
            # Convert tensor to PIL Image
            image_pil = transforms.ToPILImage()(image)
            samples.append((image_pil, label))
        
        return samples
    except:
        return []

def main():
    st.set_page_config(
        page_title="CIFAR-10 CNN Classifier",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("🖼️ CIFAR-10 CNN Image Classifier")
    st.markdown("""
    This web application uses a Convolutional Neural Network (CNN) to classify images 
    into one of 10 CIFAR-10 categories. The model architecture is based on the tutorial 
    notebook and achieves similar accuracy results.
    """)
    
    # Display CIFAR-10 classes
    st.sidebar.header("📋 CIFAR-10 Classes")
    for i, class_name in enumerate(CIFAR10_CLASSES):
        st.sidebar.write(f"{i}: {class_name}")
    
    # Model information
    st.sidebar.header("🧠 Model Information")
    st.sidebar.markdown("""
    **Architecture**: CNN with 3 convolutional blocks
    - **Conv Layers**: 6 total (32→64→128→128→256→256 channels)
    - **Pooling**: MaxPool2d after each block
    - **FC Layers**: 1024→512→10 neurons
    - **Expected Accuracy**: ~76% (similar to notebook)
    """)
    
    # Load model
    if 'model' not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.model = create_trained_model()
        st.success("Model loaded successfully!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📤 Upload an Image")
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a 32x32 image or any size image (it will be resized to 32x32)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Make prediction
            with st.spinner("Classifying image..."):
                image_tensor = preprocess_image(image)
                predicted_class, confidence, probabilities = predict_image(
                    st.session_state.model, image_tensor
                )
            
            # Display results
            st.header("🎯 Prediction Results")
            
            # Top prediction
            predicted_label = CIFAR10_CLASSES[predicted_class]
            st.success(f"**Predicted Class**: {predicted_label}")
            st.info(f"**Confidence**: {confidence:.2%}")
            
            # Probability distribution
            st.subheader("📊 Class Probabilities")
            prob_data = {
                'Class': CIFAR10_CLASSES,
                'Probability': [prob.item() for prob in probabilities]
            }
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(prob_data['Class'], prob_data['Probability'])
            ax.set_ylabel('Probability')
            ax.set_title('Class Probability Distribution')
            ax.set_ylim(0, 1)
            
            # Highlight the predicted class
            bars[predicted_class].set_color('red')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show top 3 predictions
            st.subheader("🏆 Top 3 Predictions")
            top_3_indices = torch.topk(probabilities, 3).indices
            for i, idx in enumerate(top_3_indices):
                class_name = CIFAR10_CLASSES[idx]
                prob = probabilities[idx].item()
                st.write(f"{i+1}. **{class_name}**: {prob:.2%}")
    
    # with col2:
    #     st.header("🎲 Try Sample Images")
        
    #     if st.button("Load CIFAR-10 Samples"):
    #         with st.spinner("Loading sample images..."):
    #             samples = load_sample_images()
            
    #         if samples:
    #             st.session_state.samples = samples
    #             st.success(f"Loaded {len(samples)} sample images!")
    #         else:
    #             st.warning("Could not load sample images. Try uploading your own!")
        
        if 'samples' in st.session_state:
            sample_idx = st.selectbox(
                "Choose a sample image:",
                range(len(st.session_state.samples)),
                format_func=lambda x: f"Sample {x+1} (True: {CIFAR10_CLASSES[st.session_state.samples[x][1]]})"
            )
            
            if st.button("Classify Sample"):
                sample_image, true_label = st.session_state.samples[sample_idx]
                
                # Display sample image
                st.image(sample_image, caption=f"Sample Image (True: {CIFAR10_CLASSES[true_label]})")
                
                # Make prediction
                image_tensor = preprocess_image(sample_image)
                predicted_class, confidence, probabilities = predict_image(
                    st.session_state.model, image_tensor
                )
                
                # Show results
                predicted_label = CIFAR10_CLASSES[predicted_class]
                true_label_name = CIFAR10_CLASSES[true_label]
                
                if predicted_class == true_label:
                    st.success(f"✅ Correct! Predicted: {predicted_label}")
                else:
                    st.error(f"❌ Incorrect! Predicted: {predicted_label}, True: {true_label_name}")
                
                st.write(f"Confidence: {confidence:.2%}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📚 About this Application
    This CNN model is based on the CIFAR-10 tutorial notebook and implements the same architecture:
    - **6 Convolutional layers** with increasing channel depth
    - **3 MaxPooling layers** for spatial dimension reduction  
    - **3 Fully connected layers** for final classification
    - **ReLU activations** throughout the network
    
    The model expects 32x32 RGB images and classifies them into 10 categories from the CIFAR-10 dataset.
    """)

if __name__ == "__main__":
    main()
