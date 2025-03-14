'''
baseline 4:
A video classifier using ResNet-50 as a feature extractor,
followed by an LSTM layer to capture temporal dependencies across 9-frame sequences. 
'''


from model import VideoClassifier, train_model, validate_model, create_confusion_matrix
from data_preparation import  trainloader, valloader, testloader

if __name__ == "__main__" :
    # Initialize video classification model
    model = VideoClassifier(lstm_hidden=512) 
    
    # Start training the model
    print("Start training...")
    train_model(model, trainloader, valloader, epochs=4)
    
    # Evaluate model performance on the test set
    print("\nTesting model performance:")
    validate_model(model, testloader)
    
    # Generate and display the confusion matrix and classification report
    create_confusion_matrix(model, testloader)


