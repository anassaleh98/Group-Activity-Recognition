'''
Baseline 2 part B :
    Max pool all the features for every person in the frame(2048)
    and passed through a classifier to classify group activities
'''
from model import get_feature_extractor, FrameClassifier, train_model, validate_model, create_confusion_matrix
from data_preparation import  trainloader, valloader, testloader

if __name__ == "__main__":
    # Load feature extractor
    model_load_path = "/kaggle/input/person_classifier_model_2/pytorch/default/1/person_classifier_model_2.pth"
    feature_extractor = get_feature_extractor(model_load_path)

    # Initialize classifier
    frame_classifier = FrameClassifier(input_dim=2048, num_classes=8)

    # Train the model
    train_model(frame_classifier, feature_extractor, trainloader, valloader, epochs=5, lr=0.001)

    # Test  
    print("Testing model performance:")
    validate_model(frame_classifier, feature_extractor, testloader)

    # Test confusion matrix
    create_confusion_matrix(frame_classifier, feature_extractor, testloader)