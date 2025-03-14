'''
Baseline 2 part A: person classifier
'''
from model import train, evaluate
from data_preparation import testloader

if __name__ == "__main__" :
    # strat training
    train()
    
    # Test  
    print("\nTesting model performance:")
    evaluate(testloader, "Test")    
    
    # save the model
    #torch.save(model.state_dict(), '/kaggle/working/person_classifier_model_2.pth')
    

# ---------------
# Testing model performance on Test dataset:
    
# Test Loss: 0.6958,
# Accuracy: 77.57%,
# F1 Score: 0.7480    