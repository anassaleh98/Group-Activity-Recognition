from model import train, evaluate, create_confusion_matrix
from data_preparation import testloader

# strat training
train()

# Test  
print("\nTesting model performance:")
evaluate(testloader, "Test")

# Test confusion matrix
create_confusion_matrix(testloader)



# Baseline 1 eval on testset
#-----------------------------------

#      Loss    : 0.5394,
#      Accuracy: 85.12%,
#      F1 Score: 0.8516

# Classification Report:

#               precision    recall  f1-score   support

#        r_set      0.797     0.797     0.797       192
#      r_spike      0.936     0.850     0.891       173
#       r-pass      0.776     0.824     0.799       210
#   r_winpoint      0.883     0.954     0.917        87
#   l_winpoint      0.915     0.951     0.933       102
#       l-pass      0.866     0.801     0.832       226
#      l-spike      0.929     0.872     0.899       179
#        l_set      0.787     0.881     0.831       168

#     accuracy                          0.851      1337
#    macro avg      0.861     0.866     0.862      1337
# weighted avg      0.854     0.851     0.852      1337