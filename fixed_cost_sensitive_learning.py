import torch
import torch.nn as nn

class BinaryCostSensitiveLoss(nn.Module):
    
    def __init__(self, major_class=0, minor_class=1, lambda_=0.5):
        super(BinaryCostSensitiveLoss, self).__init__()
        self.major_class = major_class
        self.minor_class = minor_class
        self.lambda_ = lambda_
    
    def forward(self, y_pred, y_true):
        # Compute the errors
        errors_sq =  torch.square(y_true - y_pred)
        
        # Compute the squared errors for each class
        j1 = torch.sum(errors_sq[y_true == self.minor_class])
        j2 = torch.sum(errors_sq[y_true == self.major_class])
        
        # Compute the cost-sensitive loss
        loss = self.lambda_ * j1 + (1 - self.lambda_) * j2
    
        return loss

class MultiClassCostSensitiveLoss(nn.Module):
    
    def __init__(self, class_weights):
        """
        Args:
            class_weights (dict): A dictionary where keys are class labels (e.g., 0, 1, 2, ...)
                                  and values are the corresponding weights for each class.
        """
        super(MultiClassCostSensitiveLoss, self).__init__()
        self.class_weights = class_weights
    
    def forward(self, y_pred, y_true):
        # Initialize the loss
        total_loss = 0.0
        if len(y_true) == 0:
            return total_loss
        
        # Compute the probabilities for each class
        probs = torch.softmax(y_pred, dim=1)
        
        # Compute the squared errors
        squared_errors = []
        for prob, y in zip(probs, y_true + 1): # Add 1 to the true label to match the indices of class in probs
            squared_errors.append(torch.square(prob[y] - y + 1))
        squared_errors = torch.stack(squared_errors)
        
        # Accumulate the weighted loss for each class
        for class_label, weight in self.class_weights.items():
            class_loss = torch.sum(squared_errors[y_true == class_label])
            total_loss += weight * class_loss
        
        return total_loss
    

def main():
    # # Define the true and predicted labels
    # y_true = torch.tensor([0, 0, 0, 1, 0, 1])
    # y_pred = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    
    # # Compute the cost-sensitive loss
    # lambda_ = lambda_ = len(y_true[y_true == 0]) / len(y_true)
    # loss_fn = BinaryCostSensitiveLoss(lambda_=lambda_)
    # loss_fn_default = BinaryCostSensitiveLoss()
    
    # loss = loss_fn(y_pred, y_true)
    # loss_default = loss_fn_default(y_pred, y_true)
    
    # print("Cost-sensitive loss:", loss.item())
    # print("Cost-sensitive loss (default):", loss_default.item())
    pass
    
if __name__ == '__main__':
    main()