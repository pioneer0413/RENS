#EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        """
        Args:
            patience (int): 성능 개선 없이 기다릴 에폭 수
            min_delta (float): 개선으로 간주할 최소 변화
        """
        print("DEPRECATED WARNING: This class will be unable since (v1.1.0).")
        
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0
    """
    def save_checkpoint(self, val_loss, model):
        '''성능이 향상되면 모델을 저장합니다.'''
        torch.save(model.state_dict(), 'checkpoint.pth')
    """