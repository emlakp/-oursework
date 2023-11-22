class TrainingLogger:
    def __init__(self, file_path):
        self.file_path = file_path

    def log_measurements(self, train_loss, train_acc, val_loss, val_acc):
        with open(self.file_path, 'a') as file:
            file.write(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")
