import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from torch           import relu, sigmoid, no_grad, save, tanh
from torch.nn        import Module, Linear, BCELoss, Dropout


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyClassifier(Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob):
        super().__init__()
        self.hidden_layer1  = Linear(input_dim,      hidden_dim)
        self.dropout_layer1 = Dropout(dropout_prob)
        self.hidden_layer2  = Linear(hidden_dim,     hidden_dim//2)
        self.dropout_layer2 = Dropout(dropout_prob)
        self.hidden_layer3  = Linear(hidden_dim//2,  hidden_dim//4)
        self.output_layer   = Linear(hidden_dim//4,  output_dim)
        self.loss_fn = BCELoss()
        self.global_epoch = 0
    

    def forward(self, x, y):
        pred = self.dropout_layer1(relu(self.hidden_layer1(   x)))
        pred = self.dropout_layer2(relu(self.hidden_layer2(pred)))
        pred = self.output_layer(  relu(self.hidden_layer3(pred)))
        #pred = self.output_layer(tanh(self.hidden_layer3(tanh(self.hidden_layer2(tanh(self.hidden_layer1(x)))))))
        prob = sigmoid(pred)
        result = {'predictions': pred, 'probabilities': prob}
        if y is not None:
            loss = self.loss(prob, y)
            result['loss'] = loss
        return result


    def loss(self, pred, y):
        y = y.unsqueeze(1)
        y = y.float().to(device)
        return self.loss_fn(pred, y)


def loss_score(losses):
    return (sum(losses)/len(losses)).item()


@no_grad()
def evaluate(model, dev_data):
    losses = []
    y_pred = []
    y_true = []
    
    for x,y in dev_data:
        out = model(x, y)
        losses.append(out['loss'])
        y_true.extend(y)
        y_pred.extend(round(x.item()) for x in out['probabilities'])
    return loss_score(losses), accuracy_score(y_true, y_pred)


def plot_history(train_values, dev_values, metric, title):
    plt.figure(figsize=(8,6))
    plt.plot(list(range(len(dev_values))),   dev_values,   label='Dev')
    plt.plot(list(range(len(train_values))), train_values, label='Train')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()



def train(model, optimizer, train_data, dev_data, epochs=5, early_stop=True, patience=5, verbose=True):

    train_loss_history = []
    train_accu_history = []
    dev_loss_history   = []
    dev_accu_history   = []
    curr_patience = patience

    best_accu = 0.66
    for epoch in range(epochs):
        losses = []
        y_pred = []
        y_true = []
        
        for x, y in train_data:
            optimizer.zero_grad()
            batch_out = model(x, y)
            loss = batch_out['loss']
            losses.append(loss)
            loss.backward()
            optimizer.step()

            y_true.extend(y)
            y_pred.extend(round(x.item()) for x in batch_out['probabilities'])

        model.global_epoch += 1
        train_loss = loss_score(losses)
        train_accu = accuracy_score(y_true, y_pred)
        dev_loss, dev_accu = evaluate(model, dev_data)

        #if (dev_accu > best_accu and dev_loss < best_loss) or dev_accu > 0.63:
        if dev_accu > best_accu:
            best_accu = dev_accu
            #best_loss = dev_loss    
            save(model.state_dict(), f"model/try3_model_{epoch}epoch_50d_{100*dev_accu:.2f}accu.pt")

        train_loss_history.append(train_loss)
        train_accu_history.append(train_accu)
        dev_loss_history.append(dev_loss)
        dev_accu_history.append(dev_accu)

        if verbose or epoch == epochs - 1:
            print(f"Epoch {model.global_epoch:3d}\tTrain vs Dev Loss: {train_loss:0.4f} --- {dev_loss:0.4f}")
            print(f"          \tTrain vs Dev Accu: {100*train_accu:.2f}% --- {100*dev_accu:.2f}%\n")

            if epoch > patience and early_stop:
                
                if dev_accu_history[-1] < dev_accu_history[-2] and dev_loss_history[-1] > dev_loss_history[-2]:
                    curr_patience -= 1
                    print(f" --> Patience ({curr_patience})")
                                        
                elif dev_accu_history[-1] > dev_accu_history[-2] and dev_loss_history[-1] < dev_loss_history[-2]:
                    if curr_patience < patience:
                        curr_patience += 1
                        print(f" --> Patience ({curr_patience})")
                
                if curr_patience == 0:
                    print(" [!] Early stop [!]")
                    break
                
    print("|---------------Summary---------------|")
    print(f"|-Best train loss  {    min(train_loss_history):0.4f}  @epoch {np.argmin(train_loss_history):3d}-|")
    print(f"|-Best dev   loss  {    min(  dev_loss_history):0.4f}  @epoch {np.argmin(  dev_loss_history):3d}-|")
    print(f"|-Best train accu  {100*max(train_accu_history):.2f}%  @epoch {np.argmax(train_accu_history):3d}-|")
    print(f"|-Best dev   accu  {100*max(  dev_accu_history):.2f}%  @epoch {np.argmax(  dev_accu_history):3d}-|")
    print("|-----------------End-----------------|")

    return {'train_loss_history' : train_loss_history,
            'train_accu_history' : train_accu_history,
            'dev_loss_history'   : dev_loss_history,
            'dev_accu_history'   : dev_accu_history}
