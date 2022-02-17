import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
#from torch.cuda      import LongTensor, FloatTensor
from torch           import LongTensor, FloatTensor
from torch           import relu, sigmoid, no_grad, save, zeros
from torch.nn        import Module, Linear, BCELoss, LSTM, Dropout


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMClassifier(Module):

    def __init__(self, rnn_input_dim, drop_prob, lstm_aggregation, sentence_aggregation, double, bidirectional):
        super().__init__()
        if double:
            self.lstm1 = LSTM(input_size=rnn_input_dim,      
                              hidden_size=rnn_hidden_dim,    
                              bidirectional = bidirectional,         
                              num_layers=1,
                              dropout=0,   
                              batch_first=True)
            self.lstm2 = LSTM(input_size=rnn_input_dim,      
                              hidden_size=rnn_hidden_dim,    
                              bidirectional = bidirectional,         
                              num_layers=1,
                              dropout=0,   
                              batch_first=True)
        else:
            self.lstm  = LSTM(input_size=rnn_input_dim,      
                              hidden_size=rnn_input_dim,    
                              bidirectional = bidirectional, 
                              num_layers=1,                  #TODO more than 1
                              dropout=0,                     #TODO set to prob. only if multilayer
                              batch_first=True)
        
        
        if sentence_aggregation == 'concatenate':
            hidden_dim = 2*rnn_input_dim
        elif sentence_aggregation == 'subtract' or sentence_aggregation == 'abs-subtract':
            hidden_dim = rnn_input_dim
        
        if bidirectional and lstm_aggregation not in ['avg-avg', 'avg-avg-all']:
            hidden_dim = 2*hidden_dim
        
        if lstm_aggregation == 'concatenate':
            hidden_dim = 2*hidden_dim

        self.hidden_layer   = Linear(hidden_dim,    hidden_dim)        
        self.hidden_layer2  = Linear(hidden_dim,    hidden_dim//2)
        self.output_layer   = Linear(hidden_dim//2, 1)
        self.dropout_layer1 = Dropout(drop_prob)
        self.dropout_layer2 = Dropout(drop_prob)
        
        self.lstm_aggregation     = lstm_aggregation
        self.sentence_aggregation = sentence_aggregation
        
        self.loss_fn        = BCELoss()
        self.global_epoch   = 0
        self.double         = double
        self.rnn_shape      = (rnn_input_dim, hidden_dim)
        self.hidden_shape   = (hidden_dim,    hidden_dim)
        self.hidden_shape2  = (hidden_dim,    hidden_dim//2)
        self.output_shape   = (hidden_dim//2, 1)
        self.drop_prob      = drop_prob
        self.bidirectional  = bidirectional 
        self.print_summary()
        

    def print_summary(self):
        print(f">> LSTM   layer   shape {self.rnn_shape}")
        print(f">> Hidden layer 1 shape {self.hidden_shape}")
        print(f">> Dropout probability of {self.drop_prob}") if self.drop_prob > 0 else 0
        print(f">> Hidden layer 2 shape {self.hidden_shape2}")
        print(f">> Dropout probability of {self.drop_prob}") if self.drop_prob > 0 else 0
        print(f">> Output layer   shape {self.output_shape}")


    def forward(self, x1, l1, x2, l2, y=None, device=device):
        # output of shape (batch_size, seq_len, hidden_size)
        if self.double:
            lstm1_out  = self.lstm1(x1)[0]
            lstm2_out  = self.lstm2(x2)[0]
        else:
            lstm1_out  = self.lstm(x1)[0]
            lstm2_out  = self.lstm(x2)[0]

        batch_size1, _, _ = lstm1_out.shape
        batch_size2, _, _ = lstm2_out.shape
        assert batch_size1 == batch_size2
    
        out1 = []
        out2 = []
        for i in range(batch_size1):
            
            if not self.bidirectional:
                # I take only one output (the last one or the one corresponding to the target word)
                #  --> this is set in the prepare_rnn_data function 
                out1.append(lstm1_out[i][l1[i]])
                out2.append(lstm2_out[i][l2[i]])

            ## methods for aggregating the bidirectional outputs
            else:
                if self.lstm_aggregation == 'avg':                              # I take the average between the first and last outputs 
                    out1.append((lstm1_out[i][0] + lstm1_out[i][l1[i]])/2)
                    out2.append((lstm2_out[i][0] + lstm2_out[i][l2[i]])/2)
                
                elif self.lstm_aggregation == 'avg-avg':                        # I average the two components of the BiLSTM output (halving the size)
                                                                                # (which are automatically concatenated)
                                                                                # of the first and last outputs
                    size = lstm1_out[i][0].size()[0]//2
                    out1.append((lstm1_out[i][0][:size] + lstm1_out[i][0][size:] + lstm1_out[i][l1[i]][:size] + lstm1_out[i][l1[i]][size:])/4)
                    out2.append((lstm2_out[i][0][:size] + lstm2_out[i][0][size:] + lstm2_out[i][l2[i]][:size] + lstm2_out[i][l2[i]][size:])/4)

                elif self.lstm_aggregation == 'avg-all':                        # I take the average of all outputs (minus the padding)
                    aux = torch.zeros_like(lstm1_out[i][0], device=device)
                    for j in range(0, l1[i]):
                        aux += lstm1_out[i][j]
                    out1.append(aux / (l1[i]+1))   # (aux / l1[i] + 1)

                    aux = torch.zeros_like(lstm2_out[i][0], device=device)
                    for j in range(0, l2[i]):
                        aux += lstm2_out[i][j]
                    out2.append(aux / (l2[i]+1))   # (aux / l2[i] + 1)

                elif self.lstm_aggregation == 'avg-avg-all':                    # I average the two components of the BiLSTM output (halving the size)
                                                                                # (which are automatically concatenated)
                    size = lstm1_out[i][0].size()[0]//2                         # of all the outputs (these are also averaged)
                    aux = torch.zeros(size, device=device)
                    for j in range(0, l1[i]):
                        aux += (lstm1_out[i][j][:size] + lstm1_out[i][j][size:])/2 
                    out1.append(aux / (l1[i]+1))   # (aux / l1[i] + 1)

                    aux = torch.zeros(size, device=device)
                    for j in range(0, l2[i]):
                        aux += (lstm2_out[i][j][:size] + lstm2_out[i][j][size:])/2
                    out2.append(aux / (l2[i]+1))

                elif self.lstm_aggregation == 'subtract':                       # I subtract the last output from the first
                    out1.append((lstm1_out[i][0] - lstm1_out[i][l1[i]]))
                    out2.append((lstm2_out[i][0] - lstm2_out[i][l2[i]]))
                
                elif self.lstm_aggregation == 'abs-subtract':                   # I subtract the last output from the first and take the absolute values
                    out1.append(torch.absolute((lstm1_out[i][0] - lstm1_out[i][l1[i]])).to(device))
                    out2.append(torch.absolute((lstm2_out[i][0] - lstm2_out[i][l2[i]])).to(device))

                elif self.lstm_aggregation == 'concatenate':                    # I concatenate the first and last outputs (increase in size)
                    out1.append(torch.cat([lstm1_out[i][0], lstm1_out[i][l1[i]]]).to(device))
                    out2.append(torch.cat([lstm2_out[i][0], lstm2_out[i][l2[i]]]).to(device))
            
        # stack the vectors to get again a batch
        out1 = torch.stack(out1).to(device)
        out2 = torch.stack(out2).to(device)
        #print(out1.shape)
        
        ## methods for aggregating the sentences

        if self.sentence_aggregation == 'concatenate':
            out = torch.cat([out1, out2], dim=1)
        elif self.sentence_aggregation == 'subtract':
            out = FloatTensor(out1).to(device) - FloatTensor(out2).to(device)
        elif self.sentence_aggregation == 'abs-subtract':
            out = torch.absolute(FloatTensor(out1).to(device) - FloatTensor(out2).to(device)).to(device)

        pred = relu(self.hidden_layer(out))
        pred = self.dropout_layer1(pred)
        pred = relu(self.hidden_layer2(pred))
        pred = self.dropout_layer2(pred)
        pred = self.output_layer(pred).squeeze(1)
        prob = torch.sigmoid(pred)
        result = {'predictions': pred, 'probabilities': prob}
        if y is not None:
            loss = self.loss(prob, y)
            result['loss'] = loss
        return result


    def loss(self, pred, y):
        #y = y.unsqueeze(1)
        y = y.float()
        return self.loss_fn(pred, y)
    

def loss_score(losses):
    return (sum(losses)/len(losses)).item()


@no_grad()
def evaluate(model, dev_data):
    losses = []
    y_pred = []
    y_true = []
    
    for x0, l0, x1, l1, ys in dev_data:
        out = model(x0, l0, x1, l1, ys)
        losses.append(out['loss'])
        y_true.extend(ys.cpu())
        y_pred.extend(round(y.cpu().item()) for y in out['probabilities'])
        conf_mat = confusion_matrix(y_true, y_pred) 
        print(f"Confusion Matrix on dev data:\n{conf_mat}")
    return loss_score(losses), accuracy_score(y_true, y_pred)


def train_rnn(model, optimizer, train_data, dev_data, epochs=5, early_stop=True, patience=5, verbose=True):

    train_loss_history = []
    train_accu_history = []
    dev_loss_history   = []
    dev_accu_history   = []
    curr_patience = patience

    best_accu = 0.69
    for epoch in range(epochs):
        losses = []
        y_pred = []
        y_true = []

        for x0, l0, x1, l1, ys in train_data:
            optimizer.zero_grad()
            batch_out = model(x0, l0, x1, l1, ys)
            loss = batch_out['loss']
            losses.append(loss)
            loss.backward()
            optimizer.step()

            y_true.extend(ys.cpu())
            y_pred.extend(round(y.cpu().item()) for y in batch_out['probabilities'])

        model.global_epoch += 1
        train_loss = loss_score(losses)
        train_accu = accuracy_score(y_true, y_pred)
        if dev_data is not None:
            dev_loss, dev_accu = evaluate(model, dev_data)

        #if (dev_accu > best_accu and dev_loss < best_loss) or dev_accu > 0.63:
        if dev_accu >= best_accu:
            best_accu = dev_accu
            #best_loss = dev_loss    
            save(model.state_dict(), f"model/try1_part2_{epoch}epoch_50d_{100*dev_accu:.2f}accu.pt")

        train_loss_history.append(train_loss)
        train_accu_history.append(train_accu)
        if dev_data is not None:
            dev_loss_history.append(dev_loss)
            dev_accu_history.append(dev_accu)

        if verbose or epoch == epochs - 1:
            if dev_data is not None:
                print(f"Epoch {model.global_epoch:3d}\tTrain vs Dev Loss: {train_loss:0.4f} --- {dev_loss:0.4f}")
                print(f"          \tTrain vs Dev Accu: {100*train_accu:.2f}% --- {100*dev_accu:.2f}%\n")
            else:
                print(f"Epoch {model.global_epoch:3d}\tTrain Loss: {train_loss:0.4f}")
                print(f"          \tTrain Accu: {100*train_accu:.2f}%\n")

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
                
    print( "|---------------Summary---------------|")
    print(f"|-Best train loss  {    min(train_loss_history):0.4f}  @epoch {np.argmin(train_loss_history):3d}-|")
    print(f"|-Best dev   loss  {    min(  dev_loss_history):0.4f}  @epoch {np.argmin(  dev_loss_history):3d}-|") if dev_data is not None else 0
    print(f"|-Best train accu  {100*max(train_accu_history):.2f}%  @epoch {np.argmax(train_accu_history):3d}-|")
    print(f"|-Best dev   accu  {100*max(  dev_accu_history):.2f}%  @epoch {np.argmax(  dev_accu_history):3d}-|") if dev_data is not None else 0
    print( "|-----------------End-----------------|")

    return {'train_loss_history' : train_loss_history,
            'train_accu_history' : train_accu_history,
            'dev_loss_history'   : dev_loss_history,
            'dev_accu_history'   : dev_accu_history}




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

