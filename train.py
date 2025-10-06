import torch
import torch.nn as nn

def train_with_gradient_tracking(model, trainloader, testloader, 
                                epochs, ln_rate,
                                optimizer, device):
    
    model = model.to(device)
    optimizer = optimizer(model.parameters(), lr=ln_rate)
    criterion = nn.CrossEntropyLoss()

    history = {
        'loss': [],
        'accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
        'gradients': {
            'epoch': [],
            'layer1': [],
            'layer2': [],
            'layer3': []
        }
    }

    layer_names = model.get_layer_names()

    for epoch in range(epochs):
        first_batch = True

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            if first_batch:
                gradient_norms = {}
                for layer_name in layer_names:
                    layer = getattr(model, layer_name)
                    if layer.weight is not None:
                        grad_norm = layer.weight.grad.norm().item()
                        gradient_norms[layer_name] = grad_norm
                # store gradients
                for layer_name in layer_names:
                    if layer_name in gradient_norms:
                        history['gradients'][layer_name].append(gradient_norms[layer_name])
                history['gradients']['epoch'].append(epoch+1)
                first_batch = False
                print("logged grads for epoch ", epoch + 1)

            optimizer.step()

            # track metrics
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)

        # calculate metrics
        train_loss /= len(trainloader)
        train_accuracy = 100.0 * train_correct / train_total

        # evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += target.size(0)
        test_loss /= len(testloader)
        test_accuracy = 100.0 * test_correct / test_total

        # store metrics
        history['loss'].append(train_loss)
        history['accuracy'].append(train_accuracy)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    return history


