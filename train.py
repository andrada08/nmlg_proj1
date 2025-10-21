import torch
import torch.nn as nn
import torch.optim as optim

def train_with_gradient_tracking(model, trainloader, testloader, 
                                epochs, ln_rate,
                                optimizer, device,
                                layer_lns: dict | None = None,
):
    
    model = model.to(device)
    
    if layer_lns:
        groups = [
            {'params': getattr(model, name).parameters(), 'lr': lr}
            for name, lr in layer_lns.items()
        ]
        opt = optimizer(groups)  # per-group lrs set above
    else:
        opt = optimizer(model.parameters(), lr=ln_rate)
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
        },
        'gradient_metrics': {
            'layer1_above_layer2': 0,
            'layer2_above_layer1': 0,
            'layer1_above_layer3': 0,
            'layer3_above_layer1': 0,
            'layer2_above_layer3': 0,
            'layer3_above_layer2': 0,
            'switches_12': 0,
            'switches_13': 0,
            'switches_23': 0,
            'layer1_large_drop': 0,
            'layer2_large_drop': 0,
            'layer3_large_drop': 0,
            'layer1vslayer2_pattern': 0,
            'layer1vslayer3_pattern': 0,
            'layer2vslayer3_pattern': 0,
            'layer2vslayer1_pattern': 0,
            'layer3vslayer1_pattern': 0,
            'layer3vslayer2_pattern': 0
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
            
            opt.zero_grad()
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

            opt.step()

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
    
    # Compute gradient comparison metrics after training
    if (len(history['gradients']['layer1']) > 0 and 
        len(history['gradients']['layer2']) > 0 and 
        len(history['gradients']['layer3']) > 0):
        
        layer1_grads = history['gradients']['layer1']
        layer2_grads = history['gradients']['layer2']
        layer3_grads = history['gradients']['layer3']
        
        # Pairwise comparisons
        history['gradient_metrics']['layer1_above_layer2'] = 1 if any(l1 > l2 for l1, l2 in zip(layer1_grads, layer2_grads)) else 0
        history['gradient_metrics']['layer2_above_layer1'] = 1 if any(l2 > l1 for l1, l2 in zip(layer1_grads, layer2_grads)) else 0
        history['gradient_metrics']['layer1_above_layer3'] = 1 if any(l1 > l3 for l1, l3 in zip(layer1_grads, layer3_grads)) else 0
        history['gradient_metrics']['layer3_above_layer1'] = 1 if any(l3 > l1 for l1, l3 in zip(layer1_grads, layer3_grads)) else 0
        history['gradient_metrics']['layer2_above_layer3'] = 1 if any(l2 > l3 for l2, l3 in zip(layer2_grads, layer3_grads)) else 0
        history['gradient_metrics']['layer3_above_layer2'] = 1 if any(l3 > l2 for l2, l3 in zip(layer2_grads, layer3_grads)) else 0
        
        # Switches between pairs
        switches_12 = sum(1 for i in range(1, len(layer1_grads)) 
                         if (layer1_grads[i-1] > layer2_grads[i-1]) != (layer1_grads[i] > layer2_grads[i]))
        switches_13 = sum(1 for i in range(1, len(layer1_grads)) 
                         if (layer1_grads[i-1] > layer3_grads[i-1]) != (layer1_grads[i] > layer3_grads[i]))
        switches_23 = sum(1 for i in range(1, len(layer2_grads)) 
                         if (layer2_grads[i-1] > layer3_grads[i-1]) != (layer2_grads[i] > layer3_grads[i]))
        
        history['gradient_metrics']['switches_12'] = 1 if switches_12 > 0 else 0
        history['gradient_metrics']['switches_13'] = 1 if switches_13 > 0 else 0
        history['gradient_metrics']['switches_23'] = 1 if switches_23 > 0 else 0
        
        # Large relative drops for all layers
        drop_threshold = 0.5  # 50% drop
        l1_drops = [(layer1_grads[i-1] - layer1_grads[i]) / layer1_grads[i-1] 
                    for i in range(1, len(layer1_grads)) if layer1_grads[i-1] > 0]
        l2_drops = [(layer2_grads[i-1] - layer2_grads[i]) / layer2_grads[i-1] 
                    for i in range(1, len(layer2_grads)) if layer2_grads[i-1] > 0]
        l3_drops = [(layer3_grads[i-1] - layer3_grads[i]) / layer3_grads[i-1] 
                    for i in range(1, len(layer3_grads)) if layer3_grads[i-1] > 0]
        
        history['gradient_metrics']['layer1_large_drop'] = 1 if any(drop > drop_threshold for drop in l1_drops) else 0
        history['gradient_metrics']['layer2_large_drop'] = 1 if any(drop > drop_threshold for drop in l2_drops) else 0
        history['gradient_metrics']['layer3_large_drop'] = 1 if any(drop > drop_threshold for drop in l3_drops) else 0
        
        # Combined patterns for all layer pairs: layerA>layerB then layerB>layerA AND layerA has large drop
        def check_pattern(layerA_grads, layerB_grads, layerA_drops, switches_AB, layerA_large_drop):
            pattern = 0
            if switches_AB > 0 and layerA_large_drop:
                for i in range(1, len(layerA_grads)):
                    if (layerA_grads[i-1] > layerB_grads[i-1] and 
                        layerB_grads[i] > layerA_grads[i] and
                        layerA_drops[i-1] > drop_threshold):
                        pattern = 1
                        break
            return pattern
        
        history['gradient_metrics']['layer1vslayer2_pattern'] = check_pattern(
            layer1_grads, layer2_grads, l1_drops, switches_12, history['gradient_metrics']['layer1_large_drop'])
        history['gradient_metrics']['layer1vslayer3_pattern'] = check_pattern(
            layer1_grads, layer3_grads, l1_drops, switches_13, history['gradient_metrics']['layer1_large_drop'])
        history['gradient_metrics']['layer2vslayer3_pattern'] = check_pattern(
            layer2_grads, layer3_grads, l2_drops, switches_23, history['gradient_metrics']['layer2_large_drop'])
        history['gradient_metrics']['layer2vslayer1_pattern'] = check_pattern(
            layer2_grads, layer1_grads, l2_drops, switches_12, history['gradient_metrics']['layer2_large_drop'])
        history['gradient_metrics']['layer3vslayer1_pattern'] = check_pattern(
            layer3_grads, layer1_grads, l3_drops, switches_13, history['gradient_metrics']['layer3_large_drop'])
        history['gradient_metrics']['layer3vslayer2_pattern'] = check_pattern(
            layer3_grads, layer2_grads, l3_drops, switches_23, history['gradient_metrics']['layer3_large_drop'])
    
    return history


