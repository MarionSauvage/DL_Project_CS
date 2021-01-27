def evaluate_model(model, dataloader):
    avg_accuracy = 0
    avg_loss = 0.0

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)

            loss = criterion(out, target)
            _, predictions = torch.max(out, 1)
            nb_correct = torch.sum(predictions == targets)

            avg_loss += loss.item()
            avg_accuracy += nb_correct
    
    return avg_loss / len(dataloader.dataset), float(avg_accuracy) / len(dataloader.dataset)


def train_classification(model,train_dataloader, val_dataloader, optimizer, criterion, epochs=20):
    # defining the optimizer
    optimizer = SGD(model.parameters(), lr=0.05)
    # defining the loss function
    criterion = CrossEntropyLoss()
    for epoch in range(epochs):
    # training
        for batch_idx, (x, target) in enumerate(train_dataloader):
            model.train()

            optimizer.zero_grad()
            x, target = Variable(x), Variable(target)
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 20 == 0:
                val_loss, accuracy = evaluate_model(model, val_dataloader)
                print('epoch {} batch {} [{}/{}] training loss: {:1.4f} \tvalidation loss: {:1.4f}\tAccuracy (val): {:.1%}'.format(epoch,batch_idx,batch_idx*len(x),
                        len(train_dataloader.dataset),loss.item(), val_loss, accuracy))
