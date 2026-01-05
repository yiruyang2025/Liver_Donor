def train_ssl(ssl_encoder, train_loader, ssl_objective, epochs=100):
    for epoch in range(epochs):
        for batch in train_loader:
            x1, x2 = augment(batch)  # data augment
            z1 = ssl_encoder(x1)
            z2 = ssl_encoder(x2)
            loss = ssl_objective(z1, z2)  # contrastive loss here
            loss.backward()
            optimizer.step()

