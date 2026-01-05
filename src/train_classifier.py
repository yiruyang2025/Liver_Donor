encoder.train()  # de-frozen encoder
loss = classification_loss(...)
loss.backward()
optimizer.step()
