import torch
NITER = 1000
LEARNING_RATE = 0.0001

A = 3
B = 5
N = 100

# Generate some dummy data and targets
data = torch.arange(N, dtype=torch.float) / 100
noise = torch.randn(N) / 10000
targets = A * data + B + noise

# Loss function: mean squared error loss
def compute_loss(output, target):
    diff = output - target
    diffsq = diff.pow(2)
    loss = diffsq.sum()
    return loss

# Model the data as a linear function
def model(data, paramA, paramB):
    prod = data * paramA
    output = prod + paramB
    return result

# We're trying to fit `data` to `targets`.
# These are the two parameters for our model.
paramA = torch.tensor(1., requires_grad=True)
paramB = torch.tensor(0., requires_grad=True)


# Training loop
for i in range(1, NITER + 1):
    # Run forward pass
    output = model(data, paramA, paramB)
    loss = compute_loss(output, targets)

    # Compute gradients
    loss.backward()

    # Update the parameters via SGD.
    with torch.no_grad():
        paramA -= paramA.grad * LEARNING_RATE
        paramB -= paramB.grad * LEARNING_RATE

    # zero grads so that we can keep training
    with torch.no_grad():
        paramA.grad.zero_()
        paramB.grad.zero_()

    if i % 50 == 0:
        print('[{}/{}][Loss: {}][learned A: {}][learned B: {}]'.format(
            i, NITER, loss.item(), paramA.item(), paramB.item()))
