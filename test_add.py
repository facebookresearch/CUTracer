import torch


def test_tensor_addition_on_gpu():
    device = torch.device("cuda")

    a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
    b = torch.tensor([2, 3, 4], dtype=torch.float32, device=device)

    print("Tensor A:", a)
    print("Tensor B:", b)
    for i in range(3):  
        a = a + b

    print("Result (A + B):", a)

    return a


if __name__ == "__main__":
    test_tensor_addition_on_gpu()
