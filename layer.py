class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None
    def forward(self,input):
        pass
    def backward(self,output_gradient, alpha):
        pass