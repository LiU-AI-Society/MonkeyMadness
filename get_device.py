import torch

class DeviceManager:
    @staticmethod
    def get_device():
        """
        Determines the optimal device for computation.
        
        Returns:
            torch.device: The best available device (CUDA > MPS > CPU)
        """
        # Check if CUDA (GPU) is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)} ✅")
            return device
        
        # Check if MPS (Metal Performance Shaders for macOS) is available
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✅ Using MPS (Apple GPU) ✅")
            return device
        
        # Fallback to CPU if neither CUDA nor MPS is available
        else:
            device = torch.device("cpu")
            print("💀s Using CPU (No GPU available) 💀")
            return device

    @staticmethod
    def print_device_info(device):
        """
        Print detailed information about the selected device.
        
        Args:
            device (torch.device): The device to get information about
        """
        if device.type == 'cuda':
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            print(f"CUDA Current Device: {torch.cuda.current_device()}")
        elif device.type == 'mps':
            print("Using Apple Metal Performance Shaders (MPS)")
        else:
            print("Using CPU")

    @staticmethod
    def is_gpu_available():
        """
        Check if a GPU is available.
        
        Returns:
            bool: True if CUDA or MPS is available, False otherwise
        """
        return torch.cuda.is_available() or torch.backends.mps.is_available()