import habana_frameworks.torch.hpu as hthpu

def get_device_name():

    device_name = hthpu.get_device_name()
    print(f"Device name: {device_name}")
    return device_name

if __name__ == "__main__":
    get_device_name()

