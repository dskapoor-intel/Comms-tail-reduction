import socket
import sys

def get_ip_address():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

if __name__ == "__main__":
    ip_address = get_ip_address()
    print(ip_address)  # Optionally print the IP address

    # Write the IP address to a text file
    mypath = sys.argv[1]
    myfile = f"{mypath}/ip_address.txt"
    with open(myfile, "w") as file:
        file.write(ip_address)

    print("IP address written to ip_address.txt")
