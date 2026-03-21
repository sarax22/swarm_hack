import socket

# Replace with the IP address you saw in the Arduino Serial Monitor
host = '192.168.0.120' 
port = 80

def send_command(command):
    try:
        # Create a socket connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall(command.encode())
            print(f"Sent: {command}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Controls: W=Forward, S=Backward, A=Left, D=Right, Q=Quit")
    while True:
        move = input("Enter command: ").upper() #just use send _command('F'/'B'/'L'/'R')
        if move == 'W': send_command('F')
        elif move == 'S': send_command('B')
        elif move == 'A': send_command('L')
        elif move == 'D': send_command('R')
        elif move == 'Q': break
        else: print("Invalid Key")
          
