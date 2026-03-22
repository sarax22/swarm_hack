import socket

host = '192.168.0.120'
port = 80

def recv_line(sock):
    """Read exactly one line from the socket"""
    data = b''
    while True:
        byte = sock.recv(1)
        if not byte:
            raise ConnectionError("Socket closed")
        if byte == b'\n':
            return data.decode().strip()
        data += byte

def send_command(sock, packet):
    sock.sendall(packet.encode())
    response = recv_line(sock)
    return response

def start_client():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.settimeout(20)
            print("Connected to Mona Buggy!")
            print("Commands: w[dist_mm], s[dur_ms], a[angle], d[angle]")
            print("Type 'exit' to quit.")

            cmd_map = {'w': 'F', 's': 'B', 'a': 'L', 'd': 'R'}

            while True:
                user_input = input("Enter Command: ").strip().lower()

                if user_input == 'exit':
                    break

                if len(user_input) < 2:
                    print("Invalid format. Use Letter+Number (e.g., w100)")
                    continue

                char = user_input[0]
                val = user_input[1:]

                if char in cmd_map and val.isdigit():
                    packet = f"{cmd_map[char]}{val}\n"
                    print(f"Sending: {packet.strip()} ...")
                    try:
                        response = send_command(s, packet)
                        print(f"Robot: {response}")
                    except socket.timeout:
                        print("Timeout — no response from robot")
                else:
                    print("Error: Use w, a, s, or d followed by a number.")

    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    start_client()
