import socket


# Update with your Mona ESP's IP
host = '192.168.0.120' 
port = 80

def start_client():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            print("Connected to Mona Buggy!")
            print("Commands: w[dist], s[dist], a[angle], d[angle] (e.g., 'w50' then Enter)")
            print("Type 'exit' to quit.")

            while True:
                user_input = input("Enter Command: ").strip().lower()
                
                if user_input == 'exit':
                    break
                
                if len(user_input) < 2:
                    print("Invalid format. Use Letter+Number (e.g., w100)")
                    continue

                # Map keyboard keys to the Arduino protocol letters
                cmd_map = {'w': 'F', 's': 'B', 'a': 'L', 'd': 'R'}
                char = user_input[0]
                val = user_input[1:]

                if char in cmd_map and val.isdigit():
                    # Send as "F50\n"
                    packet = f"{cmd_map[char]}{val}\n"
                    s.sendall(packet.encode())
                else:
                    print("Error: Use w, a, s, or d followed by a number.")

    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    start_client()
