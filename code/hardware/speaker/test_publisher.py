import redis
import json
import time

# --- Configuration ---

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
COMMAND_CHANNEL = "audio:playcommand"

# !!! IMPORTANT: Replace this path with your actual WAV file path !!!
TEST_WAV_FILE = "/home/linaro/renziang_space/voice/test_data/record (1).wav"

def send_command(redis_conn, channel, command_dict):
    """Converts a command dictionary to JSON and publishes it to a Redis channel."""
    message = json.dumps(command_dict)
    print(f"--> Sending command to channel '{channel}': {message}")
    redis_conn.publish(channel, message)

def run_test():
    """Executes a series of test commands."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping() # Check connection
        print("✅ Successfully connected to Redis.")
    except redis.exceptions.ConnectionError as e:
        print(f"❌ Could not connect to Redis: {e}")
        print("Please ensure your Redis server is running.")
        return
    time.sleep(10)
    print("\n--- Starting Command Sequence ---")

    # 1. Play a WAV file
    print("\nSending 'play' command...")
    send_command(r, COMMAND_CHANNEL, {
        "action": "play",
        "path": TEST_WAV_FILE
    })
    time.sleep(10) # Wait for a moment to let it play

    # 2. Pause playback
    print("\nSending 'stop' command...")
    send_command(r, COMMAND_CHANNEL, {
        "action": "stop"
    })
    time.sleep(2) # Wait for a moment

    
    print("\n--- Command Sequence Finished ---")

if __name__ == "__main__":
    run_test()