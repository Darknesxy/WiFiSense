import subprocess
import re
import csv
import time
from datetime import datetime

filename = "wifi_signal_data.csv"

print("WiFiSense - Signal Logger")
print("Recording Wi-Fi signal every 2 seconds.")
print("Press Ctrl + C to stop.\n")

with open(filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Signal_Percent", "RSSI_dBm"])

    try:
        while True:
            result = subprocess.run(
                ["netsh", "wlan", "show", "interfaces"],
                capture_output=True,
                text=True
            )

            signal = re.search(r"Signal\s*:\s*(\d+)%", result.stdout)
            rssi = re.search(r"Rssi\s*:\s*(-?\d+)", result.stdout)

            if signal:
                signal_value = int(signal.group(1))
                rssi_value = int(rssi.group(1)) if rssi else "N/A"

                timestamp = datetime.now().strftime("%H:%M:%S")

                writer.writerow([
                    timestamp,
                    signal_value,
                    rssi_value
                ])
                file.flush()

                print(
                    f"{timestamp} | Signal: {signal_value}% "
                    f"| RSSI: {rssi_value} dBm"
                )
            else:
                print("Wi-Fi signal information unavailable.")

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nRecording stopped.")
        print(f"Data saved in {filename}")
